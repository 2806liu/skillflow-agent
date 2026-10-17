"""S3 Hook 策略：请求预算、输入约束、审计与完成门；不包含数据库或 MCP SDK。"""

import asyncio
from time import perf_counter

from .budget import BudgetExceeded
from .context import checked_context
from .contracts import RuntimeEvent
from .finish import inspect_goal


class ModelCallError(Exception):
    """只向图传递稳定错误码，不泄露外部服务异常正文或密钥。"""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


class RuntimeHooks:
    """可替换的运行约束；图负责路由，Hook 负责边界策略。"""

    async def record(self, state, context, node, status, **details):
        """所有 Hook 和模型事件复用同一审计端口，工具边界也写入该端口。"""
        code = details.pop("error_code", None)
        duration = details.pop("duration_ms", 0)
        await context.events.append(RuntimeEvent(metadata=state.metadata, node=node, status=status,
            budget=context.budget.snapshot(), error_code=code, duration_ms=duration, details={"stage": getattr(context,"stage","S3"), **details}))

    async def save(self, state, context):
        """图保存权威状态；明确只是运行诊断，S4 才提供执行图恢复。"""
        state.budget = context.budget.snapshot()
        if context.session:
            state.loaded_skills = {k: v.version for k, v in context.session.loaded.items()}
            state.observations = [v.model_copy(deep=True) for v in context.session.observations]
        state.events = list(context.events.items)
        await context.tasks.save_state(state)

    async def before_model(self, state, context, purpose, messages=None, specs=None):
        """所有真实请求先占预算；上下文不裁断工具配对，也不允许无限增长。"""
        size = checked_context(messages or [], specs or [], context.max_context_chars)
        context.budget.reserve_model(decision=purpose == "tool_decision")
        await self.record(state, context, "before_model", "started", purpose=purpose,
                          context_chars=size, offered_tools=[spec.name for spec in specs or []])
        await self.save(state, context)

    async def model_call(self, state, context, purpose, call, messages=None, specs=None):
        """模型异常、单次超时与累计超时分别终止；不做隐藏重试或规则回退。"""
        await self.before_model(state, context, purpose, messages, specs)
        started = perf_counter()
        try:
            value = await asyncio.wait_for(call(), min(state.metadata.limits.model_timeout_seconds,
                                                        context.budget.remaining_seconds()))
            context.budget.remaining_seconds()
        except BudgetExceeded:
            raise
        except ModelCallError:
            raise
        except asyncio.TimeoutError as exc:
            code = "TASK_TIMEOUT" if context.budget.snapshot().active_seconds >= state.metadata.limits.active_timeout_seconds else "MODEL_TIMEOUT"
            raise ModelCallError(code) from exc
        except Exception as exc:
            # 参数不支持属于能力失败；其他原文只留在异常链，不写用户 Trace。
            code = ("MODEL_TIMEOUT" if exc.__class__.__name__ in {"APITimeoutError", "ReadTimeout", "ConnectTimeout"}
                    else "MODEL_CAPABILITY_UNSUPPORTED" if getattr(exc, "status_code", None) == 400 else "MODEL_FAILED")
            raise ModelCallError(code) from exc
        await self.record(state, context, "model", "returned", purpose=purpose,
                          duration_ms=(perf_counter() - started) * 1000)
        return value

    async def finish_gate(self, state, context):
        """验收条件取自固定目标；先检查当前事实，再检查必要工具收据。"""
        await context.session.verify_current()
        finding = inspect_goal(context.goal, state.constraints.get("tool_records", []),
                               logistics_enabled=getattr(context.tools, 'logistics_enabled', False))
        await self.record(state, context, "finish_gate", "incomplete" if finding.missing else finding.status,
                          missing=finding.missing, result_refs=finding.refs, stop_reason=finding.reason)
        return finding

    async def before_tool(self, state, context, call):
        """调用前保存图消息；参数、归属、依赖和范围由复用的 S2 执行边界强制校验。"""
        # Hook 不能修改模型实参或为模型补身份；业务执行器随后先占工具预算再校验执行。
        await self.save(state, context)

    async def after_tool(self, state, context, call, observation):
        """记录原调用 ID 与完整观察；模型只能依据该观察作下一轮决定。"""
        await self.record(state, context, "tool_observation", observation.status, tool=call.name,
                          tool_call_id=call.call_id, result=observation.model_dump(mode="json"))
