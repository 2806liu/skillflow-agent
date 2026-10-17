"""S3 应用服务：统一任务归属、模型图、澄清持久化和独立只读结果投影。"""

import asyncio
from langgraph.errors import NodeCancelledError

from .budget import BudgetTracker
from .contracts import AgentState, MessageEnvelope, RuntimeEvent
from .loop_graph import LoopContext, build_loop_graph

S3_ENGINE_VERSION = "tool_loop_s3_context_v1"
S3_GRAPH_VERSION = "native_tool_loop_context_v1"


class CollectedEvents:
    """每 run 收集本次图与工具事件，底层仍先持久化再推送 SSE。"""

    def __init__(self, sink):
        self.sink, self.items = sink, []

    async def append(self, event):
        """失败不伪装为已审计；不同请求没有共享可变列表。"""
        await self.sink.append(event)
        self.items.append(event)


class ToolLoopService:
    """持有无会话状态的依赖；实际工具授权与证据仅存在每次图运行上下文。"""

    def __init__(self, model, tasks, events, tools, conversations, max_context_chars=100000):
        self.model, self.tasks, self.events = model, tasks, events
        self.tools, self.conversations = tools, conversations
        self.max_context_chars = max_context_chars
        self.graph = build_loop_graph()

    async def run(self, run_id, buyer_id, session_id, message):
        """只执行登记过的 S3 任务；明确中断原因，后续 S4 才接执行图恢复。"""
        metadata = await self.tasks.get_owned(run_id, buyer_id, session_id)
        if (metadata.engine_version, metadata.graph_version, metadata.model_mode) != (
                S3_ENGINE_VERSION, S3_GRAPH_VERSION, "langchain_openai"):
            raise ValueError("ENGINE_VERSION_UNSUPPORTED")
        state = AgentState(metadata=metadata, goal=message, messages=[MessageEnvelope(role="user", content=message)])
        budget = BudgetTracker(metadata.limits)
        events = CollectedEvents(self.events)
        conversation = await self.conversations.load(metadata)
        context = LoopContext(self.model, budget, events, self.tasks, self.tools, conversation, self.max_context_chars,
                              context_store=self.conversations)
        budget.start()
        try:
            state = AgentState.model_validate(await self.graph.ainvoke(state, context=context, config={"recursion_limit": 64}))
            if state.status in {"created", "running"}:
                # 某些图内取消只结束 superstep，不向应用抛取消；不能把非终态返回为成功响应。
                state = getattr(context, "current_state", state)
                state.status, state.stop_reason = "failed", "RUN_INTERRUPTED"
                state.answer = "本次调查已中断，未创建工单或执行退款。可重新提供订单号继续已确认的调查目的。"
        except asyncio.CancelledError:
            state = getattr(context, "current_state", state)
            state.status, state.stop_reason = "failed", "RUN_INTERRUPTED"
            await asyncio.shield(context.hooks.save(state, context))
            raise
        except NodeCancelledError:
            # LangGraph 将节点内部取消包装成专用异常；与外部取消使用同一终止原因。
            state = getattr(context, "current_state", state)
            state.status, state.stop_reason = "failed", "RUN_INTERRUPTED"
            state.answer = "本次调查已中断，未创建工单或执行退款。可重新提供订单号继续已确认的调查目的。"
        except Exception:
            # 未预期异常归类基础设施失败，不把异常原文或模型回答当成事实。
            state = getattr(context, "current_state", state)
            state.status, state.stop_reason = "failed", "RUNTIME_FAILED"
            state.answer = "本次只读调查因运行异常未完成，未创建工单或执行退款。"
        finally:
            budget.pause()
        state.events = list(events.items)
        state.budget = budget.snapshot()
        await events.append(RuntimeEvent(metadata=metadata, node="finish", status=state.status,
            budget=state.budget, error_code=state.stop_reason if state.status == "failed" else None,
            details={"stage": "S3", "stop_reason": state.stop_reason}))
        await context.hooks.save(state, context)
        await self.conversations.save_outcome(state)
        return state


def response_projection(state):
    """网页展示实际工具记录；事实、来源和模型模式均带明确标记。"""
    records = state.constraints.get("tool_records", [])
    citations = []
    for record in records:
        if record["name"] == "retrieve_policy":
            for evidence in record["observation"]["data"].get("evidence", []):
                citations.append({**{k: v for k, v in evidence.items() if k != "ref"},
                                  "citation_number": len(citations) + 1})
    return citations, {"execution_mode": "real_model", "read_only": True,
        "business_completed": state.status == "completed", "metadata": state.metadata.model_dump(mode="json"),
        "stop_reason": state.stop_reason, "budget": state.budget.model_dump(),
        "task_goal": state.constraints.get("task_goal"), "loaded_skills": state.loaded_skills,
        "conversation_context": state.constraints.get("conversation_context"),
        "tool_records": records, "result_refs": state.constraints.get("result_refs", []),
        "trusted_facts": state.constraints.get("trusted_facts", []),
        "token_usage": {"scope": "tool_decision_only", "counts": state.constraints.get("token_usage", {})}}
