"""S1 最小执行图：初始化、脚本模型、终止；依赖端口，不接触数据库或 MCP。"""

import asyncio
from dataclasses import dataclass
from time import perf_counter

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from .budget import BudgetExceeded, BudgetTracker
from .contracts import AgentState, ModelReply, RuntimeEvent, TaskMetadata
from .ports import EventSink, ModelGateway, TaskStore


READY_ANSWER = "运行时骨架验证完成，尚未执行订单查询或售后操作。"
FAILED_ANSWER = "运行时验证未完成，未执行订单查询或售后操作。"


@dataclass(frozen=True)
class RunContext:
    """运行依赖由应用层注入，不进入可序列化图状态。"""

    metadata: TaskMetadata
    model: ModelGateway
    budget: BudgetTracker
    events: EventSink
    tasks: TaskStore


async def event_update(state: AgentState, context: RunContext, node: str, status: str, error_code=None, duration_ms=0):
    """事件先持久化再返回图更新；持久化失败不能伪装为成功。"""
    event = RuntimeEvent(metadata=context.metadata, node=node, status=status,
                         budget=context.budget.snapshot(), error_code=error_code, duration_ms=duration_ms)
    await context.events.append(event)
    return {"events": [*state.events, event], "budget": context.budget.snapshot()}


async def initialize(state: AgentState, runtime: Runtime[RunContext]):
    """服务端上下文与状态一致后进入模型节点，不从消息推断身份。"""
    if state.metadata != runtime.context.metadata:
        raise ValueError("运行上下文与任务归属不一致")
    update = await event_update(state, runtime.context, "initialize", "running")
    return {**update, "status": "running"}


async def model_node(state: AgentState, runtime: Runtime[RunContext]):
    """调用前占预算并保存诊断快照；S1 不接受工具执行或真实业务回答。"""
    context = runtime.context
    try:
        context.budget.reserve_model()
    except BudgetExceeded as exc:
        return {"status": "failed", "stop_reason": exc.code,
                **await event_update(state, context, "model", "blocked", exc.code)}

    update = await event_update(state, context, "model", "started")
    # 此快照证明已占用的调用预算；不是可恢复的 LangGraph 检查点。
    await context.tasks.save_state(AgentState.model_validate({
        **state.model_dump(), **update, "budget": context.budget.snapshot().model_dump(),
    }))
    error_code = None
    reply = None
    model_started = perf_counter()
    try:
        remaining = context.budget.remaining_seconds()
        timeout = min(context.metadata.limits.model_timeout_seconds, remaining)
        reply = ModelReply.model_validate(await asyncio.wait_for(
            context.model.generate(state.messages, []), timeout=timeout))
        context.budget.remaining_seconds()
        if reply.message.role != "assistant":
            error_code = "MODEL_INVALID_RESPONSE"
        elif reply.message.tool_calls:
            error_code = "TOOL_EXECUTION_NOT_READY"
    except BudgetExceeded as exc:
        error_code = exc.code
    except asyncio.TimeoutError:
        # 按实际先耗尽的保护预算区分整任务与单次请求超时。
        error_code = ("TASK_TIMEOUT" if context.budget.snapshot().active_seconds >=
                      context.metadata.limits.active_timeout_seconds else "MODEL_TIMEOUT")
    except Exception:
        # 不把异常原文、模型请求或密钥写入 Trace 和用户回答。
        error_code = "MODEL_FAILED"

    model_state = AgentState.model_validate({**state.model_dump(), **update})
    finished = await event_update(model_state, context, "model", "failed" if error_code else "returned", error_code,
                                  duration_ms=(perf_counter() - model_started) * 1000)
    result = {**finished, "stop_reason": error_code, "status": "failed" if error_code else "running"}
    if reply is not None and error_code is None:
        result["messages"] = [*state.messages, reply.message]
    return result


async def finish(state: AgentState, runtime: Runtime[RunContext]):
    """运行时验证成功与业务完成明确区分；即使脚本输出业务文本也不直接展示。"""
    reason = state.stop_reason
    if state.status != "failed":
        try:
            runtime.context.budget.remaining_seconds()
        except BudgetExceeded as exc:
            reason = exc.code
    failed = state.status == "failed" or reason is not None
    status = "failed" if failed else "runtime_ready"
    reason = reason if failed else "S1_SKELETON_ONLY"
    return {"status": status, "stop_reason": reason,
            "answer": FAILED_ANSWER if failed else READY_ANSWER,
            **await event_update(state, runtime.context, "finish", status, reason if failed else None)}


def build_graph():
    """编译唯一执行图；条件分支保证初始化失败态不会调用模型。"""
    builder = StateGraph(AgentState, context_schema=RunContext)
    builder.add_node("initialize", initialize)
    builder.add_node("model", model_node)
    builder.add_node("finish", finish)
    builder.add_edge(START, "initialize")
    builder.add_conditional_edges("initialize", lambda state: "finish" if state.status == "failed" else "model",
                                  {"finish": "finish", "model": "model"})
    builder.add_edge("model", "finish")
    builder.add_edge("finish", END)
    return builder.compile()
