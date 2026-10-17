"""将新图事件持久化并桥接已有 Trace/SSE；不传递消息或订单明文。"""

import asyncio

from skillflow.contracts import TraceEvent
from skillflow.streaming import publish_trace

from ..contracts import RuntimeEvent


def to_trace(event: RuntimeEvent) -> TraceEvent:
    """公共 Trace 携带归属和版本元数据，供已有运行记录和 SSE 展示。"""
    # 恢复段可能没有模型请求，但整个 S4 图使用真实模型入口，不能被标为 S1 脚本。
    stage="S5" if event.metadata.engine_version.startswith("tool_loop_s5_") else "S4" if event.metadata.engine_version.startswith("tool_loop_s4_") else event.details.get("stage")
    return TraceEvent(node="agent_" + event.node, status=event.status,
                      summary="S5 MCP 物流与售后任务" if stage=='S5' else "S4 可恢复售后任务" if stage=="S4" else "S3 真实模型只读调查" if stage == "S3" else "S2 技能与工具边界验证" if stage == "S2" else "S1 脚本运行时验证",
                      duration_ms=event.duration_ms,
                      details={**event.details, "stage":stage, "task_id": event.metadata.task_id, "run_id": event.metadata.run_id,
                               "buyer_id": event.metadata.buyer_id, "session_id": event.metadata.session_id,
                               "engine": event.metadata.engine, "engine_version": event.metadata.engine_version,
                               "graph_version": event.metadata.graph_version,
                               "schema_version": event.metadata.schema_version,
                               # 模型实际参与的新图明确标 real_model，S1/S2 仍为脚本模拟。
                               "execution_mode": "real_model" if stage in {"S3","S4","S5"} else "simulation", "budget": event.budget.model_dump(),
                               "error_code": event.error_code})


class SqliteEventSink:
    """独立事件表保留运行过程；SSE 回放沿用旧持久帧机制。"""

    def __init__(self, store):
        self.store = store

    async def append(self, event: RuntimeEvent) -> None:
        """数据库失败或进度队列溢出均向上传播，不丢弃错误后返回成功。"""
        await asyncio.to_thread(self.store.append_agent_event, event.metadata.run_id,
                                event.model_dump(mode="json"))
        publish_trace(to_trace(event))
