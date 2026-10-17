"""应用层执行入口；绑定服务端任务、适配器和每运行独立预算。"""

import asyncio

from .budget import BudgetTracker
from .config import ENGINE_VERSION, GRAPH_VERSION
from .contracts import AgentState, MessageEnvelope
from .graph import RunContext, build_graph
from .ports import EventSink, ModelGateway, TaskStore


class AgentService:
    """不依赖 FastAPI；实际授权由入口 Principal 和存储归属共同校验。"""

    def __init__(self, model: ModelGateway, tasks: TaskStore, events: EventSink):
        self.model, self.tasks, self.events = model, tasks, events
        self.graph = build_graph()

    async def run(self, run_id: str, buyer_id: str, session_id: str, message: str) -> AgentState:
        """S1 只创建新执行，不提供快照恢复接口，避免伪造恢复语义。"""
        metadata = await self.tasks.get_owned(run_id, buyer_id, session_id)
        if (metadata.engine, metadata.engine_version, metadata.graph_version, metadata.model_mode) != (
                "tool_loop", ENGINE_VERSION, GRAPH_VERSION, "scripted"):
            raise ValueError("ENGINE_VERSION_UNSUPPORTED")
        initial = AgentState(metadata=metadata, goal=message,
                             messages=[MessageEnvelope(role="user", content=message)])
        tracker = BudgetTracker(metadata.limits, initial.budget)
        context = RunContext(metadata, self.model, tracker, self.events, self.tasks)
        tracker.start()
        try:
            result = AgentState.model_validate(await self.graph.ainvoke(initial, context=context))
        except asyncio.CancelledError:
            # 正常取消也记录已占用预算；进程被强杀的恢复在 S4 验收。
            initial.budget = tracker.pause()
            initial.status, initial.stop_reason = "failed", "RUN_INTERRUPTED"
            await asyncio.shield(self.tasks.save_state(initial))
            raise
        except Exception:
            initial.budget = tracker.pause()
            initial.status, initial.stop_reason = "failed", "RUNTIME_FAILED"
            await self.tasks.save_state(initial)
            raise
        finally:
            tracker.pause()
        result.budget = tracker.snapshot()
        await self.tasks.save_state(result)
        return result
