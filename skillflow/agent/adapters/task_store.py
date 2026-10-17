"""业务 SQLite 的任务元数据适配器；不负责图检查点或恢复调度。"""

import asyncio

from ..contracts import AgentState, TaskMetadata


class SqliteTaskStore:
    """复用项目 B 存储连接；SQL 留在基础设施层，图只调用端口。"""

    def __init__(self, store):
        self.store = store

    async def get_owned(self, run_id: str, buyer_id: str, session_id: str) -> TaskMetadata:
        """每次执行重新验证归属，状态中的买家字段不能自行授予权限。"""
        record = await asyncio.to_thread(self.store.agent_run, run_id)
        if record is None:
            raise LookupError("任务尚未登记")
        metadata = TaskMetadata.model_validate(record["metadata"])
        if (metadata.buyer_id, metadata.session_id) != (buyer_id, session_id):
            raise PermissionError("RUN_FORBIDDEN")
        if record["status"] != "created":
            # S1 不开放恢复；重复执行不能创建空预算，必须由 SSE 回放原结果。
            raise ValueError("RUN_ALREADY_STARTED")
        return metadata

    async def save_state(self, state: AgentState) -> None:
        """保存带版本的诊断快照和预算；不宣称可从该快照恢复图。"""
        await asyncio.to_thread(self.store.save_agent_state, state.metadata.run_id,
                                state.metadata.model_dump(mode="json"), state.model_dump(mode="json"))
