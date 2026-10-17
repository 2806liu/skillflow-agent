"""S2 装配后的任务工具服务；通过服务端存储加载归属，不接受模型提供身份。"""

import asyncio

from .adapters.events import SqliteEventSink
from .adapters.local_tools import LocalBusinessAdapter
from .adapters.task_store import SqliteTaskStore
from .skills import FileSkillRegistry
from .tool_executor import TargetScope, ToolSession
from .tool_registry import LocalToolRegistry


class ToolRuntime:
    """S2 无模型调度，仅为测试脚本和后续图提供按任务隔离的工具组件。"""

    def __init__(self, store, repository, policies, tool_timeout: float = 20):
        self.repository, self.policies, self.tool_timeout = repository, policies, tool_timeout
        self.tasks, self.events = SqliteTaskStore(store), SqliteEventSink(store)
        self.registry = LocalToolRegistry(incident_available=hasattr(repository, "evaluate"))
        self.skills = FileSkillRegistry(self.registry.names)
        self._sessions, self._lock = {}, asyncio.Lock()
        self.logistics_enabled, self.mcp_manager = False, None

    async def open_task(self, run_id: str, buyer_id: str, session_id: str, scope: TargetScope) -> ToolSession:
        """在服务端初始化目标范围，同 run 不得重新建空预算或更换任务范围。"""
        async with self._lock:
            cached = self._sessions.get(run_id)
            if cached:
                if (cached.metadata.buyer_id, cached.metadata.session_id) != (buyer_id, session_id):
                    raise PermissionError("RUN_FORBIDDEN")
                if cached.scope != scope:
                    raise ValueError("TARGET_SCOPE_CHANGED")
                return cached
            metadata = await self.tasks.get_owned(run_id, buyer_id, session_id)
            if metadata.engine != "tool_loop":
                raise ValueError("ENGINE_VERSION_UNSUPPORTED")
            session = ToolSession(metadata, scope, self.registry, self.skills,
                                  LocalBusinessAdapter(self.repository, self.policies, metadata.buyer_id),
                                  self.tasks, self.events, tool_timeout=self.tool_timeout)
            self._sessions[run_id] = session
            return session

    def close_task(self, run_id: str):
        """释放已结束任务的内存证据与工具面；后续恢复必须等待 S4 的检查点实现。"""
        self._sessions.pop(run_id, None)

    def graph_session(self, metadata, scope, budget, events):
        """S3 已经核验任务登记；直接注入同一预算与事件端口，不重新登记或重置状态。"""
        adapter = LocalBusinessAdapter(self.repository, self.policies, metadata.buyer_id)
        if self.logistics_enabled:
            # 协议适配器组合本地端口，图节点无需知道 MCP SDK 或签名凭据。
            from .adapters.mcp_tools import LogisticsBusinessAdapter
            adapter = LogisticsBusinessAdapter(adapter, self.mcp_manager)
        return ToolSession(metadata, scope, self.registry, self.skills,
            adapter, self.tasks, events,
            budget=budget, tool_timeout=self.tool_timeout, graph_managed=True)

    async def application_requirements(self, metadata, application):
        """用可信商品类别补足申请验收条件，不生成工具顺序，也不扩大单号范围。"""
        order=await asyncio.to_thread(self.repository.get_order,application['order_id'],metadata.buyer_id)
        return {'incident'} if order and order.get('category')=='power_bank' else set()
