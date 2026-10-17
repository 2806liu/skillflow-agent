"""图依赖的端口；数据库、MCP 和 LangChain 具体对象只存在于适配器。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .tool_executor import TargetScope, ToolSurface
    from .budget import BudgetTracker
    from .conversation_state import ConversationView, ConversationSnapshot
    from .goals import TaskGoal

from .contracts import AgentState, MessageEnvelope, ModelReply, RuntimeEvent, SkillDocument, TaskMetadata, ToolCall, ToolObservation, ToolSpec


class ModelGateway(Protocol):
    """只负责生成消息，不调度或执行业务工具。"""

    async def generate(self, messages: list[MessageEnvelope], tool_specs: list[ToolSpec]) -> ModelReply: ...


class ToolCallingModelGateway(ModelGateway, Protocol):
    """S3 解释目标和生成最终提案，与原生工具动作消息保持不同契约。"""

    async def interpret_goal(self, text: str): ...
    async def propose_answer(self, facts: list[str], refs: list[str]): ...


class SkillRegistry(Protocol):
    """技能发现和版本化内容加载；不授予业务写权限。"""

    async def list_catalog(self) -> list[dict[str, str]]: ...
    async def load(self, name: str, version: str | None = None) -> SkillDocument: ...
    async def get_pinned(self, name: str, version: str, content_hash: str) -> SkillDocument: ...


class ToolRegistry(Protocol):
    """返回当前已开放工具的模型 Schema。"""

    def model_specs(self, names: list[str]) -> list[ToolSpec]: ...


class ToolExecutor(Protocol):
    """执行边界绑定可信归属，模型不能自行提供身份。"""

    def begin_turn(self, model_call_id: str) -> ToolSurface: ...
    async def dispatch(self, call: ToolCall, surface: ToolSurface) -> ToolObservation: ...


class TaskStore(Protocol):
    """加载服务端元数据并保存诊断快照；S1 快照不是图检查点。"""

    async def get_owned(self, run_id: str, buyer_id: str, session_id: str) -> TaskMetadata: ...
    async def save_state(self, state: AgentState) -> None: ...


class CheckpointAdapter(Protocol):
    """预留实际图检查点端口，S1 不提供恢复实现。"""

    async def load_snapshot(self, metadata: TaskMetadata) -> AgentState: ...


class EventSink(Protocol):
    """持久化并发布进度，失败须向上传播。"""

    async def append(self, event: RuntimeEvent) -> None: ...


class GraphToolSession(ToolExecutor, Protocol):
    """执行图只接触工具会话端口，不导入业务数据库、MCP 或本地 handler。"""

    skills: SkillRegistry
    observations: list[ToolObservation]

    @property
    def loaded(self) -> dict[str, SkillDocument]: ...
    async def verify_current(self) -> None: ...
    def dump_state(self) -> dict: ...
    async def restore_state(self, value: dict) -> None: ...


class GraphToolFactory(Protocol):
    """装配层创建每任务独立工具会话，注入共享预算和审计端口。"""

    def graph_session(self, metadata: TaskMetadata, scope: TargetScope, budget: BudgetTracker, events: EventSink) -> GraphToolSession: ...
    async def application_requirements(self, metadata: TaskMetadata, application: dict) -> set[str]: ...


class ConversationContextPort(Protocol):
    """短期目标存储独立于图；有效期、暂停和候选更新不提供执行图恢复。"""

    async def load(self, metadata: TaskMetadata) -> ConversationView: ...
    async def remember_goal(self, metadata: TaskMetadata, goal: TaskGoal) -> ConversationSnapshot: ...
    async def pause(self, metadata: TaskMetadata, reason: str) -> ConversationSnapshot: ...
    async def save_outcome(self, state: AgentState) -> None: ...
