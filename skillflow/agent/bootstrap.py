"""新引擎唯一装配边界；选择模型和基础设施，不发起外部请求。"""

from uuid import uuid4

from .adapters.events import SqliteEventSink
from .adapters.scripted_model import ScriptedModel
from .adapters.task_store import SqliteTaskStore
from .config import ENGINE_VERSION, GRAPH_VERSION
from .contracts import TaskMetadata
from .service import AgentService


def create_service(store, settings=None, repository=None, policies=None):
    """按显式模式装配，真实模型新图与 S1 脚本骨架保持独立版本。"""
    if settings and settings.agent_engine == "tool_loop" and settings.agent_model_mode == "langchain_openai":
        from .adapters.chat_model import ChatModelGateway
        if settings.agent_s4_enabled:
            from .persistent_service import PersistentAgentService
            service = PersistentAgentService(store,settings,repository,policies,ChatModelGateway(settings))
            if settings.agent_s5_enabled:
                # MCP 只在显式 S5 装配中启用，真正启动由应用 lifespan 负责。
                from .adapters.mcp_bootstrap import configure_logistics
                configure_logistics(service, settings)
            return service
        from .adapters.conversation import SqliteConversationContext
        from .loop_service import ToolLoopService
        return ToolLoopService(ChatModelGateway(settings), SqliteTaskStore(store), SqliteEventSink(store),
            create_tool_runtime(store, repository, policies, settings.timeout), SqliteConversationContext(store, settings.agent_context_ttl_seconds),
            settings.agent_max_context_chars)
    return AgentService(ScriptedModel(), SqliteTaskStore(store), SqliteEventSink(store))


def task_metadata(settings, run_id: str, buyer_id: str, session_id: str) -> dict:
    """首次受理时生成完整元数据，排队执行或重放不重新读取引擎配置。"""
    engine = settings.agent_engine
    # 模型模式在受理时固定对应图版本，排队与重放都不能更换。
    from .loop_service import S3_ENGINE_VERSION, S3_GRAPH_VERSION
    s3 = engine == "tool_loop" and settings.agent_model_mode == "langchain_openai"
    if s3 and settings.agent_s4_enabled:
        # S4 任务独立固定版本，不能用 S3 的诊断快照冒充恢复。
        from .persistent_service import ENGINE_VERSION as S4_ENGINE, GRAPH_VERSION as S4_GRAPH
        S3_ENGINE_VERSION,S3_GRAPH_VERSION=S4_ENGINE,S4_GRAPH
        if settings.agent_s5_enabled:
            from .adapters.mcp_bootstrap import S5_ENGINE_VERSION, S5_GRAPH_VERSION
            S3_ENGINE_VERSION,S3_GRAPH_VERSION=S5_ENGINE_VERSION,S5_GRAPH_VERSION
    return TaskMetadata(
        task_id=str(uuid4()), run_id=run_id, buyer_id=buyer_id, session_id=session_id,
        engine=engine, engine_version=S3_ENGINE_VERSION if s3 else ENGINE_VERSION if engine == "tool_loop" else "legacy_v1",
        graph_version=S3_GRAPH_VERSION if s3 else GRAPH_VERSION if engine == "tool_loop" else "structured_dialogue_v1",
        model_mode=settings.agent_model_mode if engine == "tool_loop" else settings.model_mode,
        limits=settings.agent_limits,
    ).model_dump(mode="json")


def create_tool_runtime(store, repository, policies, tool_timeout: float = 20):
    """S2 独立装配入口；S1 聊天图保持原行为，S3 再将此组件注入模型循环。"""
    from .tool_runtime import ToolRuntime
    return ToolRuntime(store, repository, policies, tool_timeout)
