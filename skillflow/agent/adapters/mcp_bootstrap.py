"""S5 装配细节集中在适配层；不在图中导入 SDK，不在构造函数启动子进程。"""

from skillflow.config import ROOT
from ..skills import FileSkillRegistry
from ..tool_registry import LocalToolRegistry
from .mcp_session import McpSessionManager

S5_ENGINE_VERSION = 'tool_loop_s5_v1'
S5_GRAPH_VERSION = 'persistent_mcp_tool_loop_v1'


def configure_logistics(service, settings):
    """固定新引擎版本；连接状态变化不增删模型 Schema，保障合法恢复。"""
    service.engine_version, service.graph_version, service.stage = S5_ENGINE_VERSION, S5_GRAPH_VERSION, 'S5'
    service.mcp_manager = McpSessionManager(settings.agent_logistics_file.resolve(),
        settings.agent_mcp_query_timeout, settings.agent_mcp_startup_timeout)
    service.tools.logistics_enabled, service.tools.mcp_manager = True, service.mcp_manager
    service.tools.registry = LocalToolRegistry(incident_available=True, writes_enabled=True, logistics_enabled=True)
    service.tools.skills = FileSkillRegistry(service.tools.registry.names, s4=True, s5=True)
    return service
