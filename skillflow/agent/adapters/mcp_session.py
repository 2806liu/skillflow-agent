"""受管 MCP 会话：专用任务拥有 SDK 上下文，查询可并行，超时不销毁共享连接。"""

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
import os
from pathlib import Path
import secrets
import sys
import time
from pydantic import ValidationError

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from skillflow.integrations.logistics_mcp.contracts import grant

_identity = ContextVar('logistics_identity', default=None)


class McpFailure(Exception):
    """跨边界仅传稳定码；SDK 原异常可能含参数，禁止写入模型与审计。"""
    def __init__(self, code, retryable=False):
        self.code, self.retryable = code, retryable
        super().__init__(code)


def stable_failure(exc):
    """SDK 的任务组会包裹异常；只提取自有稳定码，绝不序列化异常组正文。"""
    if isinstance(exc, McpFailure):
        return exc.code
    for child in getattr(exc, 'exceptions', ()):
        code = stable_failure(child)
        if code:
            return code
    return None


@dataclass
class Generation:
    """同一代连接持有自己的工具对象；旧代故障不能关闭后来重建的新代。"""
    ready: asyncio.Future
    stop: asyncio.Event
    tool: object = None
    task: asyncio.Task | None = None
    broken: bool = False


class McpSessionManager:
    """每应用一个连接；发现失败降级，按需单次重建，失败冷却至少十秒。"""
    def __init__(self, data_path: Path, query_timeout=10., startup_timeout=10., cooldown=10., command=None):
        self.data_path, self.query_timeout = Path(data_path).resolve(), query_timeout
        self.startup_timeout, self.cooldown = startup_timeout, cooldown
        self.secret = secrets.token_urlsafe(48)
        self.command = command or [sys.executable, '-B', '-m', 'skillflow.integrations.logistics_mcp.server']
        self._lock, self._generation = asyncio.Lock(), None
        self._closed, self._next_attempt = False, 0.
        self.starts, self.calls, self.last_error = 0, 0, None
        self.protocol_version, self.server_version = None, None

    def health(self):
        """健康检查纯读，不重连、不泄露密钥和内部路径。"""
        g = self._generation
        ready = bool(g and g.tool and not g.broken and not g.task.done() and not self._closed)
        return {'status': 'ready' if ready else 'unavailable', 'transport': 'stdio',
                'service': 'skillflow-logistics', 'error_code': self.last_error,
                'connections_started': self.starts, 'protocol_calls': self.calls,
                'protocol_version': self.protocol_version, 'server_version': self.server_version,
                'discovered_tools': ['logistics_get_logistics'] if ready else []}

    async def start(self):
        """启动失败只记录降级，不阻止本地订单/工单服务。"""
        try:
            await asyncio.wait_for(self._ensure(), self.startup_timeout)
        except (McpFailure, asyncio.TimeoutError):
            self.last_error = self.last_error or 'MCP_UNAVAILABLE'

    async def _ensure(self):
        """并发请求共用重建锁；等待锁也由上层单次查询期限限制。"""
        async with self._lock:
            if self._closed:
                raise McpFailure('MCP_UNAVAILABLE', True)
            old = self._generation
            if old and old.tool and not old.broken and not old.task.done():
                return old
            if time.monotonic() < self._next_attempt:
                raise McpFailure(self.last_error or 'MCP_UNAVAILABLE', self.last_error != 'MCP_SCHEMA_INVALID')
            if old:
                old.stop.set()
                await old.task
            g = Generation(asyncio.get_running_loop().create_future(), asyncio.Event())
            self._generation = g
            self.starts += 1
            g.task = asyncio.create_task(self._owner(g))
            # shield 保证某次查询超时不会取消共享的初始化或所有者任务。
            await asyncio.shield(g.ready)
            if not g.tool:
                raise McpFailure(self.last_error or 'MCP_UNAVAILABLE', self.last_error != 'MCP_SCHEMA_INVALID')
            return g

    async def _owner(self, g):
        """SDK 的 AnyIO 上下文始终在同一任务进入/退出，避免取消作用域跨任务。"""
        try:
            env = {k: os.environ[k] for k in ('SYSTEMROOT', 'WINDIR', 'PATH', 'TEMP', 'TMP') if k in os.environ}
            env.update(SKILLFLOW_LOGISTICS_SECRET=self.secret, SKILLFLOW_LOGISTICS_DATA=str(self.data_path),
                       PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
            params = StdioServerParameters(command=self.command[0], args=self.command[1:], env=env,
                                            cwd=str(Path(__file__).resolve().parents[3]))
            # 不将 SDK 的原始异常日志暴露给 Trace；服务端业务故障由规范化观察记录。
            with open(os.devnull, 'w') as errlog:
                async with stdio_client(params, errlog=errlog) as (read, write):
                    async with ClientSession(read, write) as session:
                        async with asyncio.timeout(self.startup_timeout):
                            initialized = await session.initialize()
                            tools = await load_mcp_tools(session, server_name='logistics', tool_name_prefix=True,
                                                        tool_interceptors=[self._authorize], handle_tool_errors=False)
                            if len(tools) != 1 or tools[0].name != 'logistics_get_logistics':
                                raise McpFailure('MCP_SCHEMA_INVALID')
                            schema = tools[0].args_schema
                            props = schema.get('properties', {})
                            if (set(props) != {'order_id', 'access_grant'} or schema.get('required') != ['order_id'] or
                                    props['order_id'].get('type') != 'string' or
                                    props['order_id'].get('pattern') != '^A[0-9]{3,8}$' or
                                    props['access_grant'].get('type') != 'string'):
                                raise McpFailure('MCP_SCHEMA_INVALID')
                            g.tool = tools[0]
                            self.protocol_version = initialized.protocolVersion
                            self.server_version = initialized.serverInfo.version
                            self.last_error = None
                            g.ready.set_result(True)
                        await g.stop.wait()
        except BaseException as exc:
            # 此任务只管理本应用资源；清理异常也不会使父应用的其他业务失败。
            self.last_error = stable_failure(exc) or 'MCP_UNAVAILABLE'
            self._next_attempt = time.monotonic() + self.cooldown
        finally:
            g.broken = True
            if not g.ready.done():
                g.ready.set_result(False)

    async def _authorize(self, request, handler):
        """Adapter 拦截器从当前调用上下文取身份，不接收模型传入的 access_grant。"""
        bound = _identity.get()
        if not bound or request.name != 'get_logistics' or set(request.args) != {'order_id'}:
            raise McpFailure('MCP_ACCESS_DENIED')
        buyer, order = bound
        if request.args['order_id'] != order:
            raise McpFailure('MCP_ACCESS_DENIED')
        self.calls += 1
        return await handler(request.override(args={**request.args, 'access_grant': grant(self.secret, buyer, order)}))

    async def call(self, buyer_id, order_id):
        """一次显式工具尝试最多发一个协议查询；超时取消本次等待，不重放不重启。"""
        g = None
        try:
            async with asyncio.timeout(self.query_timeout):
                g = await self._ensure()
                token = _identity.set((buyer_id, order_id))
                try:
                    # ToolCall 形式保留 Adapter 的结构化 artifact，便于严格结果验证。
                    return await g.tool.ainvoke({'type': 'tool_call', 'id': secrets.token_hex(12),
                                                 'name': g.tool.name, 'args': {'order_id': order_id}})
                finally:
                    _identity.reset(token)
        except asyncio.TimeoutError:
            raise McpFailure('MCP_TIMEOUT', True) from None
        except McpFailure:
            raise
        except Exception as exc:
            from langchain_core.tools import ToolException
            if isinstance(exc, (ToolException, ValidationError)):
                raise McpFailure('MCP_INVALID_RESPONSE') from None
            if g:
                g.broken = True
                g.stop.set()
            self.last_error = 'MCP_UNAVAILABLE'
            raise McpFailure('MCP_UNAVAILABLE', True) from None

    async def aclose(self):
        """只清理本应用拥有的子进程与 SDK 资源；退出后禁止再次启动。"""
        self._closed = True
        if self._generation:
            self._generation.stop.set()
            await self._generation.task
