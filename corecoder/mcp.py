"""MCP 标准输入输出客户端，封装 Agent 所需的最小协议子集。

Servers are configured in ~/.corecoder/mcp.json:

    {"mcpServers": {"fs": {"command": "npx", "args": ["-y", "some-fs-server", "/tmp"]}}}

Each server is a subprocess speaking JSON-RPC 2.0, one message per line over
stdin/stdout. Startup handshakes (`initialize`), pulls `tools/list`, and every
remote tool joins the agent as `mcp__<server>__<tool>`, so consent, hooks and
the main loop treat them exactly like the built-ins. A server that hangs or
dies fails that one call as an ordinary error string; it never kills the loop.
"""

# Agent（父进程）
#   ↔ stdin/stdout ↔
# MCP 服务（子进程）

# 为什么用子进程：
# 隔离：服务崩溃不影响 Agent。
# 灵活：任何语言都能写 MCP 服务。
# 标准：JSON-RPC 是通用协议。


import atexit
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

from . import __version__
from .tools.base import Tool

log = logging.getLogger(__name__)

CONFIG_FILE = Path.home() / ".corecoder" / "mcp.json"
PROTOCOL_VERSION = "2025-06-18"
INIT_TIMEOUT = 15  # seconds for initialize + tools/list at startup
CALL_TIMEOUT = 60  # seconds for one tools/call


class MCPError(RuntimeError):
    """与单个 MCP 服务通信时发生的传输或协议错误。"""


class MCPClient:
    """管理一个 stdio MCP 服务进程：握手、列举工具、调用和关闭。

    A daemon thread owns stdout and parks each response under its request id,
    so calls from parallel tool execution match their own replies; the write
    lock keeps two threads' requests from interleaving on stdin.
    """

    def __init__(self, name: str, command: str, args: list = (), env: dict | None = None):
        self.name = name
        self.call_timeout = CALL_TIMEOUT
        self._proc = subprocess.Popen(          #启动子进程
            [command, *args],
            stdin=subprocess.PIPE,              #通过管道通信
            stdout=subprocess.PIPE,
            env={**os.environ, **(env or {})},  # servers inherit the user env, config overrides
            encoding="utf-8",
            errors="replace",
            bufsize=1,              #行缓冲  # line buffered: the transport is newline-delimited JSON
        )
        self._next_id = 0
        self._dead: MCPError | None = None
        self._responses: dict[int, dict] = {}
        self._cond = threading.Condition()      #条件变量，用于线程同步
        self._write_lock = threading.Lock()     #写锁，防止多线程交错写
        threading.Thread(target=self._read_loop, daemon=True).start()   #_read_loop 线程后台读 stdout
        try:
            self._request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "corecoder", "version": __version__},
            }, INIT_TIMEOUT)
            self._notify("notifications/initialized")
            listed = self._request("tools/list", {}, INIT_TIMEOUT)
            self.tools = [MCPTool(self, t) for t in listed.get("tools", [])]
        except BaseException:
            self.close()  # a half-started server must not leak
            raise

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """调用一个远程工具，并把返回内容整理成文本。"""
        result = self._request(
            "tools/call", {"name": tool_name, "arguments": arguments}, self.call_timeout    #发 tools/call 请求。
        )
        text = "\n".join(       #从返回的 content 里提取文本部分
            part.get("text", "")
            for part in result.get("content", [])
            if part.get("type") == "text"
        )
        if result.get("isError"):
            raise MCPError(text or f"{tool_name} reported an error")
        return text or json.dumps(result)  # non-text content: hand the model the raw result

    def close(self):
        """关闭服务进程；重复调用也是安全的。"""
        if self._proc.poll() is not None:   #已死就返回
            return
        try:
            self._proc.stdin.close()        #关 stdin。
        except OSError:
            pass
        self._proc.terminate()          #terminate（温和终止）。
        try:
            self._proc.wait(timeout=5)      #等 5 秒。
        except subprocess.TimeoutExpired:
            self._proc.kill()           #还没死就 kill（强杀）。

    def _notify(self, method: str):
        try:
            self._write({"jsonrpc": "2.0", "method": method, "params": {}})
        except MCPError:
            pass  # a notification has no reply to lose; the next request meets the dead server

    def _request(self, method: str, params: dict, timeout: float) -> dict:
        """发送带 request id 的 JSON-RPC 请求，并等待对应响应。"""

        with self._cond:
            if self._dead is not None:
                raise self._dead
            self._next_id += 1          #分配请求 ID
            req_id = self._next_id
        self._write({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        with self._cond:
            while req_id not in self._responses:        #等响应（条件变量，超时就抛异常），用条件变量等待，不忙等。
                if self._dead is not None:
                    raise self._dead
                left = deadline - time.monotonic()
                if left <= 0:
                    raise MCPError(f"MCP server {self.name!r} gave no answer to {method} in {timeout:g}s")
                self._cond.wait(left)
            msg = self._responses.pop(req_id)
        if "error" in msg:
            raise MCPError(f"MCP server {self.name!r} rejected {method}: {msg['error'].get('message', msg['error'])}")
        return msg.get("result", {})

    def _write(self, msg: dict):        #写请求
        try:
            with self._write_lock:      #加锁写，防止多线程交错
                self._proc.stdin.write(json.dumps(msg) + "\n")
                self._proc.stdin.flush()
        except OSError as e:
            raise MCPError(f"MCP server {self.name!r} is not writable: {e}") from e

    def _read_loop(self):
        #后台线程读：不阻塞主线程。
        # 按 id 分发：多个并发请求能匹配各自的响应。
        # # 服务死了：所有等待者立刻知道。
        """后台读取服务 stdout，按 request id 分发响应。"""

        try:
            for line in self._proc.stdout:
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("MCP server %r sent a non-JSON line, skipped", self.name)
                    continue
                if "id" not in msg:
                    continue  # server notification: nothing here needs an answer
                with self._cond:    #按 request id 存响应，通知等待的线程
                    self._responses[msg["id"]] = msg
                    self._cond.notify_all()
        except (OSError, ValueError):
            pass
        with self._cond:
            # stdout closing means the server is gone for good
            # stdout 关闭 = 服务死了，标记 _dead，通知所有等待者
            if self._dead is None:
                self._dead = MCPError(f"MCP server {self.name!r} exited")
            self._cond.notify_all()


class MCPTool(Tool):
    """把 MCP 服务端工具适配为 CoreCoder 的本地 Tool 接口。"""

    def __init__(self, client: MCPClient, spec: dict):
        self._client = client
        self._remote_name = spec["name"]
        self.name = f"mcp__{client.name}__{spec['name']}"
        self.description = spec.get("description") or ""
        self.parameters = spec.get("inputSchema") or {"type": "object", "properties": {}}

    def execute(self, **kwargs) -> str:
        """把参数转发给远程 MCP 工具。"""

        return self._client.call_tool(self._remote_name, kwargs)


_live_clients: list[MCPClient] = []


def load_mcp_tools(path: Path = CONFIG_FILE) -> list[Tool]:
    """启动配置文件中的 MCP 服务，并返回注册到 Agent 的工具列表。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, OSError) as e:
        log.warning("ignoring %s: %s", path, e)
        return []
    tools: list[Tool] = []
    for name, spec in (data.get("mcpServers") or {}).items():
        try:
            client = MCPClient(name, spec["command"], spec.get("args", []), spec.get("env"))
        except (MCPError, OSError, KeyError, TypeError) as e:
            log.warning("MCP server %r skipped: %s", name, e)
            continue
        _live_clients.append(client)
        tools.extend(client.tools)
    return tools


@atexit.register
def _shutdown():
    """解释器退出时关闭所有仍存活的 MCP 客户端。"""

    for client in _live_clients:
        client.close()
