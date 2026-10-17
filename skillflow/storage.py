import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path


class JsonStore:
    """项目 B 的本地演示存储。

    正常情况下数据写入 JSON 文件，后续可以把这一层替换为 SQLite。
    某些受限运行环境会允许读取工作区但禁止创建临时文件；为了让
    FastAPI 演示仍能返回结构化结果，写入失败时回退到当前进程内存。
    这只保证本次服务进程内的会话和工单可用，真正部署时应配置可写目录。
    """

    def __init__(self, root: Path):
        self.root = root
        # 每个应用实例独立保存回退数据，避免不同 buyer 或测试应用互相污染。
        self._memory: dict[str, object] = {}
        self._disk_writable = True
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            # 只读沙箱仍可读取已有 fixture；后续 write 会走内存回退。
            self._disk_writable = False

    def read(self, name: str, default=None):
        # 内存值优先，确保一次写入失败后同一进程能读回最新会话状态。
        if name in self._memory:
            return deepcopy(self._memory[name])
        path = self.root / (name + ".json")
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
        except OSError:
            return default

    def write(self, name: str, value):
        # 先复制到内存，保证即使文件系统不可写，业务链路也能继续。
        self._memory[name] = deepcopy(value)
        if not self._disk_writable:
            return
        path = self.root / (name + ".json")
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, path)
        except OSError:
            # 文件写入失败属于存储降级，不应让已经完成的 Agent 请求变成 500。
            self._disk_writable = False
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def session_key(session_id: str):
        return "session_" + hashlib.sha256(session_id.encode()).hexdigest()

    def session(self, session_id: str, buyer_id: str | None = None):
        # 对话状态保存当前商品、当前工单和待确认动作，供“刚才那个/对的”这类表达解析。
        # 这些字段只保存会话提示；真正的工单状态仍由 TicketStore 重新校验。
        # 同一个 session_id 在不同买家下使用不同存储键，防止复用聊天 ID 读取他人上下文。
        key = f"{buyer_id}:{session_id}" if buyer_id else session_id
        defaults = {
            "order_id": None,
            "messages": [],
            "active_ticket_id": None,
            "active_ticket_type": None,
            "pending_action": None,
            "pending_product_confirmation": None,
            "confirmation_text": None,
            "pending_goal": None,
            "pending_intent": None,
            # 取消意图跨一次补订单号追问保留；具体工单仍须每轮从数据库核验。
            "pending_cancel": False,
            "proposed_type": None,
            "product_confirmed": False,
        }
        return {**defaults, **(self.read(self.session_key(key), {}) or {})}
