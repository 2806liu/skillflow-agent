"""User consent for tool calls, distilled from Claude Code's permissions.

The tools split in two. Read-only ones (read_file, glob, grep, todo_write)
run the moment the model asks; the mutating ones (edit_file, write_file,
bash, and spawning a sub-agent) stop for a yes first. "Always allow" is
remembered per tool, and the REPL passes a persist_path so the grant also
survives restarts via ~/.corecoder/permissions.json. Per tool rather than
per command, because one approved bash prefix says nothing about the next
command anyway.

When there is nobody to ask (one-shot -p mode, or a library embedding with
no callback), a mutating call is refused instead of blocking on input that
can never arrive. The refusal travels back as an ordinary tool result, so
the loop survives and the model can route around it.
"""

#权限管理
# Permission（权限管理）
# ├── READ_ONLY    只读工具，直接放行
# ├── allow_all    --yes 模式，全部放行
# ├── _always      记住"总是允许"的工具
# ├── ask          问用户的回调
# └── _persist_path 持久化"总是允许"

import json
from pathlib import Path

DEFAULT_STORE = Path.home() / ".corecoder" / "permissions.json"


class Permission:
    """会话级授权状态；由 CLI 注入 ask 回调，不在本类中直接询问用户。"""

    READ_ONLY = frozenset({"read_file", "glob", "grep", "todo_write"})#用 frozenset（不可变

    def __init__(self, ask=None, allow_all: bool = False, persist_path: Path | None = None):
        # ask(tool_name, arguments) -> "once" | "always" | "deny"
        self.ask = ask
        self.allow_all = allow_all
        self._always: set[str] = set()  #记住“总是允许”的工具。
        self._persist_path = persist_path
        if persist_path is not None:
            try:
                saved = json.loads(persist_path.read_text(encoding="utf-8"))
                self._always = {t for t in saved.get("always", []) if isinstance(t, str)}
            except (OSError, ValueError, AttributeError):
                pass  # a missing or corrupt store just means nothing granted yet

    def check(self, tool_name: str, arguments: dict) -> str | None:

# 1. 只读工具 / allow_all / 已"总是允许" → 放行（返回 None）
#    ↓
# 2. 没有 ask 回调（无人可问） → 拒绝
#    ↓
# 3. 有 ask 回调 → 问用户
#    ├── "always" → 记住，保存，放行
#    ├── "once" → 放行
#    └── "deny" → 拒绝

        """判断一次工具调用是否放行；字符串结果会作为拒绝原因返回模型。"""
        if tool_name in self.READ_ONLY or self.allow_all or tool_name in self._always:
            return None
        if self.ask is None:
            return (
                f"Permission denied: {tool_name} mutates state and this session is "
                "non-interactive, so nobody can approve it. Rerun with --yes, or "
                "tell the user the exact step so they can run it themselves."
            )
        verdict = self.ask(tool_name, arguments)
        if verdict == "always":
            self._always.add(tool_name)
            self._save()
            return None
        if verdict == "once":
            return None
        return (
            f"Permission denied: the user refused this {tool_name} call. "
            "Do not retry it unchanged; ask what they would prefer instead."
        )

    def _save(self) -> None:
        if self._persist_path is None:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(
                json.dumps({"always": sorted(self._always)}, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass  # the grant still stands for the rest of this session
