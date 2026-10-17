"""S3 有界上下文：保留完整工具消息对，超预算明确终止，不做隐式摘要或记忆。"""

import json

from .budget import BudgetExceeded


def checked_context(messages, tool_specs, max_chars: int):
    """同时检查工具调用配对与工具 Schema 体积；拒绝截断工具观察。"""
    pending, seen = set(), set()
    for item in messages:
        if item.role == "assistant" and item.tool_calls:
            if pending:
                raise ValueError("INCOMPLETE_TOOL_MESSAGES")
            for call in item.tool_calls:
                if call.call_id in seen:
                    raise ValueError("DUPLICATE_TOOL_CALL_ID")
                pending.add(call.call_id)
                seen.add(call.call_id)
        elif item.role == "tool":
            if item.tool_call_id not in pending:
                raise ValueError("ORPHAN_TOOL_MESSAGE")
            pending.remove(item.tool_call_id)
        elif pending:
            raise ValueError("INCOMPLETE_TOOL_MESSAGES")
    if pending:
        raise ValueError("INCOMPLETE_TOOL_MESSAGES")
    size = len(json.dumps([m.model_dump() for m in messages], ensure_ascii=False))
    size += len(json.dumps([s.model_dump() for s in tool_specs], ensure_ascii=False))
    if size > max_chars:
        raise BudgetExceeded("CONTEXT_BUDGET_EXCEEDED")
    return size
