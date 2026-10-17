"""纯会话策略：识别约定的单号续问；目标切换仍交由模型解释并由服务端校验。"""

import re
from dataclasses import dataclass

from .conversation_state import ConversationView
from .goals import TaskGoal
from .tool_executor import TargetScope

ORDER_PATTERN = r"(?<![A-Za-z0-9])A[0-9]{3,8}(?![A-Za-z0-9])"


@dataclass(frozen=True)
class ConversationDecision:
    """策略结果不含工具动作；程序不能据此绕过新任务的工具授权与取证。"""

    action: str
    reason: str
    goal: TaskGoal | None = None
    question: str = ""
    keep_pending: bool = False


def decide_conversation(message: str, view: ConversationView) -> ConversationDecision:
    """只继承约定的明确单号短句；有完整新需求时不把旧目标交给模型。"""
    text = message.strip().strip("，。！？!?：: ")
    ids = tuple(dict.fromkeys(re.findall(ORDER_PATTERN, text.upper())))
    prefix = r"(?:(?:选择|选|就|那就|那|再查|再看看|再调查|查看|查询|调查|订单|帮我查|查)|(?:再)?按刚才(?:的)?(?:方式)?查|按同样方式查|同样查)?"
    short = len(ids) == 1 and re.fullmatch(
        prefix + r"\s*A[0-9]{3,8}\s*(?:呢|吧|看看|也一样|也查一下)?", text, re.I)
    if short:
        saved = view.snapshot
        if view.status == "active" and saved and saved.template:
            if saved.candidates and ids[0] not in saved.candidates:
                return ConversationDecision("clarify", "CANDIDATE_NOT_LISTED", question=
                    "该单号不在本次待选候选中。请选择：" + "、".join(saved.candidates) + "；如需调查其他订单，请完整说明调查目的。", keep_pending=True)
            goal = TaskGoal(requirements=list(saved.template.requirements), scope=TargetScope(explicit_order_ids=ids),
                original_request=message, continuation_of_task_id=saved.source_task_id)
            return ConversationDecision("inherit", "PENDING_SELECTION" if saved.candidates else "FOLLOWUP_ORDER_ID", goal)
        reasons = {"expired": "CONTEXT_EXPIRED", "paused": "CONTEXT_PAUSED", "incompatible": "CONTEXT_INCOMPATIBLE", "missing": "CONTEXT_MISSING"}
        return ConversationDecision("clarify", reasons[view.status], question=
            ("上次调查目标已过期。" if view.status == "expired" else "当前没有可直接沿用的调查目标。") +
            "请说明这笔订单要查询订单状态、退换资格、质量事件还是其他售后问题，例如“A701 是否受质量事件影响，能否退货，有哪些替代品”。")
    # 本轮没有实现无单号代词解析；裸多单也不能继承旧任务的批量权限。
    residue = re.sub(ORDER_PATTERN, "", text.upper()).strip(" ，、和及与/;；")
    if (not ids and re.search(r"这个|那个|它|刚才那笔|还有呢", text)) or (len(ids) > 1 and not residue):
        return ConversationDecision("clarify", "CONTEXT_AMBIGUOUS", question="请给出一个明确订单号，并说明要继续哪项调查；批量质量事件查询请明确写出批量需求。")
    return ConversationDecision("new", "EXPLICIT_NEW_REQUEST")
