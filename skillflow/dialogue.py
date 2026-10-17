"""项目 B 的确定性对话识别，用于 Mock 演示及确认门的明确肯定/否定校验。

业务任务与操作动作分开输出；本模块不读订单、不写工单。真实模式由 LLM
理解口语，随后也要经过服务端目标绑定和确认门，不能只相信模型的字段。
"""

import re

from skillflow.contracts import DialogueDecision
from skillflow.planner import classify_intent, extract_order_ids, resolve_order

TICKET_PATTERN = re.compile(r"TK-[A-Z0-9]+", re.IGNORECASE)


def affirmative(message: str) -> bool:
    """只匹配整句肯定，避免把“我不同意”或引用他人的“确认”当作授权。"""
    text = re.sub(r"[\s，,。.!！?？]", "", message)
    return bool(re.fullmatch(r"(?:对|对的|是|是的|没错|就是这件|确认|确认商品|接受|我接受|同意|我同意|好的|可以|提交|确认提交|同意提交|确认取消|取消吧)(?:我接受|我同意|提交|取消)?", text))


def negative(message: str) -> bool:
    """否定优先于确认；确认门不会因为否定句中的“提交”触发写入。"""
    return bool(re.search(r"不(?:是|对|要|同意|接受|确认|提交|取消)|别提交|别取消|算了|先等等|稍后", message))


def decide_dialogue(message: str, context: dict) -> DialogueDecision:
    """Mock 模式输出与 LLM 相同的契约，并保留补订单号前的原任务。"""
    ids = extract_order_ids(message)
    order_id = resolve_order(message, context)
    tickets = TICKET_PATTERN.findall(message)
    ticket_id = tickets[0].upper() if len(tickets) == 1 else None
    reference = "explicit" if ids or tickets else "current_order" if order_id else "none"
    common = dict(order_id=order_id, ticket_id=ticket_id, reference=reference)
    pending = context.get("pending_action")
    if pending and negative(message):
        return DialogueDecision(intent="ticket_action", requested_action="decline", **common)
    if affirmative(message):
        action = {"confirm_product": "confirm_product", "confirm_submit": "confirm_submit",
                  "confirm_cancel": "confirm_cancel"}.get(pending, "confirm_submit")
        return DialogueDecision(intent="ticket_action", requested_action=action,
                                ticket_id=context.get("active_ticket_id"), order_id=order_id,
                                reference="current_ticket" if context.get("active_ticket_id") else reference)
    # 取消动词优先于退款/优惠业务词；“质量不好我不要了”属于退货意图，不属于撤销工单。
    if re.search(r"取消|撤销|撤回", message):
        request_type = "price_adjustment" if re.search(r"价差|价保|补差", message) else (
            "discount_dispute" if re.search(r"优惠|争议", message) else (
            "exchange" if "换货" in message else "return" if "退货" in message else None)
        )
        if request_type is None and not tickets:
            # “取消退款”没有可取消的退款对象；必须先澄清具体工单，不能猜测当前工单。
            return DialogueDecision(intent="ticket_action", requested_action="cancel",
                                    needs_clarification=True,
                                    clarification_question="当前没有退款操作。你是要取消退货工单、换货工单还是优惠争议工单？",
                                    **common)
        return DialogueDecision(intent="ticket_action", requested_action="cancel", request_type=request_type, **common)
    # “申请价差”是建草稿动作；随后仍须价保证据门和用户明确确认，不能直接退款。
    if re.search(r"申请(?:价差|补差|价保|价格保护)|发起(?:价差|补差|价保)|创建(?:价差|补差|价保)", message):
        return DialogueDecision(intent="ticket_action", requested_action="create_draft",
                                request_type="price_adjustment", **common)
    # 价保/补差价先于普通订单兜底判断，避免“补我价差”只返回订单摘要。
    # 这里只识别意图；没有当前价、购买价或降价凭证时，后续分支会明确说明暂不支持。
    if re.search(r"补(?:我|下|一下)?差价|价差|降价|价保|买贵了|价格保护|购买后降价", message):
        return DialogueDecision(intent="price_adjustment", requested_action="investigate", **common)
    if tickets or re.search(r"工单.*(?:状态|进度|历史)|查询.*工单|看看.*工单|申请.*进度", message):
        return DialogueDecision(intent="ticket_action", requested_action="status", **common)
    if re.search(r"申请优惠争议|创建优惠争议|发起争议", message):
        return DialogueDecision(intent="ticket_action", requested_action="create_draft", request_type="discount_dispute", **common)
    # 已确认商品后，“我要退货/我想换货”等自然表达应创建草稿；首次提出时
    # 仍由 pending_goal 保存意图并追问订单号，不能缺少订单就写入工单。
    if re.search(r"申请(?:退货|换货)|创建(?:退货|换货)|申请吧|帮我(?:退|换)|我要(?:退货|换货)|我想(?:退货|换货)|(?:退货|换货)申请", message):
        request_type = ("exchange" if re.search(r"换货|帮我换", message) else
                        context.get("proposed_type") if "申请吧" in message else "return")
        return DialogueDecision(intent="ticket_action", requested_action="create_draft", request_type=request_type, **common)
    # 只补订单号时继续上轮业务任务，但不会因为补号而直接写入工单。
    only_id = len(ids) == 1 and re.sub(r"[\s-]", "", message.upper()) == ids[0]
    goal = context.get("pending_goal") if only_id else message
    kind = context.get("pending_intent") if only_id else None
    kind = kind or classify_intent(goal or message)
    intent = {"order_tracking": "order_query", "unsupported_coupon": "unsupported"}.get(kind, kind)
    # 无业务表达的闲聊不会默认执行当前订单查询。
    if intent == "order_query" and not ids and not re.search(r"查|订单|物流|快递|签收", message):
        intent = "unknown"
    return DialogueDecision(intent=intent, requested_action="investigate", **common)


def build_plan(decision: DialogueDecision, goal: str):
    """将已校验的任务编译成固定只读计划，模型不能改变工具顺序或注入参数。"""
    from skillflow.contracts import Plan, PlanStep
    from skillflow.planner import price_adjustment_message
    skill = {"order_query": "order_tracking", "unsupported": "coupon_dispute"}.get(decision.intent, decision.intent)
    if skill == "price_adjustment":
        if not decision.order_id:
            return Plan(decision="clarify", skill=skill, goal=goal[:500],
                        message="请提供订单号，例如 A123，并准备购买价、当前价或降价凭证。")
        return Plan(decision="execute", skill=skill, goal=goal[:500], order_id=decision.order_id,
                    steps=[PlanStep(step_id="order", tool="get_order", arguments={"order_id": decision.order_id}),
                           PlanStep(step_id="policy", tool="retrieve_policy", depends_on=["order"]),
                           PlanStep(step_id="price", tool="analyze_price_adjustment", depends_on=["order", "policy"])])
    if decision.intent == "unsupported":
        return Plan(decision="unsupported", skill="coupon_dispute", goal=goal[:500],
                    message="优惠券未生效投诉暂不支持。可以先核对订单优惠金额。")
    if not decision.order_id:
        return Plan(decision="clarify", skill=skill, goal=goal[:500],
                    message="请提供需要核对优惠的订单号，例如 A123。" if skill == "coupon_dispute" else "请提供订单号，例如 A123。")
    steps = [PlanStep(step_id="order", tool="get_order", arguments={"order_id": decision.order_id})]
    if skill in {"return_exchange", "coupon_dispute"}:
        steps.append(PlanStep(step_id="policy", tool="retrieve_policy", depends_on=["order"]))
    if skill == "return_exchange":
        steps.extend([PlanStep(step_id="assessment", tool="assess_after_sales", depends_on=["order", "policy"]),
                      PlanStep(step_id="inventory", tool="check_inventory", depends_on=["order", "assessment"])])
    elif skill == "coupon_dispute":
        steps.append(PlanStep(step_id="discount", tool="analyze_discount", depends_on=["order", "policy"]))
    return Plan(decision="execute", skill=skill, goal=goal[:500], order_id=decision.order_id, steps=steps)
