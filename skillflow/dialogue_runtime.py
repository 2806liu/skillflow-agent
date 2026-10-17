"""客服对话状态图：理解 → 目标/确认门 → 只读调查或工单图 → 事实回答。

输入消息、买家隔离的会话和服务端依赖，输出响应及更新后的会话。模型输出
只作为动作提议；订单来源、工单归属、确认内容和实时状态在写操作前重校验。
所有节点返回 Trace；模型异常时记录 fallback，并切换到同一项目内的确定性路由。
"""

import asyncio
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from skillflow.contracts import DialogueDecision, TraceEvent
from skillflow.streaming import publish_trace
from skillflow.dialogue import TICKET_PATTERN, affirmative, build_plan, decide_dialogue, negative
from skillflow.planner import classify_intent, extract_order_ids, resolve_order
from skillflow.ticket_runtime import request_type_label


class DialogueState(TypedDict, total=False):
    """每次请求独立传递状态，会话记忆由 API 在图运行后持久化。"""
    message: str
    session_id: str
    context: dict
    dialogue: DialogueDecision
    status: str
    answer: str
    events: list
    results: dict
    citations: list
    plan: object
    ready: bool
    ticket: object
    ticket_events: list


def clear_pending(context):
    """失效/换话题时清除确认门，避免一句“对”确认先前的其他动作。"""
    for name in ("pending_action", "pending_confirmation", "confirmation_text", "pending_product_confirmation"):
        context[name] = None


def bind_ticket(context, ticket):
    """网页和聊天共同绑定工单；只针对草稿展示提交确认提示。"""
    value = ticket.model_dump() if hasattr(ticket, "model_dump") else ticket
    if context.get("order_id") != value["order_id"]:
        clear_pending(context)
    context.update(order_id=value["order_id"], active_ticket_id=value["ticket_id"],
                   active_ticket_type=value["request_type"], product_confirmed=True)
    clear_pending(context)
    if value["status"] == "draft":
        prompt = f"请确认提交订单 {value['order_id']} 的{request_type_label(value['request_type'])}工单 {value['ticket_id']}。"
        context.update(pending_action="confirm_submit", confirmation_text=prompt,
                       pending_confirmation={"action": "confirm_submit", "ticket_id": value["ticket_id"],
                                             "order_id": value["order_id"], "updated_at": value["updated_at"]})


class DialogueRuntime:
    """复用订单调查图和工单图，不让 LLM 直接执行业务写入。"""
    def __init__(self, runtime, ticket_runtime, ticket_store, settings, buyer_id):
        self.runtime, self.tickets, self.store, self.settings = runtime, ticket_runtime, ticket_store, settings
        # 每个实例只服务一个已认证买家；并发请求不会覆盖共享的“当前买家”。
        self.buyer_id = buyer_id
        graph = StateGraph(DialogueState)
        for name in ("understand", "guard", "perform", "respond"):
            graph.add_node(name, getattr(self, name))
        graph.add_edge(START, "understand")
        graph.add_conditional_edges("understand", lambda s: "guard" if s.get("ready") else "respond")
        graph.add_conditional_edges("guard", lambda s: "perform" if s.get("ready") else "respond")
        graph.add_edge("perform", "respond")
        graph.add_edge("respond", END)
        self.graph = graph.compile()

    @staticmethod
    def event(state, node, status, summary, **kwargs):
        """只记录对外决策和执行事实，不记录密钥或模型内部推理。"""
        event = TraceEvent(node=node, status=status, summary=summary, **kwargs)
        publish_trace(event)
        return [*state.get("events", []), event]

    def clarify(self, state, question):
        """统一输出澄清决策；未经过目标门的动作不会执行任何业务工具。"""
        decision = state["dialogue"].model_copy(update={"needs_clarification": True,
                                                       "clarification_question": question})
        return {"dialogue": decision, "ready": False, "status": "clarify", "answer": question,
                "events": self.event(state, "dialogue_guard", "skipped", question)}

    async def understand(self, state):
        """从数据库刷新当前工单，再把最近消息与可验证上下文传给模型。"""
        context = state["context"]
        record = self.store.get(context.get("active_ticket_id"))
        if record and record["buyer_id"] == self.buyer_id:
            context["active_ticket"] = self.store.public(record).model_dump()
            if record["status"] == "cancelled":
                clear_pending(context)
        else:
            context["active_ticket"] = None
            context["active_ticket_id"] = None
            if context.get("pending_action") in {"confirm_submit", "confirm_cancel"}:
                clear_pending(context)
        try:
            decision = await asyncio.wait_for(self.runtime.plan_service.decide(state["message"], context),
                                              self.settings.model_timeout)
            return {"dialogue": decision, "ready": True,
                    "events": self.event(state, "dialogue", "success", "已生成结构化对话决策。",
                                         details=decision.model_dump())}
        except Exception:
            # LLM 只负责增强自然语言理解；服务商超时或不支持结构化输出时，
            # 回退到项目 B 自己的确定性路由，再由同一套权限、政策和确认门校验。
            # 这样模型故障不会把价差请求退化成订单摘要，也不会放行未经确认的写操作。
            fallback = decide_dialogue(state["message"], context)
            return {"dialogue": fallback, "ready": True,
                    "events": self.event(state, "dialogue", "fallback", "模型不可用，已使用项目 B 确定性对话路由。",
                                         details=fallback.model_dump())}

    async def guard(self, state):
        """重新绑定可信目标，验证确认门和多目标歧义；忽略模型编造的订单号。"""
        decision, context, message = state["dialogue"], state["context"], state["message"]
        ids = extract_order_ids(message)
        ticket_ids = list(dict.fromkeys(value.upper() for value in TICKET_PATTERN.findall(message)))
        only_order_id = len(ids) == 1 and re.sub(r"[\s-]", "", message.upper()) == ids[0]
        cancel_words = bool(re.search(r"取消|撤销|撤回", message)) and not negative(message)
        continue_cancel = only_order_id and context.get("pending_cancel")
        # “不是它”否定的是当前商品目标，而不只是拒绝提交；必须先解除旧订单绑定。
        if re.search(r"不是(?:它|这(?:个|件|款)(?:商品|东西)?|该(?:商品|订单)|这(?:单|笔订单))|搞错(?:商品|订单)|换(?:一|个)(?:件|笔)?(?:商品|订单)", message):
            old_order = context.get("order_id")
            old_ticket = context.get("active_ticket") or {}
            clear_pending(context)
            context.update(order_id=None, active_product=None, product_confirmed=False,
                           active_ticket_id=None, active_ticket_type=None, active_ticket=None,
                           proposed_type=None, pending_cancel=False,
                           pending_goal="退货调查", pending_intent="return_exchange")
            ticket_note = ("原草稿仍保留且未提交；如需取消，请明确说出该订单的工单。" if old_ticket.get("status") == "draft" else
                           "原已提交工单不会自动取消。" if old_ticket.get("status") == "submitted" else "")
            return self.clarify(state, f"已停止按订单 {old_order} 继续处理。请提供正确的订单号。{ticket_note}" if old_order else
                                "请提供正确的订单号，我会先核对商品。")
        if len(ids) > 1 or len(ticket_ids) > 1:
            clear_pending(context)
            return self.clarify(state, "请指定一笔订单或一个工单进行处理。")
        trusted = resolve_order(message, context)
        if decision.order_id and decision.order_id != trusted:
            return self.clarify(state, "订单号必须来自当前消息或已经确认的商品，请提供订单号。")
        decision.order_id = trusted
        # 单独“提交”只针对数据库里实际存在的草稿；已提交则返回原单，不能变成订单摘要。
        if re.fullmatch(r"\s*(?:提交|确认提交)[。.!！]?\s*", message) and not context.get("pending_action"):
            record = self.store.get(context.get("active_ticket_id"))
            if record and record["buyer_id"] == self.buyer_id and record["order_id"] == context.get("order_id"):
                if record["status"] == "submitted":
                    return {"ready": False, "status": "already_submitted", "ticket": self.store.public(record),
                            "ticket_events": self.store.events(record),
                            "answer": f"订单 {record['order_id']} 的{request_type_label(record['request_type'])}工单 {record['ticket_id']} 已提交，无需重复提交。尚未执行真实退款。",
                            "events": self.event(state, "confirmation_gate", "success", "已从数据库核验原工单已提交。")}
                if record["status"] == "draft":
                    bind_ticket(context, self.store.public(record))
                    return self.clarify(state, context["confirmation_text"])
            return self.clarify(state, "当前没有待提交的工单草稿。请先调查订单并申请工单。")
        if affirmative(message) and not context.get("pending_action"):
            return self.clarify(state, "当前没有待确认的商品或工单操作。请说明要处理哪笔订单。")
        if decision.ticket_id and decision.ticket_id not in {*ticket_ids, context.get("active_ticket_id")}:
            return self.clarify(state, "请提供要操作的工单号，模型不能选择未在当前对话中指明的工单。")
        if ids and ids[0] != context.get("order_id"):
            # 显式切换订单立即清除旧目标，即使新订单不存在也不能回退处理旧商品。
            clear_pending(context)
            context.update(order_id=None, product_confirmed=False, active_ticket_id=None,
                           active_ticket_type=None, active_product=None, proposed_type=None)
            decision.order_id = ids[0]
            # 新订单不能沿用上一单的工单号；仅保留“正在寻找取消目标”的意图。
            decision.ticket_id = ticket_ids[0] if ticket_ids else None
        # 取消动作比“退款/订单查询”业务词优先。模型的“没有工单”只是推测，
        # 实际候选必须按买家和订单从工单库查出，再由用户确认具体对象。
        if cancel_words or continue_cancel:
            decision.intent = "ticket_action"
            decision.requested_action = "cancel"
            decision.needs_clarification = False
            decision.clarification_question = ""
            decision.request_type = ("price_adjustment" if re.search(r"价差|补差|价保", message) else
                                     "discount_dispute" if re.search(r"优惠争议|优惠工单", message) else
                                     "exchange" if "换货" in message else
                                     "return" if "退货" in message else None)
            decision.ticket_id = ticket_ids[0] if ticket_ids else decision.ticket_id
        elif decision.requested_action == "none" and classify_intent(message) == "return_exchange":
            # “A123求奥退货”等含明确退货词的口语仍可只读调查，不困在泛化追问。
            decision.intent = "return_exchange"
            decision.requested_action = "investigate"
            decision.needs_clarification = False
        explicit_draft = bool(re.search(
            r"申请吧|申请(?:退货|换货|优惠争议|价差|补差|价保|价格保护)|创建(?:退货|换货|争议|价差|补差|价保)|发起(?:退货|换货|争议|价差|补差|价保)|帮我(?:退|换)|我要(?:退货|换货)|我想(?:退货|换货)|给我(?:退货|换货)", message))
        # 调查回答已明确提出草稿且确认过商品时，单独“申请”也算对该提议的明确选择。
        explicit_draft = explicit_draft or bool(re.fullmatch(r"\s*申请[吧。.!！]?\s*", message)
                                              and context.get("proposed_type") and context.get("product_confirmed"))
        if decision.requested_action == "create_draft" and not explicit_draft:
            # 模型不能把质量抱怨或价差查询升级成写草稿；按用户话语回到只读任务。
            kind = classify_intent(message)
            decision.intent = {"order_tracking": "order_query", "unsupported_coupon": "unsupported"}.get(kind, kind)
            decision.requested_action = "investigate"
            decision.request_type = None
            decision.needs_clarification = False
        if re.search(r"申请(?:价差|补差|价保|价格保护)|创建(?:价差|补差|价保)|发起(?:价差|补差|价保)", message) and not cancel_words:
            # 真实模型若把“申请价差”仅理解成查询，守卫仍固定为价保草稿申请。
            decision.intent = "ticket_action"
            decision.requested_action = "create_draft"
            decision.request_type = "price_adjustment"
            decision.needs_clarification = False
        explicit_after_sales = re.search(
            r"(?:申请|创建|发起|我要|我想)(?:退货|换货)", message)
        asking_about_application = re.search(
            r"查询|查一下|进度|状态|条件|资格|能否|可不可以|可以吗|如何|怎么|吗|[?？]", message)
        if explicit_after_sales and not asking_about_application and not cancel_words:
            # 明确建单口令由确定性守卫固定动作和类型，避免模型偶发地把“A123申请退货”误判为追问。
            # 这里只确定用户意图，实际资格、跨类型冲突和幂等仍由数据库工单门核验。
            decision.intent = "ticket_action"
            decision.requested_action = "create_draft"
            decision.request_type = "exchange" if "换货" in explicit_after_sales.group() else "return"
            decision.needs_clarification = False
            decision.clarification_question = ""
        # 模型可能把“查询订单 A123”误写成工单状态查询。只有消息提到工单、
        # 申请进度或明确工单号时才可查工单；普通订单查询沿用只读订单链。
        if (decision.requested_action == "status"
                and not ticket_ids
                and not re.search(r"工单|申请进度|退货进度|换货进度|争议进度", message)
                and classify_intent(message) == "order_tracking"):
            decision.intent = "order_query"
            decision.requested_action = "investigate"
            decision.ticket_id = None
        # “退差价/优惠差额”属于订单金额核对；模型若误解为购买后降价，
        # 在执行前固定为 coupon_dispute，防止调用价保工具链并输出错误结论。
        # “补我价差/降价/价保”仍保留 price_adjustment 语义。
        if (decision.requested_action == "investigate"
                and re.search(r"退差价|差价退款|优惠差额|金额差异|金额不一致", message)
                and not re.search(r"降价|价保|价格保护|买贵了", message)):
            decision.intent = "coupon_dispute"
        action = decision.requested_action
        if negative(message) and context.get("pending_action"):
            action = decision.requested_action = "decline"
        if affirmative(message) and context.get("pending_action"):
            action = decision.requested_action = context["pending_action"]
            decision.intent = "ticket_action"
        if action in {"confirm_product", "confirm_submit", "confirm_cancel"}:
            if negative(message) or not affirmative(message):
                return self.clarify(state, "请明确确认本次操作，或说暂不处理。")
            pending = context.get("pending_confirmation") or {}
            if context.get("pending_action") != action or not context.get("confirmation_text"):
                # 已提交工单的重复肯定只读返回，不能选择其他草稿提交。
                record = self.store.get(context.get("active_ticket_id"))
                if action == "confirm_submit" and record and record["buyer_id"] == self.buyer_id and record["status"] == "submitted":
                    return {"ready": False, "status": "already_submitted", "ticket": self.store.public(record),
                            "ticket_events": self.store.events(record), "answer": "该工单已经提交，不会重复创建或再次退款。",
                            "events": self.event(state, "confirmation_gate", "success", "已提交工单返回原状态。")}
                return self.clarify(state, "当前没有待确认操作，请先申请工单或说明要取消哪一笔工单。")
            if action != "confirm_product":
                record = self.store.get(pending.get("ticket_id"))
                if (not record or record["buyer_id"] != self.buyer_id
                        or record["order_id"] != context.get("order_id")
                        or record["updated_at"] != pending.get("updated_at")
                        or decision.ticket_id not in {None, record["ticket_id"]}):
                    clear_pending(context)
                    return self.clarify(state, "工单状态或确认对象已变化，请重新查询并确认。")
                decision.ticket_id = record["ticket_id"]
            elif pending.get("order_id") != context.get("order_id"):
                clear_pending(context)
                return self.clarify(state, "商品对象已变化，请重新提供订单号。")
        elif action in {"cancel", "status"}:
            candidates = []
            if ticket_ids:
                record = self.store.get(ticket_ids[0])
                if record and record["buyer_id"] == self.buyer_id:
                    candidates = [record]
            elif trusted:
                candidates = self.store.for_order(buyer_id=self.buyer_id, order_id=trusted,
                                                   request_type=decision.request_type)
                active = [item for item in candidates if item["status"] != "cancelled"]
                candidates = active or candidates
                # 明确说“刚才那个工单”才允许在多个候选中沿用当前引用。
                if decision.reference == "current_ticket" and len(candidates) > 1:
                    candidates = [item for item in candidates if item["ticket_id"] == context.get("active_ticket_id")]
            if len(candidates) != 1:
                clear_pending(context)
                if action == "cancel":
                    context["pending_cancel"] = True
                    if not candidates:
                        question = (f"订单 {trusted} 没有可取消的售后工单，请核对订单号或提供工单号。" if trusted else
                                    "当前不能取消原订单或退款交易。如需取消售后申请，请提供订单号或工单号。")
                    else:
                        options = "；".join(f"{request_type_label(item['request_type'])}工单 {item['ticket_id']}（{item['status']}）" for item in candidates)
                        question = f"订单 {trusted} 有多笔工单：{options}。请指定要取消的工单号或类型。"
                else:
                    question = "未找到唯一工单，请提供工单号，或明确订单号和工单类型。"
                return self.clarify(state, question)
            record = candidates[0]
            if ids and record["order_id"] != ids[0] or decision.request_type and record["request_type"] != decision.request_type:
                return self.clarify(state, "订单或工单类型与请求不一致，请重新指定。")
            decision.ticket_id, decision.order_id = record["ticket_id"], record["order_id"]
            if action == "cancel" and record["status"] == "cancelled":
                context["pending_cancel"] = False
                return {"ready": False, "status": "already_cancelled", "ticket": self.store.public(record),
                        "ticket_events": self.store.events(record),
                        "answer": f"订单 {record['order_id']} 的{request_type_label(record['request_type'])}工单 {record['ticket_id']} 已取消；没有执行真实退款。",
                        "events": self.event(state, "cancel_gate", "success", "原工单已取消，未重复写入。")}
            if action == "cancel" and record["status"] != "cancelled":
                clear_pending(context)
                context["pending_cancel"] = True
                context.update(order_id=record["order_id"], active_ticket_id=record["ticket_id"],
                               active_ticket_type=record["request_type"], pending_action="confirm_cancel",
                               pending_confirmation={"action": "confirm_cancel", "ticket_id": record["ticket_id"],
                                                     "order_id": record["order_id"], "updated_at": record["updated_at"]})
                prefix = ("当前不能取消原订单；可以取消售后工单。" if re.search(r"取消.*订单|订单.*取消", message) else
                          "当前尚未执行退款；可以取消售后工单。" if "退款" in message else "")
                context["confirmation_text"] = prefix + f"是否取消订单 {record['order_id']} 的{request_type_label(record['request_type'])}工单 {record['ticket_id']}？请回复确认取消。"
                return self.clarify(state, context["confirmation_text"])
        if decision.intent == "unknown":
            clear_pending(context)
            return self.clarify(state, "我可以帮你查订单、调查退换条件、核对金额或处理工单。请说明需要哪项帮助。")
        if decision.needs_clarification and decision.order_id and action not in {"cancel", "status"}:
            # 有订单号却缺少明确动作时，引导用户选择调查或工单操作，避免死循环。
            question = (f"订单 {decision.order_id} 已识别。你是要查询订单、调查退货条件，"
                        "还是取消这笔订单的售后工单？")
            return self.clarify(state, question)
        return {"dialogue": decision, "ready": True,
                "events": self.event(state, "dialogue_guard", "success", "目标来源和确认状态校验通过。")}

    async def investigate(self, state, decision):
        """编译固定计划并调用只读图；订单参数仍接受原始消息的来源校验。"""
        goal = state["context"].get("pending_goal") or state["message"]
        plan = build_plan(decision, goal)
        return await self.runtime.graph.ainvoke({"message": state["message"], "context": state["context"],
                                                "supplied_plan": plan, "skip_llm_answer": True,
                                                "events": [], "results": {}, "citations": []})

    async def perform(self, state):
        """只读事实确认商品；写动作通过现有工单图，草稿始终与正式提交分离。"""
        decision, context = state["dialogue"], state["context"]
        action = decision.requested_action
        if action == "decline":
            if context.get("pending_action") == "confirm_product":
                context.update(order_id=None, active_product=None, product_confirmed=False)
            if context.get("pending_action") == "confirm_submit" and context.get("active_ticket_id"):
                self.store.append_event(context["active_ticket_id"], session_id=state["session_id"],
                                        action="confirm_declined", result="draft_kept")
            clear_pending(context)
            context["pending_cancel"] = False
            return {"status": "clarify", "answer": "已放弃本次确认操作，尚未提交或取消工单。",
                    "events": self.event(state, "confirmation_gate", "skipped", "用户未确认，未执行写操作。")}
        if action in {"confirm_submit", "confirm_cancel", "status", "cancel"}:
            if action == "status":
                context["pending_cancel"] = False
                record = self.store.get(decision.ticket_id)
                context.update(active_ticket_id=record["ticket_id"], active_ticket_type=record["request_type"])
                self.store.append_event(record["ticket_id"], session_id=state["session_id"], action="query", result=record["status"])
                record = self.store.get(record["ticket_id"])
                return {"status": "completed", "ticket": self.store.public(record), "ticket_events": self.store.events(record),
                        "answer": f"工单 {record['ticket_id']} 当前状态：{record['status']}。尚未执行真实退款。",
                        "events": self.event(state, "ticket_query", "success", "已读取工单状态和操作历史。")}
            graph = self.tickets.confirm_graph if action == "confirm_submit" else self.tickets.cancel_graph
            request = {"ticket_id": decision.ticket_id, "session_id": state["session_id"], "confirmed": True}
            output = await graph.ainvoke({"request": request, "events": []})
            clear_pending(context)
            context["pending_cancel"] = False
            if output.get("ticket"):
                bind_ticket(context, output["ticket"])
            return self.ticket_output(state, output)
        if action == "confirm_product":
            pending = context.get("pending_product_confirmation") or {}
            decision = DialogueDecision.model_validate(pending["decision"])
            decision.requested_action = "investigate"
            if decision.intent == "ticket_action":
                decision.intent = ("coupon_dispute" if decision.request_type == "discount_dispute" else
                                   "price_adjustment" if decision.request_type == "price_adjustment" else "return_exchange")
            context["product_confirmed"] = True
            clear_pending(context)
        if not decision.order_id and decision.intent not in {"price_adjustment", "unsupported"}:
            context["pending_goal"] = state["message"]
            context["pending_intent"] = (("coupon_dispute" if decision.request_type == "discount_dispute" else
                                          "price_adjustment" if decision.request_type == "price_adjustment" else "return_exchange")
                                         if action == "create_draft" else decision.intent)
            question = "请提供需要核对优惠的订单号，例如 A123。" if context["pending_intent"] == "coupon_dispute" else "请提供订单号，例如 A123。"
            clarified = self.clarify(state, question)
            # 保留结构化技能，便于右侧计划显示 pending_goal 的业务分支。
            plan_decision = decision
            if decision.intent == "ticket_action":
                plan_decision = decision.model_copy(update={
                    "intent": context["pending_intent"],
                    "requested_action": "investigate"})
            clarified["plan"] = build_plan(plan_decision, state["message"])
            return clarified
        if not decision.order_id and decision.intent == "price_adjustment":
            # 价差请求也要保存 pending_goal；用户随后只输入 A123 时，必须继续
            # 原价保任务，而不是退回普通订单查询。
            context["pending_goal"] = state["message"]
            context["pending_intent"] = "price_adjustment"
            clarified = self.clarify(state, "请提供订单号，例如 A123，并准备购买价、当前价或降价凭证。")
            clarified["plan"] = build_plan(decision, state["message"])
            return clarified
        # 只有补充订单号或使用模糊当前商品时追问确认；消息已明确写出完整订单号
        # 时视为目标已指定，保持原有“查询订单 A123”验收路径一次完成。
        needs_product_confirmation = (
            self.settings.confirm_product and not context.get("product_confirmed")
            and ((decision.intent == "return_exchange"
                  and (context.get("pending_goal") is not None or decision.reference != "explicit"))
                 or action == "create_draft")
        )
        if decision.order_id and needs_product_confirmation:
            # 确认前只运行 get_order，不提前调查政策或展示工单申请按钮。
            lookup = DialogueDecision(intent="order_query", order_id=decision.order_id, requested_action="investigate")
            output = await self.investigate(state, lookup)
            order = output.get("results", {}).get("get_order", {}).get("order")
            if not order:
                return {**output, "events": [*state["events"], *output["events"]]}
            context.update(order_id=order["order_id"], active_product={"order_id": order["order_id"], "sku": order["sku"], "product": order["product"]},
                           pending_action="confirm_product", pending_confirmation={"action": "confirm_product", "order_id": order["order_id"]},
                           pending_product_confirmation={"decision": decision.model_dump()})
            context["confirmation_text"] = f"查到订单 {order['order_id']}：{order['product']}，实际支付 ¥{order['paid_cents']/100:.2f}。是否就是这件商品？"
            return {"status": "clarify", "answer": context["confirmation_text"], "plan": output["plan"],
                    "results": output["results"], "events": [*state["events"], *output["events"],
                                                              TraceEvent(node="product_confirmation", status="skipped", summary="等待用户确认商品。")],
                    "dialogue": decision.model_copy(update={"needs_clarification": True, "clarification_question": context["confirmation_text"]})}
        if action == "create_draft":
            context["pending_cancel"] = False
            if not decision.request_type:
                return self.clarify(state, "请明确申请退货、换货还是优惠争议工单。")
            output = await self.tickets.draft_graph.ainvoke({
                "request": {"session_id": state["session_id"], "order_id": decision.order_id,
                            "request_type": decision.request_type, "reason": "用户通过聊天申请"},
                "events": [], "citations": []})
            if output.get("ticket"):
                bind_ticket(context, output["ticket"])
            return self.ticket_output(state, output)
        clear_pending(context)
        context["pending_cancel"] = False
        output = await self.investigate(state, decision)
        context["pending_goal"] = None
        context["pending_intent"] = None
        order = output.get("results", {}).get("get_order", {}).get("order")
        if order:
            context.update(order_id=order["order_id"], product_confirmed=True,
                           active_product={"order_id": order["order_id"], "sku": order["sku"], "product": order["product"]})
        assessment = output.get("results", {}).get("assess_after_sales", {})
        discount = output.get("results", {}).get("analyze_discount", {})
        price = output.get("results", {}).get("analyze_price_adjustment", {})
        preferred = "exchange" if "换货" in state["message"] else "return"
        requested_type = ("price_adjustment" if price else preferred if assessment else
                          "discount_dispute" if discount and not discount.get("consistent", True) else None)
        conflict = (self.store.find_conflict_for_order(
            buyer_id=self.buyer_id, order_id=order["order_id"],
            request_type=requested_type)
            if order and requested_type and output["status"] == "completed" else None)
        existing = (self.store.find_active_for_order(
            buyer_id=self.buyer_id, order_id=order["order_id"],
            request_type=requested_type)
            if order and requested_type and output["status"] == "completed" and not conflict else None)
        if conflict:
            # 政策调查和建单资格分开：即使政策允许，也先展示数据库中的有效异类工单并隐藏申请入口。
            output["results"]["ticket_conflict"] = {
                "ticket_id": conflict["ticket_id"], "request_type": conflict["request_type"],
                "status": conflict["status"], "order_id": conflict["order_id"]}
            output["answer"] = output["answer"].replace(
                "本次已完成调查，尚未创建工单或退款。", "本次仅完成调查，未创建新工单或执行退款。")
            if price:
                output["results"]["price_application_blocked"] = True
                output["answer"] = output["answer"].replace(
                    "可按演示政策申请补差，当前只完成价格比较，未执行退款或提交工单。",
                    "当前只完成价格比较，尚未创建价保工单或执行退款。")
            output["answer"] += (f"\n订单 {conflict['order_id']} 已有{request_type_label(conflict['request_type'])}工单 "
                                 f"{conflict['ticket_id']}（{conflict['status']}）；目前不能创建"
                                 f"{request_type_label(requested_type)}草稿。若要改办，请先提出取消原工单并明确确认。")
        elif existing:
            # 同类型有效工单是幂等恢复目标；只读调查直接展示数据库状态，不调用建单节点写入 draft_reused 历史。
            output["results"]["ticket_existing"] = {
                "ticket_id": existing["ticket_id"], "request_type": existing["request_type"],
                "status": existing["status"], "order_id": existing["order_id"]}
            # 已有工单时明确限定“本次没有新建”，避免与随后展示的原工单状态看起来矛盾。
            output["answer"] = output["answer"].replace(
                "本次已完成调查，尚未创建工单或退款。", "本次仅完成调查，未创建新工单或执行退款。")
            if price:
                output["answer"] = output["answer"].replace(
                    "可按演示政策申请补差，当前只完成价格比较，未执行退款或提交工单。",
                    "本次仅完成价格比较，未执行新的退款或工单操作。")
            output["answer"] += (f"\n订单 {existing['order_id']} 已有{request_type_label(existing['request_type'])}工单 "
                                 f"{existing['ticket_id']}（{existing['status']}）；"
                                 "可查询或明确提出取消原工单，无需重复申请。")
        context["proposed_type"] = None
        if output["status"] == "completed":
            if discount and not discount["consistent"] and not existing:
                context["proposed_type"] = "discount_dispute"
            elif price and price.get("eligible") and price.get("difference_cents", 0) > 0 and not conflict and not existing:
                context["proposed_type"] = "price_adjustment"
            elif assessment and not conflict and not existing:
                eligible = assessment.get(preferred + "_eligible")
                if preferred == "exchange":
                    eligible = eligible and output["results"].get("check_inventory", {}).get("available", 0) > 0
                if eligible:
                    context["proposed_type"] = preferred
            if context["proposed_type"]:
                output["answer"] += f"\n要为你创建{request_type_label(context['proposed_type'])}申请草稿吗？可回复“申请吧”。"
        return {**output, "events": [*state["events"], *output["events"]]}

    def ticket_output(self, state, output):
        """把工单调查证据、实际结果和持久化历史完整传回聊天响应。"""
        context = state["context"]
        answer = output["message"] + "尚未执行真实退款。"
        results = dict(output.get("investigation", {}).get("results", {}))
        if output.get("error_code"):
            # 聊天响应保留业务拒绝码，网页右侧和自动验收可以区分冲突与系统故障。
            results["ticket_error_code"] = output["error_code"]
        ticket = output.get("ticket")
        if ticket:
            answer = f"订单 {ticket.order_id} · {request_type_label(ticket.request_type)} · {ticket.status} · 工单号 {ticket.ticket_id}；" + answer
            if ticket.status == "draft":
                answer += "\n" + (context.get("confirmation_text") or "请明确确认提交工单。")
        return {"status": output["status"], "answer": answer, "ticket": ticket,
                "ticket_events": output.get("ticket_events", []), "citations": output.get("citations", []),
                "results": results,
                "events": [*state["events"], *output["events"]]}

    async def respond(self, state):
        """业务拒绝时保留数据库原始原因；其他回答才交给模型润色并校验事实。"""
        if state.get("status") == "rejected":
            # 冲突工单号和状态必须来自数据库，不让模型改写；这也是预期拒绝而非模型故障。
            return {"events": self.event(state, "llm_answer", "skipped", "资格门已拒绝，保留数据库核验的原始原因。")}
        result = await self.runtime.llm_answer({**state, "results": {**state.get("results", {}),
                                                                    "ticket": state["ticket"].model_dump() if state.get("ticket") else None}})
        return result
