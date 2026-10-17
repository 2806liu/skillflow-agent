"""售后、优惠争议与价保工单的 LangGraph 状态图：调查、资格门、草稿和确认提交。"""

from typing import Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from pydantic import Field

from skillflow.business.sqlite_tickets import SqliteTicketStore
from skillflow.contracts import Contract, DialogueDecision, Evidence, TicketEvent, TicketRecord, ToolResult, TraceEvent
from skillflow.dialogue import build_plan
from skillflow.streaming import publish_trace
from skillflow.tools import TypedCoreTool



def request_type_label(request_type: str) -> str:
    """将内部工单类型转换为用户可读名称，未知类型保留原值便于审计。"""
    return {"return": "退货", "exchange": "换货", "discount_dispute": "优惠争议",
            "price_adjustment": "价保补差"}.get(request_type, request_type)

class DraftToolInput(Contract):
    """草稿工具只接受服务端已核验的数据，用户身份由工具闭包注入。"""
    session_id: str
    order_id: str = Field(pattern=r"^A[0-9]{3,8}$")
    sku: str
    request_type: Literal["return", "exchange", "discount_dispute", "price_adjustment"]
    reason: str
    source_run_id: str
    # 金额争议和价保工单写入可复算差额与引用；普通退换货工单使用空字典。
    metadata: dict = Field(default_factory=dict)


class SubmitToolInput(Contract):
    """提交工具只接受工单号与会话；工单归属仍需在工具内复核。"""
    ticket_id: str
    session_id: str


class CancelToolInput(Contract):
    """取消工具只接受工单号与会话，避免跨会话修改工单。"""
    ticket_id: str
    session_id: str

    #状态图共享状态
class TicketState(TypedDict, total=False):
    """一次工单操作在 LangGraph 节点间传递的最小状态。"""
    request: dict
    status: str
    message: str
    investigation: dict
    citations: list[Evidence]
    events: list[TraceEvent]
    ticket: TicketRecord | None
    ticket_events: list[TicketEvent]
    error_code: str | None


class TicketRuntime:
    """将工单写入动作放在确认和资格门之后，并通过 LangChain 工具执行。"""

    def __init__(self, investigation_runtime, ticket_store: SqliteTicketStore, buyer_id: str):
        self.investigation_runtime = investigation_runtime
        self.ticket_store = ticket_store
        # buyer_id 是服务端注入的工单归属身份；session_id 只用于记录聊天来源。
        self.buyer_id = buyer_id

        async def save_draft(inputs: DraftToolInput):
            """业务写入函数只访问项目 B 的本地工单库。"""
            record, created, conflict = ticket_store.create_draft(
                buyer_id=buyer_id, session_id=inputs.session_id, order_id=inputs.order_id,
                sku=inputs.sku, request_type=inputs.request_type, reason=inputs.reason,
                source_run_id=inputs.source_run_id, metadata=inputs.metadata)
            if conflict:
                conflict_type = request_type_label(conflict["request_type"])
                return ToolResult(
                    status="empty", error_code="TICKET_CONFLICT",
                    message=f"订单 {inputs.order_id} 已有{conflict_type}工单 {conflict['ticket_id']}（{conflict['status']}），请先关闭或取消原工单。",
                    data={"conflict_ticket": ticket_store.public(conflict).model_dump()})
            if record is None:
                return ToolResult(status="empty", error_code="TICKET_UNAVAILABLE", message="当前会话无法创建该工单。")
            return ToolResult(status="success", data={"ticket": ticket_store.public(record).model_dump(), "created": created})

        async def submit(inputs: SubmitToolInput):
            """正式提交前在工具内部再次验证会话归属。"""
            record = ticket_store.get(inputs.ticket_id)
            if not record or record["buyer_id"] != buyer_id:
                return ToolResult(status="empty", error_code="TICKET_NOT_FOUND", message="工单不存在。")
            if record["status"] == "submitted":
                record = ticket_store.append_event(record["ticket_id"], session_id=inputs.session_id,
                                                    action="confirm_repeated", result="already_submitted")
                # 重复确认先走幂等返回，即使历史数据中存在另一种类型工单也不改变原状态。
                return ToolResult(status="success", data={"ticket": ticket_store.public(record).model_dump(), "changed": False})
            # 退货和换货互斥；价保也不能与有效退换工单并行，优惠争议只做金额复核。
            conflict = None
            if record["request_type"] in {"return", "exchange"}:
                opposite = "exchange" if record["request_type"] == "return" else "return"
                conflict = ticket_store.find_active_for_order(
                    buyer_id=record["buyer_id"], order_id=record["order_id"],
                    request_type=opposite, exclude_ticket_id=record["ticket_id"])
            if conflict:
                conflict_type = request_type_label(conflict["request_type"])
                return ToolResult(
                    status="empty", error_code="TICKET_CONFLICT",
                    message=f"订单 {record['order_id']} 已有{conflict_type}工单 {conflict['ticket_id']}（{conflict['status']}），请先关闭或取消原工单。",
                    data={"conflict_ticket": ticket_store.public(conflict).model_dump()})
            record, changed = ticket_store.confirm(inputs.ticket_id, session_id=inputs.session_id, buyer_id=buyer_id)
            if not record or record["status"] == "cancelled":
                return ToolResult(status="empty", error_code="TICKET_NOT_CONFIRMABLE", message="工单不能提交。")
            return ToolResult(status="success", data={"ticket": ticket_store.public(record).model_dump(), "changed": changed})

        async def cancel(inputs: CancelToolInput):
            """取消业务函数在写入前复核会话归属，并保持重复取消幂等。"""
            record = ticket_store.get(inputs.ticket_id)
            if not record or record["buyer_id"] != buyer_id:
                return ToolResult(status="empty", error_code="TICKET_NOT_FOUND", message="工单不存在。")
            record, changed = ticket_store.cancel(inputs.ticket_id, session_id=inputs.session_id, buyer_id=buyer_id)
            return ToolResult(status="success", data={"ticket": ticket_store.public(record).model_dump(), "changed": changed})

        # 与只读调查工具相同：TypedCoreTool 同时保持 CoreCoder 协议和 LangChain StructuredTool 调用。
        self.draft_tool = TypedCoreTool("create_ticket_draft", "保存售后或价保工单草稿", DraftToolInput, save_draft)
        self.submit_tool = TypedCoreTool("submit_ticket", "确认后提交售后或价保工单", SubmitToolInput, submit)
        self.cancel_tool = TypedCoreTool("cancel_ticket", "取消已有售后或价保工单", CancelToolInput, cancel)


        #草稿图 draft_graph，调查 → 资格门 → 创建草稿。调查没完成或资格不满足，直接 END，不创建草稿。
        draft = StateGraph(TicketState)
        for name in ("investigate", "eligibility", "draft"):
            draft.add_node(name, getattr(self, name))
        draft.add_edge(START, "investigate")
        draft.add_conditional_edges("investigate", lambda s: "eligibility" if s["status"] == "investigated" else END)
        draft.add_conditional_edges("eligibility", lambda s: "draft" if s["status"] == "eligible" else END)
        draft.add_edge("draft", END)
        self.draft_graph = draft.compile()

        #确认图 confirm_graph,确认门 → 提交。确认门不通过就 END，不会走到提交
        confirm = StateGraph(TicketState)
        confirm.add_node("confirmation_gate", self.confirmation_gate)
        confirm.add_node("submit", self.submit)
        confirm.add_edge(START, "confirmation_gate")
        confirm.add_conditional_edges("confirmation_gate", lambda s: "submit" if s["status"] == "confirmed" else END)
        confirm.add_edge("submit", END)
        self.confirm_graph = confirm.compile()

        #取消图 cancel_graph，取消门 → 取消。
        cancel = StateGraph(TicketState)
        cancel.add_node("cancel_gate", self.cancel_gate)
        cancel.add_node("cancel", self.cancel)
        cancel.add_edge(START, "cancel_gate")
        cancel.add_conditional_edges("cancel_gate", lambda s: "cancel" if s["status"] == "cancel_ready" else END)
        cancel.add_edge("cancel", END)
        self.cancel_graph = cancel.compile()

    @staticmethod
    def event(state, node, status, summary, **kwargs):
        """每个工单节点追加 Trace，网页可查看调查与写入发生的真实顺序。"""
        event = TraceEvent(node=node, status=status, summary=summary, **kwargs)
        publish_trace(event)
        return [*state.get("events", []), event]

    async def investigate(self, state):
        """用服务端固定计划重查订单和政策，输出资格门所需的真实工具结果。

        输入为已校验的工单请求；模型不参与二次规划，避免合法申请因漏掉
        必需步骤而被误报为缺少政策。工具或证据失败仍会阻止草稿写入。
        """
        order_id = state["request"]["order_id"]
        request_type = state["request"]["request_type"]
        # 工单调查只使用固定只读计划；价保与优惠分别走自己的证据和计算链。
        message = (f"查询订单 {order_id}，核对优惠金额、原价、实付金额和退款依据。"
                   if request_type == "discount_dispute" else
                   f"查询订单 {order_id}，结合价保政策比较购买价、当前价和降价时间。"
                   if request_type == "price_adjustment" else
                   f"查询订单 {order_id}，结合店铺政策，调查这件商品能否退换。")
        decision = DialogueDecision(
            intent="coupon_dispute" if request_type == "discount_dispute" else
                   "price_adjustment" if request_type == "price_adjustment" else "return_exchange",
            order_id=order_id, reference="explicit", requested_action="investigate")
        # 固定工具顺序仍交给调查图的 validate 校验，并从工具重新读取最新事实。
        plan = build_plan(decision, message)
        investigation = await self.investigation_runtime.graph.ainvoke({
            "message": message,
            "context": {"order_id": None, "messages": []}, "supplied_plan": plan,
            "skip_llm_answer": True, "events": [], "results": {}, "citations": []})
        if investigation["status"] != "completed":
            label = request_type_label(request_type)
            # 计划/工具故障与证据不足分开告知，避免把系统错误说成商家政策拒绝。
            flow_failed = investigation["status"] in {"planner_failed", "plan_rejected", "execution_failed", "tool_failed"}
            message = (f"创建{label}工单前的调查流程未完成，请稍后重试；未创建工单。" if flow_failed
                       else f"没有足够的订单或政策依据，不能创建{label}工单。")
            return {"status": "rejected", "message": message,
                    "investigation": investigation, "citations": investigation.get("citations", []),
                    "events": [*investigation["events"],
                               TraceEvent(node="ticket_investigate", status="failed", summary=f"{label}调查未完成。"),
                               TraceEvent(node="ticket_gate", status="failed", summary=f"调查未完成，禁止创建{label}工单。" )]}
        return {"status": "investigated", "investigation": investigation,
                "citations": investigation.get("citations", []),
                "events": [*investigation["events"], TraceEvent(node="ticket_investigate", status="success", summary="只读调查完成。")]}

    async def eligibility(self, state):
        """草稿前验证证据与实时工单冲突；不通过时停在资格门，写入节点不会运行。"""
        results = state["investigation"]["results"]
        request_type = state["request"]["request_type"]
        conflict = self.ticket_store.find_conflict_for_order(
            buyer_id=self.buyer_id, order_id=state["request"]["order_id"], request_type=request_type)
        if conflict:
            # 调查仍保留政策结论；跨类型工单冲突是另一道资格门，不能先报告成功再在草稿节点失败。
            label = request_type_label(conflict["request_type"])
            reason = (f"订单 {conflict['order_id']} 已有{label}工单 {conflict['ticket_id']}"
                      f"（{conflict['status']}），请先处理原工单")
            return {"status": "rejected", "error_code": "TICKET_CONFLICT",
                    "message": reason + "。",
                    "events": self.event(state, "ticket_gate", "failed", reason,
                                         details={"ticket_id": conflict["ticket_id"],
                                                  "request_type": conflict["request_type"],
                                                  "status": conflict["status"]})}
        if request_type == "discount_dispute":
            analysis = results.get("analyze_discount")
            if not analysis:
                reason = "缺少优惠金额核对结果"
                return {"status": "rejected", "message": f"当前不能创建优惠争议工单：{reason}。",
                        "events": self.event(state, "ticket_gate", "failed", reason)}
            if analysis.get("consistent"):
                reason = "订单原价、优惠和实付金额核对一致"
                return {"status": "rejected", "message": "当前金额核对一致，不需要创建优惠争议工单。",
                        "events": self.event(state, "ticket_gate", "failed", reason)}
            return {"status": "eligible", "events": self.event(state, "ticket_gate", "success", "优惠金额存在差异，允许创建争议工单。")}

        if request_type == "price_adjustment":
            analysis = results.get("analyze_price_adjustment")
            if not analysis or not analysis.get("citations_used"):
                reason = "缺少价保计算结果或政策引用"
            elif not analysis.get("eligible") or analysis.get("difference_cents", 0) <= 0:
                reason = "当前价格或价保期限不满足演示申请条件"
            else:
                reason = ""
            if reason:
                return {"status": "rejected", "message": f"当前不能创建价保补差工单：{reason}。",
                        "events": self.event(state, "ticket_gate", "failed", reason)}
            return {"status": "eligible", "events": self.event(state, "ticket_gate", "success", "价差和期限已核实，允许创建价保草稿。")}

        assessment = results["assess_after_sales"]
        stock = results.get("check_inventory", {}).get("available", 0)
        eligible = assessment["return_eligible"] if request_type == "return" else assessment["exchange_eligible"] and stock > 0
        if not eligible:
            reason = "退货期限或质量条件不满足" if request_type == "return" else "换货政策条件不满足或当前库存为 0"
            return {"status": "rejected", "message": f"当前不能创建工单：{reason}。",
                    "events": self.event(state, "ticket_gate", "failed", reason)}
        return {"status": "eligible", "events": self.event(state, "ticket_gate", "success", "资格门通过。")}

    async def draft(self, state):
        """通过 LangChain StructuredTool 保存草稿；本节点不提交正式工单。"""
        request = state["request"]
        order = state["investigation"]["results"]["get_order"]["order"]
        metadata = {}
        if request["request_type"] == "discount_dispute":
            analysis = state["investigation"]["results"].get("analyze_discount", {})
            # 只保存已通过政策证据核对的数值，供工单查询和人工复核复算。
            metadata = {key: analysis[key] for key in (
                "difference_cents", "expected_paid_cents", "paid_cents", "citations_used"
            ) if key in analysis}
        elif request["request_type"] == "price_adjustment":
            analysis = state["investigation"]["results"].get("analyze_price_adjustment", {})
            # 保存重新核验时需要比较的金额、降价时间和政策依据；不保存支付动作。
            metadata = {key: analysis[key] for key in (
                "purchase_price_cents", "current_price_cents", "difference_cents",
                "price_drop_days_ago", "price_protection_days", "price_protection_basis", "citations_used"
            ) if key in analysis}
            metadata["policy_document_ids"] = sorted({hit.document_id for hit in state["citations"]})
        result = await self.draft_tool.call({**request, "sku": order["sku"],
                                             "source_run_id": str(uuid4()), "metadata": metadata})
        if result.status != "success":
            return {"status": "rejected", "error_code": result.error_code, "message": result.message,
                    "events": self.event(state, "draft", "failed", result.message)}
        ticket = TicketRecord.model_validate(result.data["ticket"])
        created = result.data["created"]
        label = request_type_label(ticket.request_type)
        if ticket.status == "submitted":
            status, message = "already_submitted", f"该订单的{label}工单已经提交，已恢复原工单。"
        elif created:
            status, message = "draft_created", f"{label}工单草稿已生成，请明确确认后提交。"
        else:
            status, message = "draft_created", f"已恢复该买家的{label}工单草稿，请明确确认后提交。"
        return {"status": status, "message": message, "ticket": ticket,
                "ticket_events": self.ticket_store.events(self.ticket_store.get(ticket.ticket_id)),
                "events": self.event(state, "draft", "success", "草稿已保存，尚未提交。", details={"created": result.data["created"]})}

    async def confirmation_gate(self, state):
        """明确确认、会话归属和草稿状态三者满足才允许进入提交节点。"""
        request = state["request"]
        record = self.ticket_store.get(request["ticket_id"])
        if not record or record["buyer_id"] != self.buyer_id:
            return {"status": "rejected", "message": "工单不存在。",
                    "error_code": "TICKET_NOT_FOUND",
                    "events": self.event(state, "confirmation_gate", "failed", "工单不存在或不属于当前会话。")}
        if record["status"] == "submitted":
            record = self.ticket_store.append_event(record["ticket_id"], session_id=request["session_id"],
                                                    action="confirm_repeated", result="already_submitted")
            return {"status": "already_submitted", "message": "该工单已经提交，重复确认不会创建新工单。",
                    "ticket": self.ticket_store.public(record),
                    "ticket_events": self.ticket_store.events(record),
                    "events": self.event(state, "confirmation_gate", "success", "工单已提交，进入幂等返回路径。")}
        if not request["confirmed"]:
            record = self.ticket_store.append_event(record["ticket_id"], session_id=request["session_id"],
                                                    action="confirm_declined", result="draft_kept")
            state["ticket_events"] = self.ticket_store.events(record)
            return {"status": "draft_created", "message": "已保留工单草稿，尚未提交。",
                    "ticket": self.ticket_store.public(record),
                    "events": self.event(state, "confirmation_gate", "skipped", "用户未确认，未执行提交。")}
        if record["status"] == "cancelled":
            record = self.ticket_store.append_event(record["ticket_id"], session_id=request["session_id"],
                                                    action="confirm_cancelled", result="rejected",
                                                    error_code="TICKET_CANCELLED")
            state["ticket_events"] = self.ticket_store.events(record)
            return {"status": "rejected", "message": "工单已取消。",
                    "error_code": "TICKET_CANCELLED",
                    "events": self.event(state, "confirmation_gate", "failed", "已取消工单不能提交。")}
        if record["request_type"] == "price_adjustment":
            # 价保确认时重新读取订单、价格凭证与政策；草稿快照变化必须重新申请。
            fresh = await self.investigate({"request": {"order_id": record["order_id"],
                                                        "request_type": "price_adjustment"},
                                            "events": [], "citations": []})
            fresh_gate = await self.eligibility({**fresh, "request": {"order_id": record["order_id"],
                                                                      "request_type": "price_adjustment"}}) if fresh["status"] == "investigated" else fresh
            analysis = fresh.get("investigation", {}).get("results", {}).get("analyze_price_adjustment", {})
            compared = ("purchase_price_cents", "current_price_cents", "difference_cents",
                        "price_drop_days_ago", "price_protection_days", "price_protection_basis",
                        "citations_used")
            current_ids = sorted({hit.document_id for hit in fresh.get("citations", [])})
            if (fresh_gate["status"] != "eligible" or
                    any(record["metadata"].get(key) != analysis.get(key) for key in compared) or
                    record["metadata"].get("policy_document_ids") != current_ids):
                updated = self.ticket_store.append_event(record["ticket_id"], session_id=request["session_id"],
                                                         action="confirm_revalidation_failed", result="rejected",
                                                         error_code="PRICE_EVIDENCE_CHANGED")
                return {"status": "rejected", "message": "价保政策或价格凭证已变化，原草稿不能提交；请取消后重新申请。",
                        "error_code": "PRICE_EVIDENCE_CHANGED", "ticket": self.ticket_store.public(updated),
                        "ticket_events": self.ticket_store.events(updated),
                        "events": [*state.get("events", []), *fresh_gate["events"],
                                   TraceEvent(node="confirmation_gate", status="failed", summary="价保确认复核失败，未提交工单。") ]}
            state["events"] = [*state.get("events", []), *fresh_gate["events"]]
        return {"status": "confirmed", "events": self.event(state, "confirmation_gate", "success", "已收到明确确认。")}

    async def submit(self, state):
        """通过 LangChain StructuredTool 提交；重复确认复用原工单。"""
        result = await self.submit_tool.call({"ticket_id": state["request"]["ticket_id"],
                                              "session_id": state["request"]["session_id"]})
        if result.status != "success":
            return {"status": "rejected", "error_code": result.error_code, "message": result.message,
                    "events": self.event(state, "submit", "failed", result.message)}
        changed = result.data["changed"]
        # 确认接口只传 ticket_id 和 session_id，类型必须从服务端工单记录读取。
        record = self.ticket_store.get(state["request"]["ticket_id"])
        request_type = record["request_type"] if record else "ticket"
        return {"status": "submitted" if changed else "already_submitted",
                "message": (f"{request_type_label(request_type)}工单已提交。" if changed else
                           "该工单已经提交，重复确认不会创建新工单。"),
                "ticket": TicketRecord.model_validate(result.data["ticket"]),
                "ticket_events": self.ticket_store.events(self.ticket_store.get(state["request"]["ticket_id"])),
                "events": self.event(state, "submit", "success", "工单状态已保存。", details={"idempotent": not changed})}

    async def cancel_gate(self, state):
        """取消前验证会话和工单状态，避免把不存在的工单当作已取消。"""
        request = state["request"]
        record = self.ticket_store.get(request["ticket_id"])
        if not record or record["buyer_id"] != self.buyer_id:
            return {"status": "rejected", "message": "工单不存在。",
                    "error_code": "TICKET_NOT_FOUND",
                    "events": self.event(state, "cancel_gate", "failed", "工单不存在或不属于当前会话。")}
        if record["status"] == "cancelled":
            record = self.ticket_store.append_event(record["ticket_id"], session_id=request["session_id"],
                                                    action="cancel_repeated", result="already_cancelled")
            state["ticket_events"] = self.ticket_store.events(record)
            return {"status": "cancelled", "message": "工单已经取消。",
                    "ticket": self.ticket_store.public(record),
                    "events": self.event(state, "cancel_gate", "success", "重复取消返回原状态。")}
        return {"status": "cancel_ready", "events": self.event(state, "cancel_gate", "success", "取消权限校验通过。")}

    async def cancel(self, state):
        """通过 LangChain 工具执行取消，让取消动作也进入完整 Trace。"""
        request = state["request"]
        result = await self.cancel_tool.call({"ticket_id": request["ticket_id"], "session_id": request["session_id"]})
        if result.status != "success":
            return {"status": "rejected", "error_code": result.error_code, "message": result.message,
                    "events": self.event(state, "cancel", "failed", result.message)}
        changed = result.data["changed"]
        # 取消工具已经把事件持久化，这里把最新历史放回 LangGraph 状态供 API 和网页展示。
        state["ticket_events"] = self.ticket_store.events(self.ticket_store.get(request["ticket_id"]))
        request_type = self.ticket_store.get(request["ticket_id"])["request_type"]
        return {"status": "cancelled", "message": f"{request_type_label(request_type)}工单已取消。",
                "ticket": TicketRecord.model_validate(result.data["ticket"]),
                "events": self.event(state, "cancel", "success", "工单状态已取消。", details={"changed": changed})}
