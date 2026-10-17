"""质量事件咨询与售后入口。

订单事实只从 B 业务库按买家查询；批次判断与资格判断由确定性代码执行。
回答里保留事件版本和原因码，模型不能把用户自报批次变成可信订单字段。
"""

import asyncio
from uuid import uuid4

from skillflow.business.surge import BusinessRepository
from skillflow.business.sqlite_tickets import SqliteTicketStore
from skillflow.contracts import (ChatResponse, Plan, PlanStep, TicketResponse, TraceEvent)
from skillflow.planner import extract_order_ids
from skillflow.streaming import publish


class IncidentService:
    def __init__(self, repository: BusinessRepository, tickets: SqliteTicketStore, model_mode: str):
        self.repository = repository
        self.tickets = tickets
        self.model_mode = model_mode

    def should_handle(self, message: str, context: dict, buyer_id: str) -> bool:
        """已知充电宝订单及其后续咨询走确定性流程，其余仍走原 Agent。"""
        if any(term in message for term in ("充电宝", "批次", "质量事件", "替代品")):
            return True
        ids = extract_order_ids(message)
        order_id = ids[0] if len(ids) == 1 else context.get("incident_order_id")
        order = self.repository.get_order(order_id, buyer_id) if order_id else None
        return bool(order and order.get("category") == "power_bank")

    async def chat(self, *, message: str, session_id: str, run_id: str,
                   buyer_id: str, context: dict) -> ChatResponse:
        """只读调查可并行；同会话上下文顺序由 API 的会话锁保护。"""
        ids = extract_order_ids(message)
        order_id = ids[0] if len(ids) == 1 else context.get("incident_order_id")
        if not order_id:
            if any(term in message for term in ("替代品", "替换商品", "推荐")):
                # 无订单时只给目录内有库存且不在生效事件范围内的候选品，不做个人售后判断。
                choices = await asyncio.to_thread(self.repository.alternatives)
                publish("tool_result", {"tool": "alternatives", "status": "success"})
                answer = ("项目 B 目录中有库存的充电宝候选：" + "；".join(
                    f"{item['name']}（{item['sku']}，库存 {item['available']}）" for item in choices)
                    if choices else "当前目录中没有可推荐的有库存候选品。")
                answer += " 这仅是品类、库存和已生效事件筛选，不保证商品绝对安全。"
                return ChatResponse(run_id=run_id, session_id=session_id, status="completed", answer=answer,
                                    model_mode=self.model_mode,
                                    result={"alternatives": choices, "route": "incident_deterministic"},
                                    events=[TraceEvent(node="alternatives", status="success", summary="查询项目 B 候选目录")])
            if "买过" in message or "订单" in message:
                orders = await asyncio.to_thread(self.repository.list_orders, buyer_id)
                publish("tool_result", {"tool": "buyer_order_lookup", "status": "success"})
                relevant = [item for item in orders if item.get("category") == "power_bank"]
                answer = ("你在本演示库中的充电宝订单：" + "、".join(item["order_id"] for item in relevant)
                          if relevant else "本演示库未找到你的充电宝订单。")
                return ChatResponse(run_id=run_id, session_id=session_id, status="completed", answer=answer,
                                    model_mode=self.model_mode, result={"orders": relevant, "route": "incident_deterministic"},
                                    events=[TraceEvent(node="buyer_order_lookup", status="success", summary="仅按当前买家查询订单")])
            events = await asyncio.to_thread(self.repository.active_event_summary)
            publish("tool_result", {"tool": "approved_event_lookup", "status": "success"})
            answer = ("已审核的模拟质量事件：" + "；".join(
                f"{item['event_id']} v{item['version']}：{item['explanation']}" for item in events)
                if events else "当前没有已审核生效的质量事件；新闻线索不能用作售后资格依据。")
            answer += " 如需核对自己的商品，请提供订单号。"
            return ChatResponse(run_id=run_id, session_id=session_id, status="completed", answer=answer,
                                model_mode=self.model_mode, result={"events": events, "route": "incident_deterministic"},
                                events=[TraceEvent(node="approved_event_lookup", status="success", summary="仅读取生效事件")])
        # 集中咨询只持久化一次处理中提示；事实、原因码和引用完整保存在最终结果中。
        publish("status", {"code": "INCIDENT_INVESTIGATING"})
        order = await asyncio.to_thread(self.repository.get_order, order_id, buyer_id)
        if not order:
            return ChatResponse(run_id=run_id, session_id=session_id, status="rejected", answer="无权访问（ORDER_FORBIDDEN）。",
                                model_mode=self.model_mode, result={"error_code": "ORDER_FORBIDDEN"})
        if order["category"] != "power_bank":
            return ChatResponse(run_id=run_id, session_id=session_id, status="clarify",
                                answer="这不是本次充电宝质量事件的商品，请说明需要查询的业务。",
                                model_mode=self.model_mode)
        context["incident_order_id"] = order_id
        assessment = await asyncio.to_thread(self.repository.evaluate, order_id, buyer_id)
        plan = Plan(decision="execute", skill="return_exchange", goal=message[:500], order_id=order_id,
                    steps=[PlanStep(step_id="order", tool="get_order", arguments={"order_id": order_id}),
                           PlanStep(step_id="batch", tool="match_quality_incident", depends_on=["order"])])
        result = {"get_order": {"order": order}, "incident": assessment, "route": "incident_deterministic"}
        answer = (f"订单 {order_id}：{order['product']}；批次凭证来源：{order['batch_source']}。"
                  f" 核对结果：{assessment['decision']}（{assessment['reason_code']}）。")
        if assessment["event_id"]:
            answer += f" 依据已审核质量事件 {assessment['event_id']} v{assessment['event_version']}。"
        if assessment["manual_review"]:
            answer += " 当前不能自动判定退货资格，请转人工核实；也不能据此认定商品安全。"
        elif assessment["return_eligible"]:
            result["assess_after_sales"] = {"return_eligible": True, "exchange_eligible": False}
            existing = self.tickets.find_active_for_order(buyer_id=buyer_id, order_id=order_id, request_type="return")
            if existing:
                # 已有工单时只说明现状，避免在重复咨询中继续引导创建新草稿。
                result["ticket_existing"] = {"ticket_id": existing["ticket_id"], "status": existing["status"]}
                answer += f" 已有退货工单 {existing['ticket_id']}（{existing['status']}），无需重复申请；未执行真实退款。"
            else:
                answer += " 可以创建退货申请草稿；明确确认后才会提交，不执行真实退款。"
        else:
            answer += " 本事件不提供自动退货资格结论；如有其他售后依据请人工核实。"
        if any(term in message for term in ("替代品", "替换产品", "推荐")):
            choices = await asyncio.to_thread(self.repository.alternatives, order["sku"])
            result["alternatives"] = choices
            if choices:
                answer += " 目录中有库存的同类候选：" + "；".join(
                    f"{item['name']}（{item['sku']}，库存 {item['available']}）" for item in choices)
                answer += "。仅按类别和库存筛选，不保证全面安全。"
            else:
                answer += " 当前目录无有库存的同类候选商品。"
        events = [TraceEvent(node="buyer_order_lookup", status="success", summary="按服务端身份读取本人订单"),
                  TraceEvent(node="quality_scope_match", status="success", summary="按可信批次与已审核事件版本确定性匹配",
                             details={"decision": assessment["decision"], "reason_code": assessment["reason_code"]})]
        return ChatResponse(run_id=run_id, session_id=session_id, status="completed", answer=answer,
                            model_mode=self.model_mode, plan=plan, result=result, events=events)

    async def create_draft(self, *, buyer_id: str, session_id: str, order_id: str,
                           reason: str, source_run_id: str) -> TicketResponse:
        """草稿前重新读事件；缺批次、未审核、撤回或超期无特则拒绝自动建单。"""
        order = await asyncio.to_thread(self.repository.get_order, order_id, buyer_id)
        assessment = await asyncio.to_thread(self.repository.evaluate, order_id, buyer_id)
        if not order or not assessment or not assessment["return_eligible"]:
            code = assessment["reason_code"] if assessment else "ORDER_FORBIDDEN"
            return TicketResponse(status="rejected", error_code=code,
                                  message="当前不能自动创建退货草稿，请人工核实订单、批次和已审核规则。")
        metadata = {"incident_id": assessment["event_id"], "incident_version": assessment["event_version"],
                    "batch_code": order["batch_code"], "batch_source": order["batch_source"],
                    "reason_code": assessment["reason_code"]}
        record, created, conflict = await asyncio.to_thread(
            self.tickets.create_draft, buyer_id=buyer_id, session_id=session_id, order_id=order_id,
            sku=order["sku"], request_type="return", reason=reason, source_run_id=source_run_id,
            metadata=metadata)
        if conflict:
            return TicketResponse(status="rejected", error_code="TICKET_CONFLICT", message="该订单已有冲突工单。")
        if record is None:
            return TicketResponse(status="rejected", error_code="INCIDENT_CHANGED_OR_UNVERIFIED",
                                  message="事件或批次依据已变化，未创建草稿；请重新调查。")
        return TicketResponse(status="draft_created" if created else "already_submitted" if record["status"] == "submitted" else "draft_created",
                              message="退货草稿已创建，请明确确认后提交；未执行真实退款。" if created else "已恢复原工单。",
                              ticket=self.tickets.public(record), ticket_events=self.tickets.events(record))

    async def confirm(self, *, buyer_id: str, session_id: str, ticket_id: str, confirmed: bool) -> TicketResponse:
        """明确确认才调用同一 SQLite 短事务；撤回或变更时返回稳定错误码。"""
        if not confirmed:
            record = await asyncio.to_thread(self.tickets.append_event, ticket_id, session_id=session_id,
                                             action="confirm_declined", result="draft_kept")
            return TicketResponse(status="draft_created", message="未确认，草稿保留。",
                                  ticket=self.tickets.public(record) if record else None)
        record, changed, error = await asyncio.to_thread(self.tickets.confirm_incident, ticket_id,
                                                         session_id=session_id, buyer_id=buyer_id)
        if error:
            return TicketResponse(status="rejected", error_code=error,
                                  message="事件或批次依据已变化，原草稿不能提交，请重新调查。",
                                  ticket=self.tickets.public(record) if record else None,
                                  ticket_events=self.tickets.events(record) if record else [])
        return TicketResponse(status="submitted" if changed else "already_submitted",
                              message="退货工单已提交；未执行真实退款。" if changed else "工单已经提交，重复确认未新增工单。",
                              ticket=self.tickets.public(record), ticket_events=self.tickets.events(record))
