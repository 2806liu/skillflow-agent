"""项目 B 的退换货工单存储；当前使用 JSON，后续可替换为 SQLite Repository。"""

from datetime import datetime, timezone
from uuid import uuid4

from skillflow.contracts import TicketEvent, TicketRecord
from skillflow.storage import JsonStore


class TicketStore:
    """保存草稿和已提交工单，并用幂等键阻止重复创建。"""

    def __init__(self, store: JsonStore):
        self.store = store

    def _records(self):
        """读取项目 B 自己的工单集合；没有文件时从空集合开始。"""
        return self.store.read("tickets", {})

    def _save(self, records):
        """通过 JsonStore 原子替换保存，避免写半条工单记录。"""
        self.store.write("tickets", records)

    @staticmethod
    def _event(record, *, session_id, action, result, error_code=None):
        """为一次操作生成事件；buyer_id 从原工单取得，不信任会话传入的身份。"""
        return TicketEvent(event_id=str(uuid4()), ticket_id=record["ticket_id"],
                           buyer_id=record["buyer_id"], session_id=session_id,
                           action=action, result=result,
                           created_at=datetime.now(timezone.utc).isoformat(),
                           error_code=error_code).model_dump()

    def append_event(self, ticket_id, *, session_id, action, result, error_code=None):
        """记录重复、拒绝或放弃操作；调用方在 API 工单锁内执行。"""
        #记录“重复确认、拒绝、放弃”这类不改变工单状态但需要留痕的操作。比如用户第二次点确认，状态不变，但审计里要记一条 confirm_repeated。
        records = self._records()
        record = records.get(ticket_id)
        if not record:
            return None
        record.setdefault("ticket_events", []).append(self._event(
            record, session_id=session_id, action=action, result=result, error_code=error_code))
        self._save(records)
        return record

    @staticmethod
    def events(record) -> list[TicketEvent]:
        """旧工单缺少事件字段时返回空历史，不伪造过去的操作。"""
        return [TicketEvent.model_validate(event) for event in record.get("ticket_events", [])]

    def find_by_idempotency(self, idempotency_key: str):
        """按服务端构造的幂等键查找已有工单，不信任客户端传入 buyer_id。"""
        #幂等键是服务端构造,规则在 create_draft 里
        for record in self._records().values():
            if record["idempotency_key"] == idempotency_key and record["status"] != "cancelled":
                return record
        return None

    def find_active_for_order(self, *, buyer_id, order_id, request_type=None, exclude_ticket_id=None):
        """查找同一用户订单上的活跃工单，供跨类型冲突检查使用。"""
        for record in self._records().values():
            if record.get("buyer_id") != buyer_id or record.get("order_id") != order_id:
                continue
            if record.get("status") not in {"draft", "submitted"}:
                continue
            if request_type is not None and record.get("request_type") != request_type:
                continue
            if exclude_ticket_id and record.get("ticket_id") == exclude_ticket_id:
                continue
            return record
        return None

    def get(self, ticket_id: str):
        """按工单号读取内部记录；调用方还需校验会话归属。"""
        return self._records().get(ticket_id)

    def for_order(self, *, buyer_id, order_id, request_type=None):
        """按买家、订单和可选类型返回记录；对话遇到多个候选时必须追问。"""
        return [record for record in self._records().values()
                if record.get("buyer_id") == buyer_id and record.get("order_id") == order_id
                and (request_type is None or record.get("request_type") == request_type)]

    def create_draft(self, *, buyer_id, session_id, order_id, sku, request_type, reason, source_run_id, metadata=None):
        """创建或复用草稿；buyer_id 决定归属，session_id 只记录来源会话。"""
        idempotency_key = f"{buyer_id}:{order_id}:{request_type}"
        existing = self.find_by_idempotency(idempotency_key)
        if existing:
            # 同一买家跨会话找回草稿或已提交工单时，也记录本次恢复动作。
            existing = self.append_event(existing["ticket_id"], session_id=session_id,
                                         action="draft_reused", result=existing["status"])
            # 同一个买家开启新聊天后仍可恢复自己的工单；生产环境由认证 buyer_id 保证安全。
            return existing, False, None
        # 退货与换货互斥；优惠争议独立记录金额问题，不阻塞售后路径。
        conflict = None
        if request_type in {"return", "exchange"}:
            opposite = "exchange" if request_type == "return" else "return"
            conflict = self.find_active_for_order(
                buyer_id=buyer_id, order_id=order_id, request_type=opposite)
        if conflict:
            # 冲突尝试归档到已有活跃工单，不创建无归属的新工单。
            self.append_event(conflict["ticket_id"], session_id=session_id,
                              action="draft_conflict", result="rejected", error_code="TICKET_CONFLICT")
            # 同一订单的退货和换货是互斥售后路径，必须先关闭或取消已有工单。
            return None, False, conflict
        now = datetime.now(timezone.utc).isoformat()
        ticket_id = "TK-" + uuid4().hex[:10].upper()
        record = {"ticket_id": ticket_id, "buyer_id": buyer_id, "session_id": session_id,
                  "order_id": order_id, "sku": sku, "request_type": request_type,
                  "reason": reason, "status": "draft", "idempotency_key": idempotency_key,
                  "source_run_id": source_run_id, "metadata": metadata or {},
                  "created_at": now, "updated_at": now}
        # 草稿和首条审计事件一次写入，避免状态存在而历史缺失。
        record["ticket_events"] = [self._event(record, session_id=session_id,
                                                action="draft_created", result="draft")]
        records = self._records()
        records[ticket_id] = record
        self._save(records)
        return record, True, None

    def confirm(self, ticket_id: str, *, session_id: str):
        """将草稿幂等地提交；已提交记录再次确认时保持原工单号。"""
        records = self._records()
        record = records.get(ticket_id)
        if not record:
            return None, False
        if record["status"] == "submitted":
            record.setdefault("ticket_events", []).append(self._event(
                record, session_id=session_id, action="confirm_repeated", result="already_submitted"))
            self._save(records)
            return record, False
        if record["status"] != "draft":
            return record, False
        record["status"] = "submitted"
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        record.setdefault("ticket_events", []).append(self._event(
            record, session_id=session_id, action="confirm", result="submitted"))
        records[ticket_id] = record
        self._save(records)
        return record, True

    def cancel(self, ticket_id: str, *, session_id: str):
        """幂等取消草稿或已提交工单，使用户可以切换另一种售后路径。"""
        records = self._records()
        record = records.get(ticket_id)
        if not record:
            return None, False
        if record["status"] == "cancelled":
            record.setdefault("ticket_events", []).append(self._event(
                record, session_id=session_id, action="cancel_repeated", result="already_cancelled"))
            self._save(records)
            return record, False
        if record["status"] not in {"draft", "submitted"}:
            return record, False
        record["status"] = "cancelled"
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        record.setdefault("ticket_events", []).append(self._event(
            record, session_id=session_id, action="cancel", result="cancelled"))
        self._save(records)
        return record, True

    @staticmethod
    def public(record) -> TicketRecord:
        """去掉内部身份和审计存储字段；审计历史通过授权接口单独返回。"""
        return TicketRecord.model_validate({key: value for key, value in record.items()
                                            if key not in {"buyer_id", "session_id", "ticket_events"}})
