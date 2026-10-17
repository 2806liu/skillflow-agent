"""项目 B 的 SQLite 工单仓库；状态与历史在一个事务提交。

输入为服务端买家、订单、工单类型与确认动作；输出当前工单及审计事件。
数据库写入失败会回滚并报错，不会退回进程内存或执行真实退款。
"""

import json
from contextlib import closing, nullcontext
from datetime import datetime, timezone
from uuid import uuid4

from skillflow.contracts import TicketEvent, TicketRecord
from skillflow.sqlite_store import SqliteStore


class SqliteTicketStore:
    """对外保持原工单接口，内部按买家隔离并使用数据库事务。"""

    def __init__(self, store: SqliteStore):
        self.store = store

    @staticmethod
    def _load(conn, ticket_id):
        """读取工单和有序历史；不存在返回 None。"""
        row = conn.execute("SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
        if not row:
            return None
        record = dict(row)
        record["metadata"] = json.loads(record.pop("metadata_json"))
        record["ticket_events"] = [dict(item) for item in conn.execute(
            "SELECT * FROM ticket_events WHERE ticket_id=? ORDER BY created_at, rowid", (ticket_id,))]
        return record

    @staticmethod
    def _event(conn, record, session_id, action, result, error_code=None):
        """在调用方事务内写历史，保持状态与审计同步提交。"""
        value = TicketEvent(event_id=str(uuid4()), ticket_id=record["ticket_id"],
                            buyer_id=record["buyer_id"], session_id=session_id,
                            action=action, result=result,
                            created_at=datetime.now(timezone.utc).isoformat(),
                            error_code=error_code).model_dump()
        conn.execute("INSERT INTO ticket_events VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                     tuple(value[key] for key in ("event_id", "ticket_id", "buyer_id", "session_id",
                                                 "action", "result", "created_at", "error_code")))

    @staticmethod
    def events(record) -> list[TicketEvent]:
        """仅返回已经通过买家校验的工单历史。"""
        return [TicketEvent.model_validate(item) for item in record.get("ticket_events", [])]

    @staticmethod
    def public(record) -> TicketRecord:
        """去除内部身份与历史字段，避免在普通工单响应中泄漏买家。"""
        return TicketRecord.model_validate({key: value for key, value in record.items()
                                            if key not in {"buyer_id", "session_id", "ticket_events"}})

    def get(self, ticket_id):
        """读取最新状态；调用方仍须比对服务端 buyer_id。"""
        if not ticket_id:
            return None
        with closing(self.store.connect()) as conn:
            return self._load(conn, ticket_id)

    def for_order(self, *, buyer_id, order_id, request_type=None):
        """按买家和订单查询候选工单，歧义时由对话层追问。"""
        sql = "SELECT ticket_id FROM tickets WHERE buyer_id=? AND order_id=?"
        args = [buyer_id, order_id]
        if request_type:
            sql += " AND request_type=?"
            args.append(request_type)
        with closing(self.store.connect()) as conn:
            return [self._load(conn, row["ticket_id"]) for row in conn.execute(sql + " ORDER BY created_at", args)]

    def find_by_idempotency(self, idempotency_key):
        """只查草稿和已提交记录；取消后允许重新申请。"""
        with closing(self.store.connect()) as conn:
            row = conn.execute("SELECT ticket_id FROM tickets WHERE idempotency_key=? "
                               "AND status IN ('draft','submitted')", (idempotency_key,)).fetchone()
            return self._load(conn, row["ticket_id"]) if row else None

    def find_active_for_order(self, *, buyer_id, order_id, request_type=None, exclude_ticket_id=None):
        """查询当前活跃工单，供退换与价保互斥、重复提交检查。"""
        sql = "SELECT ticket_id FROM tickets WHERE buyer_id=? AND order_id=? AND status IN ('draft','submitted')"
        args = [buyer_id, order_id]
        if request_type:
            sql += " AND request_type=?"
            args.append(request_type)
        if exclude_ticket_id:
            sql += " AND ticket_id<>?"
            args.append(exclude_ticket_id)
        with closing(self.store.connect()) as conn:
            row = conn.execute(sql + " LIMIT 1", args).fetchone()
            return self._load(conn, row["ticket_id"]) if row else None

    def find_conflict_for_order(self, *, buyer_id, order_id, request_type):
        """只返回同买家同订单的异类有效工单；同类型保留原有幂等复用路径。"""
        if request_type not in {"return", "exchange", "price_adjustment"}:
            return None
        active = self.find_active_for_order(buyer_id=buyer_id, order_id=order_id)
        return active if active and active["request_type"] in {
            "return", "exchange", "price_adjustment"} and active["request_type"] != request_type else None

    def append_event(self, ticket_id, *, session_id, action, result, error_code=None):
        """对恢复、拒绝和重复确认留痕；不存在则不写孤立事件。"""
        with self.store.transaction() as conn:
            record = self._load(conn, ticket_id)
            if record:
                self._event(conn, record, session_id, action, result, error_code)
                return self._load(conn, ticket_id)
            return None

    @staticmethod
    def _incident_eligible(conn, *, buyer_id, order_id, sku, metadata) -> bool:
        """在已取得 SQLite 写锁的事务中核对当前事件版本、可信批次和期限。"""
        if not metadata.get("incident_id") or not metadata.get("incident_version"):
            return False
        row = conn.execute("""
            SELECT o.delivered_days_ago, i.sku, i.batch_code, i.batch_source, i.batch_verified,
                   v.status, v.special_return_days, q.current_version
            FROM business_orders o
            JOIN business_order_items i ON i.order_id=o.order_id
            JOIN quality_incidents q ON q.incident_id=?
            JOIN incident_versions v ON v.incident_id=q.incident_id AND v.version=q.current_version
            WHERE o.order_id=? AND o.buyer_id=?
        """, (metadata["incident_id"], order_id, buyer_id)).fetchone()
        if not (row and row["status"] == "active" and
                row["current_version"] == metadata["incident_version"] and
                row["sku"] == sku and row["batch_verified"] and
                row["batch_source"] in {"merchant_order", "warehouse_scan", "manufacturer_record"} and
                row["batch_code"] == metadata.get("batch_code") and
                (row["delivered_days_ago"] <= 7 or
                 row["special_return_days"] is not None and
                 row["delivered_days_ago"] <= row["special_return_days"])):
            return False
        return conn.execute("""
            SELECT 1 FROM affected_scopes WHERE incident_id=? AND version=? AND sku=?
            AND batch_start<=? AND batch_end>=? LIMIT 1
        """, (metadata["incident_id"], metadata["incident_version"], sku,
              row["batch_code"], row["batch_code"])).fetchone() is not None

    def create_draft(self, *, buyer_id, session_id, order_id, sku, request_type, reason, source_run_id, metadata=None, connection=None):
        """共享工单命令可注入已持锁事务，使草稿和操作回执同事务；旧入口仍自行建事务。"""
        key = f"{buyer_id}:{order_id}:{request_type}"
        with (nullcontext(connection) if connection is not None else self.store.transaction()) as conn:
            # 事件可能在服务层初次判断后撤回；复用旧草稿前也核对最新版本。
            if metadata and metadata.get("incident_id") and not self._incident_eligible(
                    conn, buyer_id=buyer_id, order_id=order_id, sku=sku, metadata=metadata):
                return None, False, None
            row = conn.execute("SELECT ticket_id FROM tickets WHERE idempotency_key=? "
                               "AND status IN ('draft','submitted')", (key,)).fetchone()
            if row:
                record = self._load(conn, row["ticket_id"])
                self._event(conn, record, session_id, "draft_reused", record["status"])
                return self._load(conn, row["ticket_id"]), False, None
            if request_type in {"return", "exchange", "price_adjustment"}:
                row = conn.execute("SELECT ticket_id FROM tickets WHERE buyer_id=? AND order_id=? "
                                   "AND request_type IN ('return','exchange','price_adjustment') "
                                   "AND status IN ('draft','submitted')",
                                   (buyer_id, order_id)).fetchone()
                if row:
                    conflict = self._load(conn, row["ticket_id"])
                    # 数据库唯一索引与事务内检查共同阻止退换和价保补偿并行建单。
                    self._event(conn, conflict, session_id, "draft_conflict", "rejected", "TICKET_CONFLICT")
                    return None, False, self._load(conn, row["ticket_id"])
            if metadata and metadata.get("incident_id") and not self._incident_eligible(
                    conn, buyer_id=buyer_id, order_id=order_id, sku=sku, metadata=metadata):
                # 事件可能在调查与点击建单之间撤回；不产生不可信草稿。
                return None, False, None
            now, ticket_id = datetime.now(timezone.utc).isoformat(), "TK-" + uuid4().hex[:10].upper()
            conn.execute("INSERT INTO tickets VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (ticket_id, buyer_id, session_id, order_id, sku, request_type, reason, "draft",
                          key, source_run_id, json.dumps(metadata or {}, ensure_ascii=False), now, now))
            record = self._load(conn, ticket_id)
            self._event(conn, record, session_id, "draft_created", "draft")
            return self._load(conn, ticket_id), True, None

    def confirm(self, ticket_id, *, session_id, buyer_id, connection=None):
        """共享命令可注入事务，确认消费、提交及回执原子；旧接口仍核归属和状态。"""
        with (nullcontext(connection) if connection is not None else self.store.transaction()) as conn:
            record = self._load(conn, ticket_id)
            if not record or record["buyer_id"] != buyer_id:
                return None, False
            if record["status"] == "submitted":
                self._event(conn, record, session_id, "confirm_repeated", "already_submitted")
                return self._load(conn, ticket_id), False
            if record["status"] != "draft":
                return record, False
            conn.execute("UPDATE tickets SET status='submitted', updated_at=? WHERE ticket_id=?",
                         (datetime.now(timezone.utc).isoformat(), ticket_id))
            self._event(conn, record, session_id, "confirm", "submitted")
            return self._load(conn, ticket_id), True

    def confirm_incident(self, ticket_id, *, session_id, buyer_id):
        """在原工单短事务内复核事件、订单批次及归属，随后原子提交状态和历史。

        事件撤回、版本变化或批次凭证变化时保留草稿并写失败审计；重试不会多次正式提交。
        """
        with self.store.transaction() as conn:
            record = self._load(conn, ticket_id)
            if not record or record["buyer_id"] != buyer_id:
                return None, False, "TICKET_FORBIDDEN"
            if record["status"] == "submitted":
                self._event(conn, record, session_id, "confirm_repeated", "already_submitted")
                return self._load(conn, ticket_id), False, None
            if record["status"] != "draft" or record["request_type"] != "return":
                return record, False, "TICKET_NOT_CONFIRMABLE"
            meta = record["metadata"]
            if not meta.get("incident_id") or not meta.get("incident_version"):
                return record, False, "INCIDENT_EVIDENCE_MISSING"
            valid = self._incident_eligible(conn, buyer_id=buyer_id, order_id=record["order_id"],
                                            sku=record["sku"], metadata=meta)
            if not valid:
                self._event(conn, record, session_id, "confirm_revalidation_failed", "rejected",
                            "INCIDENT_CHANGED_OR_UNVERIFIED")
                return self._load(conn, ticket_id), False, "INCIDENT_CHANGED_OR_UNVERIFIED"
            conn.execute("UPDATE tickets SET status='submitted', updated_at=? WHERE ticket_id=?",
                         (datetime.now(timezone.utc).isoformat(), ticket_id))
            self._event(conn, record, session_id, "confirm", "submitted")
            return self._load(conn, ticket_id), True, None

    def cancel(self, ticket_id, *, session_id, buyer_id):
        """取消和历史同事务；重复取消只写一条重复操作记录。"""
        with self.store.transaction() as conn:
            record = self._load(conn, ticket_id)
            if not record or record["buyer_id"] != buyer_id:
                return None, False
            if record["status"] == "cancelled":
                self._event(conn, record, session_id, "cancel_repeated", "already_cancelled")
                return self._load(conn, ticket_id), False
            if record["status"] not in {"draft", "submitted"}:
                return record, False
            conn.execute("UPDATE tickets SET status='cancelled', updated_at=? WHERE ticket_id=?",
                         (datetime.now(timezone.utc).isoformat(), ticket_id))
            self._event(conn, record, session_id, "cancel", "cancelled")
            return self._load(conn, ticket_id), True
