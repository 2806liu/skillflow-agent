"""项目 B 的质量事件演示业务库与确定性判定。

输入是服务端认证的买家 ID 和本人订单；输出为可追溯的三态结论。
SQLAlchemy Session 每次调用单独创建，绝不在并发任务之间共享。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import ForeignKey, Integer, String, Text, create_engine, event, func, or_, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def backup_before_business_migration(database_path: Path) -> Path | None:
    """首次加业务表前备份已有项目 B 数据；SQLite backup 会包含已提交的 WAL 内容。"""
    if not database_path.exists():
        return None
    with closing(sqlite3.connect(database_path, timeout=30)) as source:
        tables = {row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "business_products" in tables:
            return None
        # 全新隔离库只有空框架表，无需备份；旧会话、工单或身份映射必须保留。
        existing = any(source.execute(f"SELECT 1 FROM {name} LIMIT 1").fetchone()
                       for name in ("kv", "tickets", "auth_sessions", "identity_bindings") if name in tables)
        if not existing:
            return None
        destination = database_path.with_name(
            f"{database_path.stem}.pre-business-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}.sqlite3")
        with closing(sqlite3.connect(destination, timeout=30)) as backup:
            source.backup(backup)
        return destination


class Base(DeclarativeBase):
    pass


class Buyer(Base):
    __tablename__ = "business_buyers"
    buyer_id: Mapped[str] = mapped_column(String(80), primary_key=True)


class Product(Base):
    __tablename__ = "business_products"
    sku: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(160))
    brand: Mapped[str] = mapped_column(String(100))
    model: Mapped[str] = mapped_column(String(100))
    category: Mapped[str] = mapped_column(String(40), index=True)
    stock: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(24), default="active")


class Order(Base):
    __tablename__ = "business_orders"
    order_id: Mapped[str] = mapped_column(String(24), primary_key=True)
    buyer_id: Mapped[str] = mapped_column(ForeignKey("business_buyers.buyer_id"), index=True)
    delivered_days_ago: Mapped[int] = mapped_column(Integer)
    quality_issue: Mapped[bool] = mapped_column(default=False)
    paid_cents: Mapped[int] = mapped_column(Integer)
    original_cents: Mapped[int] = mapped_column(Integer)
    discount_cents: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(40))
    item: Mapped[OrderItem] = relationship(back_populates="order", uselist=False)


class OrderItem(Base):
    __tablename__ = "business_order_items"
    order_id: Mapped[str] = mapped_column(ForeignKey("business_orders.order_id"), primary_key=True)
    sku: Mapped[str] = mapped_column(ForeignKey("business_products.sku"), index=True)
    batch_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    batch_source: Mapped[str] = mapped_column(String(80), default="missing")
    batch_verified: Mapped[bool] = mapped_column(default=False)
    order: Mapped[Order] = relationship(back_populates="item")
    product: Mapped[Product] = relationship()


class Incident(Base):
    __tablename__ = "quality_incidents"
    incident_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    created_by: Mapped[str] = mapped_column(String(80))
    current_version: Mapped[int] = mapped_column(Integer, default=1)


class IncidentVersion(Base):
    __tablename__ = "incident_versions"
    incident_id: Mapped[str] = mapped_column(ForeignKey("quality_incidents.incident_id"), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(String(24))
    evidence: Mapped[str] = mapped_column(Text)
    explanation: Mapped[str] = mapped_column(Text)
    special_return_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(80), nullable=True)
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(40))


class AffectedScope(Base):
    __tablename__ = "affected_scopes"
    incident_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    sku: Mapped[str] = mapped_column(ForeignKey("business_products.sku"), primary_key=True)
    batch_start: Mapped[str] = mapped_column(String(80), primary_key=True)
    batch_end: Mapped[str] = mapped_column(String(80))


class IncidentAudit(Base):
    __tablename__ = "incident_audit"
    audit_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str] = mapped_column(String(80), index=True)
    version: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(24))
    actor: Mapped[str] = mapped_column(String(80))
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(40))


class BusinessRepository:
    """在项目 B 自有 SQLite 文件中管理演示业务事实；旧工单表原样保留。"""

    def __init__(self, database_path: Path, fixtures_dir: Path | None = None):
        # SQLite URL 必须使用绝对路径；相对路径在不同启动目录下会指向错误位置。
        self.database_path = database_path.resolve()
        backup_before_business_migration(self.database_path)
        self.engine = create_engine(f"sqlite:///{self.database_path.as_posix()}", connect_args={"timeout": 30})

        @event.listens_for(self.engine, "connect")
        def configure_sqlite(connection, _):
            # 与旧工单仓库使用同一 WAL 数据库；连接只在一次操作内持有。
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        self.price_changes = {}
        self.legacy_policies = {}
        if fixtures_dir:
            price_file = fixtures_dir / "price_adjustments.json"
            self.price_changes = json.loads(price_file.read_text(encoding="utf-8")) if price_file.exists() else {}
            self.legacy_policies = {order_id: item["policy_key"] for order_id, item in
                                    json.loads((fixtures_dir / "orders.json").read_text(encoding="utf-8")).items()}
            self.import_legacy(fixtures_dir)

    def import_legacy(self, fixtures: Path) -> None:
        """仅补入缺失的旧 Mock 行；无批次的旧订单保持待核实，不猜测批次。"""
        buyers = json.loads((fixtures / "buyers.json").read_text(encoding="utf-8"))
        orders = json.loads((fixtures / "orders.json").read_text(encoding="utf-8"))
        stock = json.loads((fixtures / "inventory.json").read_text(encoding="utf-8"))
        with self.sessions.begin() as db:
            for buyer_id in buyers:
                if not db.get(Buyer, buyer_id):
                    db.add(Buyer(buyer_id=buyer_id))
            for row in orders.values():
                sku = row["sku"]
                if not db.get(Product, sku):
                    db.add(Product(sku=sku, name=row["product"], brand="未登记", model="未登记",
                                   category="legacy", stock=int(stock.get(sku, 0)), status="active"))
                if db.get(Order, row["order_id"]):
                    continue
                db.add(Order(order_id=row["order_id"], buyer_id=row["buyer_id"],
                             delivered_days_ago=row["delivered_days_ago"], quality_issue=row["quality_issue"],
                             paid_cents=row["paid_cents"], original_cents=row["original_cents"],
                             discount_cents=row["discount_cents"], status=row["status"],
                             item=OrderItem(sku=sku, batch_code=None, batch_source="missing", batch_verified=False)))

    def has_buyer(self, buyer_id: str) -> bool:
        with self.sessions() as db:
            return db.get(Buyer, buyer_id) is not None

    def list_buyers(self) -> list[dict]:
        with self.sessions() as db:
            return [{"buyer_id": row.buyer_id} for row in db.scalars(select(Buyer).limit(100))]

    def owner_of_order(self, order_id: str) -> str | None:
        with self.sessions() as db:
            row = db.get(Order, order_id)
            return row.buyer_id if row else None

    def list_orders(self, buyer_id: str, limit: int = 100) -> list[dict]:
        with self.sessions() as db:
            ids = db.scalars(select(Order.order_id).where(Order.buyer_id == buyer_id).limit(limit)).all()
        return [self.get_order(order_id, buyer_id) for order_id in ids]

    def query_orders(self, buyer_id: str, keyword: str = "", after_order_id: str | None = None,
                     page_size: int = 20, order_ids: tuple[str, ...] | None = None) -> dict:
        """新图候选发现只执行一次联表查询，服务端按买家过滤，避免逐订单 N+1。"""
        if not 1 <= page_size <= 50:
            raise ValueError("分页大小必须在 1 到 50 之间")
        query = (select(Order.order_id, Product.name.label("product"), Product.sku)
                 .join(OrderItem, OrderItem.order_id == Order.order_id)
                 .join(Product, Product.sku == OrderItem.sku).where(Order.buyer_id == buyer_id))
        if order_ids is not None:
            query = query.where(Order.order_id.in_(order_ids))
        if after_order_id is not None:
            query = query.where(Order.order_id > after_order_id)
        if keyword:
            # 参数绑定并转义 LIKE 通配符，模型不能用 %/_ 绕过商品筛选。
            matches = [func.lower(column).contains(keyword.lower(), autoescape=True)
                       for column in (Product.name, Product.sku, Order.order_id)]
            # 品牌和型号只使用已登记事实；兼容旧 JSON 的“未登记”占位不作为匹配依据。
            matches.extend((column != "未登记") & func.lower(column).contains(keyword.lower(), autoescape=True)
                           for column in (Product.brand, Product.model))
            query = query.where(or_(*matches))
        with self.sessions() as db:
            rows = db.execute(query.order_by(Order.order_id).limit(page_size + 1)).mappings().all()
        more = len(rows) > page_size
        items = [dict(row) for row in rows[:page_size]]
        return {"items": items, "has_more": more,
                "next_after_order_id": items[-1]["order_id"] if more else None}

    def get_order(self, order_id: str, buyer_id: str) -> dict | None:
        """买家 ID 必须来自服务端；SQL 查询同时限定归属，不返回他人详情。"""
        with self.sessions() as db:
            row = db.scalar(select(Order).where(Order.order_id == order_id, Order.buyer_id == buyer_id))
            if not row:
                return None
            item, product = row.item, row.item.product
            return {"order_id": row.order_id, "product": product.name, "sku": item.sku,
                    "policy_key": ("quality_incident" if product.category == "power_bank" else
                                   self.legacy_policies.get(order_id, "unknown")),
                    "delivered_days_ago": row.delivered_days_ago, "quality_issue": row.quality_issue,
                    "paid_cents": row.paid_cents, "original_cents": row.original_cents,
                    "discount_cents": row.discount_cents, "status": row.status,
                    "category": product.category, "brand": product.brand, "model": product.model,
                    "batch_code": item.batch_code, "batch_source": item.batch_source,
                    "batch_verified": item.batch_verified}

    def inventory(self, sku: str) -> int:
        with self.sessions() as db:
            product = db.get(Product, sku)
            return product.stock if product else 0

    def price_adjustment(self, order_id: str, buyer_id: str) -> dict | None:
        return self.price_changes.get(order_id) if self.get_order(order_id, buyer_id) else None

    def create_incident(self, *, incident_id: str, actor: str, evidence: str, explanation: str,
                        sku: str, batch_start: str, batch_end: str, special_return_days: int | None = None) -> None:
        """录入仅生成 draft；证据和范围必填，运营线索不会直接参与资格判断。"""
        if not all((incident_id, actor, evidence.strip(), explanation.strip(), sku, batch_start, batch_end)):
            raise ValueError("事件、操作者、证据、说明、SKU 和批次范围不能为空")
        if batch_start > batch_end or (special_return_days is not None and special_return_days < 0):
            raise ValueError("批次范围或特殊退货期限无效")
        with self.sessions.begin() as db:
            if db.get(Incident, incident_id) or not db.get(Product, sku):
                raise ValueError("事件已存在或商品 SKU 不存在")
            db.add(Incident(incident_id=incident_id, created_by=actor, current_version=1))
            db.add(IncidentVersion(incident_id=incident_id, version=1, status="draft", evidence=evidence,
                                   explanation=explanation, special_return_days=special_return_days,
                                   reviewed_by=None, reason="录入待审核", created_at=now()))
            db.add(AffectedScope(incident_id=incident_id, version=1, sku=sku,
                                 batch_start=batch_start, batch_end=batch_end))
            db.add(IncidentAudit(incident_id=incident_id, version=1, action="create", actor=actor,
                                 reason="录入待审核", created_at=now()))

    def change_incident(self, incident_id: str, *, actor: str, action: str, reason: str) -> int:
        """发布或撤回都追加不可变版本和审计；审核人不能与录入人相同。"""
        if action not in {"publish", "withdraw"} or not actor or not reason.strip():
            raise ValueError("审核动作、操作者或原因无效")
        with self.sessions.begin() as db:
            incident = db.get(Incident, incident_id)
            if not incident:
                raise ValueError("事件不存在")
            previous = db.get(IncidentVersion, (incident_id, incident.current_version))
            author = db.scalar(select(IncidentAudit.actor).where(
                IncidentAudit.incident_id == incident_id,
                IncidentAudit.version == incident.current_version,
                IncidentAudit.action.in_(["create", "revise"])))
            if action == "publish" and (previous.status != "draft" or actor == author):
                raise ValueError("仅其他审核人可发布待审核版本")
            if action == "withdraw" and previous.status != "active":
                raise ValueError("仅已生效事件可撤回")
            scopes = db.scalars(select(AffectedScope).where(AffectedScope.incident_id == incident_id,
                                                              AffectedScope.version == incident.current_version)).all()
            version = incident.current_version + 1
            status = "active" if action == "publish" else "withdrawn"
            db.add(IncidentVersion(incident_id=incident_id, version=version, status=status,
                                   evidence=previous.evidence, explanation=previous.explanation,
                                   special_return_days=previous.special_return_days,
                                   reviewed_by=actor, reason=reason, created_at=now()))
            for scope in scopes:
                db.add(AffectedScope(incident_id=incident_id, version=version, sku=scope.sku,
                                     batch_start=scope.batch_start, batch_end=scope.batch_end))
            incident.current_version = version
            db.add(IncidentAudit(incident_id=incident_id, version=version, action=action,
                                 actor=actor, reason=reason, created_at=now()))
            return version

    def revise_incident(self, incident_id: str, *, actor: str, reason: str, evidence: str,
                        explanation: str, sku: str, batch_start: str, batch_end: str,
                        special_return_days: int | None = None) -> int:
        """范围、证据或例外规则变更生成新待审核版本，旧版本保留以解释历史工单。"""
        if not all((actor, reason.strip(), evidence.strip(), explanation.strip(), sku, batch_start, batch_end)):
            raise ValueError("修订字段不能为空")
        if batch_start > batch_end or (special_return_days is not None and special_return_days < 0):
            raise ValueError("批次范围或特殊期限无效")
        with self.sessions.begin() as db:
            incident = db.get(Incident, incident_id)
            if not incident or not db.get(Product, sku):
                raise ValueError("事件或 SKU 不存在")
            version = incident.current_version + 1
            db.add(IncidentVersion(incident_id=incident_id, version=version, status="draft",
                                   evidence=evidence, explanation=explanation,
                                   special_return_days=special_return_days, reviewed_by=None,
                                   reason=reason, created_at=now()))
            db.add(AffectedScope(incident_id=incident_id, version=version, sku=sku,
                                 batch_start=batch_start, batch_end=batch_end))
            incident.current_version = version
            db.add(IncidentAudit(incident_id=incident_id, version=version, action="revise",
                                 actor=actor, reason=reason, created_at=now()))
            return version

    def evaluate(self, order_id: str, buyer_id: str) -> dict | None:
        """对本人可信订单批次与当前已审核事件版本做确定性比较。"""
        order = self.get_order(order_id, buyer_id)
        if not order:
            return None
        result = {"decision": "UNVERIFIED", "reason_code": "NO_ACTIVE_INCIDENT",
                  "order_id": order_id, "event_id": None, "event_version": None,
                  "return_eligible": False, "manual_review": True}
        if order["category"] != "power_bank":
            result["reason_code"] = "NOT_INCIDENT_PRODUCT"
            return result
        # 用户自报批次即使被误标为 verified，也不能充当商家订单的可信批次凭证。
        if (not order["batch_code"] or not order["batch_verified"] or
                order["batch_source"] not in {"merchant_order", "warehouse_scan", "manufacturer_record"}):
            result["reason_code"] = "BATCH_UNVERIFIED"
            return result
        with self.sessions() as db:
            rows = db.execute(select(Incident, IncidentVersion, AffectedScope)
                              .join(IncidentVersion, (Incident.incident_id == IncidentVersion.incident_id)
                                    & (Incident.current_version == IncidentVersion.version))
                              .join(AffectedScope, (AffectedScope.incident_id == IncidentVersion.incident_id)
                                    & (AffectedScope.version == IncidentVersion.version))
                              .where(IncidentVersion.status == "active", AffectedScope.sku == order["sku"])).all()
            if not rows:
                return result
            matching = [(incident, version) for incident, version, scope in rows
                        if scope.batch_start <= order["batch_code"] <= scope.batch_end]
            incident, version = matching[0] if matching else rows[0][:2]
            matched = bool(matching)
            result.update(event_id=incident.incident_id, event_version=version.version,
                          decision="AFFECTED" if matched else "NOT_AFFECTED",
                          reason_code="BATCH_IN_SCOPE" if matched else "BATCH_OUT_OF_SCOPE",
                          manual_review=False)
            if matched:
                days = order["delivered_days_ago"]
                if days <= 7:
                    result["return_eligible"] = True
                elif version.special_return_days is not None and days <= version.special_return_days:
                    result["return_eligible"] = True
                    result["reason_code"] = "APPROVED_SPECIAL_RETURN"
                else:
                    result["manual_review"] = True
                    result["reason_code"] = "OUTSIDE_WINDOW_MANUAL_REVIEW"
            return result

    def alternatives(self, sku: str | None = None) -> list[dict]:
        """只列 B 项目目录中有库存的候选品；不推断候选品绝对安全。"""
        with self.sessions() as db:
            # 已生效事件覆盖的 SKU 整体排除；库存无逐件可信批次时不能推荐同款。
            affected_skus = db.scalars(select(AffectedScope.sku)
                                       .join(Incident, Incident.incident_id == AffectedScope.incident_id)
                                       .join(IncidentVersion,
                                             (IncidentVersion.incident_id == Incident.incident_id) &
                                             (IncidentVersion.version == Incident.current_version) &
                                             (AffectedScope.version == IncidentVersion.version))
                                       .where(IncidentVersion.status == "active")).all()
            query = select(Product).where(Product.category == "power_bank", Product.stock > 0,
                                          Product.status == "active", Product.sku.not_in(affected_skus))
            if sku:
                query = query.where(Product.sku != sku)
            rows = db.scalars(query.limit(10)).all()
            return [{"sku": row.sku, "name": row.name, "brand": row.brand,
                     "model": row.model, "available": row.stock,
                     "basis": "同类且目录显示有库存；未作全面安全保证"} for row in rows]

    def active_event_summary(self) -> list[dict]:
        """通用事件说明只读取已审核生效的版本，绝不把草稿新闻当政策。"""
        with self.sessions() as db:
            rows = db.execute(select(Incident, IncidentVersion)
                              .join(IncidentVersion, (Incident.incident_id == IncidentVersion.incident_id)
                                    & (Incident.current_version == IncidentVersion.version))
                              .where(IncidentVersion.status == "active")).all()
            return [{"event_id": incident.incident_id, "version": version.version,
                     "explanation": version.explanation, "evidence": version.evidence}
                    for incident, version in rows]
