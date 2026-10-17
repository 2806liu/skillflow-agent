"""项目 B 本机质量事件受控命令。

seed 只允许写入项目 B 的隔离演示库；事件录入、发布和撤回要求显式操作者与原因。
本命令不读取新闻网站，不访问项目 A，也不执行真实退款。
"""

import argparse
from pathlib import Path
import sys

from sqlalchemy import select

from skillflow.business.surge import (AffectedScope, BusinessRepository, Buyer, Incident, IncidentAudit,
                                      IncidentVersion, Order, OrderItem, Product)
from skillflow.config import ROOT
from skillflow.sqlite_store import SqliteStore


def inside_project(path: Path) -> Path:
    """拒绝跨项目路径，避免将模拟订单导入项目 A 或任意真实业务库。"""
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError("数据目录必须位于项目 B 内")
    return resolved


def seed(repository: BusinessRepository, count: int, buyers: int) -> None:
    """固定规则生成多个批次及异常样本，结果可按订单号复现。"""
    if count < 10000 or buyers < 1000:
        raise ValueError("验收样本至少需要 10000 笔订单和 1000 个买家")
    products = (
        Product(sku="PB-SIM-A", name="模拟充电宝 A", brand="演示品牌", model="PB-A",
                category="power_bank", stock=0, status="active"),
        Product(sku="PB-SIM-B", name="模拟充电宝 B", brand="候选品牌", model="PB-B",
                category="power_bank", stock=50, status="active"),
        Product(sku="PB-SIM-C", name="模拟充电宝 C", brand="候选品牌", model="PB-C",
                category="power_bank", stock=0, status="active"),
    )
    with repository.sessions.begin() as db:
        if db.get(Order, "A100000"):
            raise ValueError("目标库已有模拟订单；请使用新的隔离数据目录")
        for item in products:
            if not db.get(Product, item.sku):
                db.add(item)
        db.flush()
        db.add_all(Buyer(buyer_id=f"surge-buyer-{i:04d}") for i in range(buyers))
        db.flush()
        for i in range(count):
            # 0: 受影响；1: 同 SKU 其他批次；2: 缺批次；3: 未核实批次；4: 超期。
            case = i % 5
            batch = None if case == 2 else ("B20260025" if case in (0, 3, 4) else "B20260090")
            order = Order(order_id=f"A{100000+i}", buyer_id=f"surge-buyer-{i % buyers:04d}",
                          delivered_days_ago=30 if case == 4 else 2, quality_issue=case in (0, 4),
                          paid_cents=9900, original_cents=10900, discount_cents=1000, status="已签收",
                          item=OrderItem(sku="PB-SIM-A", batch_code=batch,
                                         batch_source="merchant_order" if case != 3 else "user_claim",
                                         batch_verified=case not in (2, 3)))
            db.add(order)
            if i and i % 1000 == 0:
                db.flush()
    repository.create_incident(incident_id="SIM-POWERBANK-001", actor="seed-operator",
                               evidence="项目 B 固定模拟质量事件证据；仅供本机验收",
                               explanation="模拟品牌充电宝指定批次需核查售后处理",
                               sku="PB-SIM-A", batch_start="B20260020", batch_end="B20260030",
                               special_return_days=None)
    print(f"已生成 {count} 笔模拟订单、{buyers} 个买家；事件为 draft，尚未生效。")


def main() -> None:
    parser = argparse.ArgumentParser(description="项目 B 质量事件演示管理")
    parser.add_argument("--data-dir", type=Path, required=True, help="项目 B 内的 SQLite 数据目录")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("seed", help="生成隔离模拟负载数据")
    generate.add_argument("--orders", type=int, default=10000)
    generate.add_argument("--buyers", type=int, default=1000)
    create = commands.add_parser("create", help="录入待审核事件")
    for name in ("incident-id", "actor", "evidence", "explanation", "sku", "batch-start", "batch-end"):
        create.add_argument("--" + name, required=True)
    create.add_argument("--special-return-days", type=int)
    revise = commands.add_parser("revise", help="修订范围或规则并形成待审核的新版本")
    for name in ("incident-id", "actor", "reason", "evidence", "explanation",
                 "sku", "batch-start", "batch-end"):
        revise.add_argument("--" + name, required=True)
    revise.add_argument("--special-return-days", type=int)
    for action in ("publish", "withdraw"):
        change = commands.add_parser(action, help="审核生效或撤回事件")
        for name in ("incident-id", "actor", "reason"):
            change.add_argument("--" + name, required=True)
    show = commands.add_parser("show", help="查看事件版本和审计")
    show.add_argument("--incident-id", required=True)
    args = parser.parse_args()
    data_dir = inside_project(args.data_dir)
    if args.command == "seed" and not data_dir.is_relative_to((ROOT / ".skillflow" / "surge-tests").resolve()):
        parser.error("seed 只允许写入项目 B 的 .skillflow/surge-tests 下的隔离目录")
    # 先初始化与旧工单共用的项目 B SQLite，再为此文件创建 SQLAlchemy 业务表。
    store = SqliteStore(data_dir)
    repository = BusinessRepository(store.database_path)
    if args.command == "seed":
        seed(repository, args.orders, args.buyers)
    elif args.command == "create":
        repository.create_incident(incident_id=args.incident_id, actor=args.actor,
                                   evidence=args.evidence, explanation=args.explanation,
                                   sku=args.sku, batch_start=args.batch_start, batch_end=args.batch_end,
                                   special_return_days=args.special_return_days)
        print("事件已录入为 draft；需由其他审核人执行 publish。")
    elif args.command == "revise":
        version = repository.revise_incident(
            args.incident_id, actor=args.actor, reason=args.reason, evidence=args.evidence,
            explanation=args.explanation, sku=args.sku, batch_start=args.batch_start,
            batch_end=args.batch_end, special_return_days=args.special_return_days)
        print(f"事件修订为 v{version} draft；需由其他审核人执行 publish。")
    elif args.command in {"publish", "withdraw"}:
        version = repository.change_incident(args.incident_id, actor=args.actor,
                                             action=args.command, reason=args.reason)
        print(f"事件版本 {version} 已记录，动作：{args.command}。")
    else:
        with repository.sessions() as db:
            incident = db.get(Incident, args.incident_id)
            if not incident:
                parser.error("事件不存在")
            for version in db.scalars(select(IncidentVersion).where(
                    IncidentVersion.incident_id == args.incident_id).order_by(IncidentVersion.version)):
                print(f"v{version.version} {version.status} 审核人={version.reviewed_by or '-'} 原因={version.reason}")
            for audit in db.scalars(select(IncidentAudit).where(
                    IncidentAudit.incident_id == args.incident_id).order_by(IncidentAudit.audit_id)):
                print(f"审计 {audit.created_at} {audit.actor} {audit.action} {audit.reason}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc
