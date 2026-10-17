"""S2 固定质量事件样本，仅用于隔离验收库，不生成或迁移用户业务数据。"""

from skillflow.business.surge import Order, OrderItem, Product


def seed_quality_demo(repository):
    """四笔订单分别覆盖受影响、范围外、缺批次和超期，另有一个库存候选。"""
    with repository.sessions.begin() as db:
        db.add_all([
            Product(sku="S2-PB-A", name="S2 模拟充电宝", brand="模拟品牌", model="S2-A",
                    category="power_bank", stock=0, status="active"),
            Product(sku="S2-PB-B", name="S2 候选充电宝", brand="候选品牌", model="S2-B",
                    category="power_bank", stock=8, status="active"),
        ])
        db.flush()
        for order_id, batch, days in (("A700", "B025", 2), ("A701", "B090", 2),
                                      ("A702", None, 2), ("A704", "B025", 30)):
            db.add(Order(order_id=order_id, buyer_id="demo-buyer", delivered_days_ago=days,
                         quality_issue=True, paid_cents=9900, original_cents=10900,
                         discount_cents=1000, status="已签收", item=OrderItem(
                             sku="S2-PB-A", batch_code=batch, batch_source="merchant_order",
                             batch_verified=batch is not None)))
    repository.create_incident(incident_id="S2-INCIDENT", actor="s2-author", evidence="固定模拟事件证据",
                               explanation="S2 指定批次需核查", sku="S2-PB-A", batch_start="B020", batch_end="B030")
    repository.change_incident("S2-INCIDENT", actor="s2-reviewer", action="publish", reason="模拟审核通过")
