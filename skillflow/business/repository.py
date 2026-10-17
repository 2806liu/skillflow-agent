import json
from pathlib import Path
from typing import Protocol


class OrderRepository(Protocol):
    # 新图的候选发现统一走有界分页；身份由执行器绑定，不接收模型身份字段。
    def owner_of_order(self, order_id: str) -> str | None: ...
    def query_orders(self, buyer_id: str, keyword: str = "", after_order_id: str | None = None,
                     page_size: int = 20, order_ids: tuple[str, ...] | None = None) -> dict: ...
    def get_order(self, order_id: str, buyer_id: str) -> dict | None: ...
    def inventory(self, sku: str) -> int: ...
    def price_adjustment(self, order_id: str, buyer_id: str) -> dict | None: ...


class MockOrderRepository:
    """JSON fixtures; swap this implementation for SQL without changing tools."""

    def __init__(self, fixtures: Path):
        # 买家资料与订单资料分开保存，便于用不同服务端身份验证访问边界。
        self.buyers = json.loads((fixtures / "buyers.json").read_text(encoding="utf-8"))
        self.orders = json.loads((fixtures / "orders.json").read_text(encoding="utf-8"))
        self.stock = json.loads((fixtures / "inventory.json").read_text(encoding="utf-8"))
        # 价格保护数据单独存放，便于演示时替换当前价而不改订单金额；缺失文件表示
        # 当前商品没有价格凭证，工具必须停止分析并转人工复核。
        price_path = fixtures / "price_adjustments.json"
        self.price_changes = json.loads(price_path.read_text(encoding="utf-8")) if price_path.exists() else {}

    def list_buyers(self) -> list[dict]:
        """返回脱敏的演示买家列表；真实系统应改为认证系统提供身份。"""
        return list(self.buyers.values())

    def has_buyer(self, buyer_id: str) -> bool:
        """判断服务端配置的演示买家是否存在，避免使用拼写错误的身份启动服务。"""
        return buyer_id in self.buyers

    def get_order(self, order_id: str, buyer_id: str) -> dict | None:
        order = self.orders.get(order_id)
        if not order or order["buyer_id"] != buyer_id:
            return None
        return {key: value for key, value in order.items() if key != "buyer_id"}

    def owner_of_order(self, order_id: str) -> str | None:
        """只供受控执行边界核验归属，不能把该结果直接暴露给模型。"""
        return (self.orders.get(order_id) or {}).get("buyer_id")

    def query_orders(self, buyer_id: str, keyword: str = "", after_order_id: str | None = None,
                     page_size: int = 20, order_ids: tuple[str, ...] | None = None) -> dict:
        """按订单号稳定排序，查询本人匹配商品；多取一条判断是否仍有下一页。"""
        if not 1 <= page_size <= 50:
            raise ValueError("分页大小必须在 1 到 50 之间")
        matched = []
        for order_id, order in sorted(self.orders.items()):
            if order["buyer_id"] != buyer_id or (order_ids is not None and order_id not in order_ids):
                continue
            if after_order_id is not None and order_id <= after_order_id:
                continue
            # 商品关键词同时兼容已登记的品牌、型号；缺失字段不能猜测补齐。
            if keyword.casefold() not in " ".join(str(order.get(key, "")) for key in ("product", "sku", "order_id", "brand", "model")).casefold():
                continue
            matched.append({key: order[key] for key in ("order_id", "product", "sku")})
            if len(matched) > page_size:
                break
        more = len(matched) > page_size
        items = matched[:page_size]
        return {"items": items, "has_more": more,
                "next_after_order_id": items[-1]["order_id"] if more else None}

    def inventory(self, sku: str) -> int:
        return int(self.stock.get(sku, 0))

    def price_adjustment(self, order_id: str, buyer_id: str) -> dict | None:
        """按订单和买家隔离读取演示购买价、当前价及降价时间。"""
        order = self.orders.get(order_id)
        if not order or order.get("buyer_id") != buyer_id:
            return None
        return self.price_changes.get(order_id)
