"""使用项目 B 自己的 FastAPI HTTP 接口验收退换货工单闭环。"""

import json
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from skillflow.api import create_app
from skillflow.config import ROOT, Settings


def main():
    """在项目 B 的独立 JSON Store 中验证草稿、确认、幂等、越权和资格拦截。"""
    # 使用项目 B 自己的忽略目录，避免 Windows 临时目录权限差异；每次运行使用独立目录。
    data_dir = ROOT / ".skillflow" / ("ticket-verification-" + uuid4().hex)
    settings = Settings(model_mode="mock", fixtures_dir=ROOT / "data", data_dir=data_dir)
    with TestClient(create_app(settings=settings)) as client:
        session = str(uuid4())
        payload = {"session_id": session, "order_id": "A123", "request_type": "return", "reason": "商品质量问题"}
        draft = client.post("/tickets/draft", json=payload)
        assert draft.status_code == 200
        draft_data = draft.json()
        assert draft_data["status"] == "draft_created"
        ticket_id = draft_data["ticket"]["ticket_id"]
        assert draft_data["ticket"]["status"] == "draft"

        # 未确认时不应变成 submitted，确认后再次提交必须复用同一个工单号。
        kept = client.post(f"/tickets/{ticket_id}/confirm", json={"session_id": session, "confirmed": False}).json()
        assert kept["ticket"]["status"] == "draft"
        submitted = client.post(f"/tickets/{ticket_id}/confirm", json={"session_id": session, "confirmed": True}).json()
        assert submitted["status"] == "submitted"
        repeated = client.post(f"/tickets/{ticket_id}/confirm", json={"session_id": session, "confirmed": True}).json()
        assert repeated["status"] == "already_submitted"
        assert repeated["ticket"]["ticket_id"] == ticket_id

        # 同一 buyer_id 换新聊天会话后仍能恢复原工单，session_id 只记录操作来源。
        recovered = client.post("/tickets/draft", json={
            "session_id": str(uuid4()), "order_id": "A123", "request_type": "return", "reason": "商品退货申请"}).json()
        assert recovered["status"] == "already_submitted"
        assert recovered["ticket"]["ticket_id"] == ticket_id

        # 同一订单的另一种售后类型不能并行提交；取消原工单后才允许切换路径。
        conflict = client.post("/tickets/draft", json={
            "session_id": session, "order_id": "A123", "request_type": "exchange", "reason": "商品换货申请"}).json()
        assert conflict["status"] == "rejected" and conflict["error_code"] == "TICKET_CONFLICT", conflict
        cancelled = client.post(f"/tickets/{ticket_id}/cancel", json={"session_id": session}).json()
        assert cancelled["status"] == "cancelled"
        cancelled_confirm = client.post(f"/tickets/{ticket_id}/confirm", json={"session_id": session, "confirmed": True}).json()
        assert cancelled_confirm["status"] == "rejected" and cancelled_confirm["error_code"] == "TICKET_CANCELLED"
        switched = client.post("/tickets/draft", json={
            "session_id": session, "order_id": "A123", "request_type": "exchange", "reason": "商品换货申请"}).json()
        assert switched["status"] == "draft_created", switched

        # 缺货、无政策、超期和越权订单不能进入工单写入路径。
        # 优惠争议只能由金额不一致触发；提交后重复确认复用同一个工单号。
        consistent = client.post("/tickets/draft", json={
            "session_id": "discount-consistent", "order_id": "A123",
            "request_type": "discount_dispute", "reason": "金额核对一致不应建单"}).json()
        assert consistent["status"] == "rejected" and consistent["ticket"] is None
        dispute = client.post("/tickets/draft", json={
            "session_id": "discount-dispute", "order_id": "A127",
            "request_type": "discount_dispute", "reason": "优惠金额存在差异"}).json()
        assert dispute["status"] == "draft_created"
        assert dispute["ticket"]["request_type"] == "discount_dispute"
        dispute_id = dispute["ticket"]["ticket_id"]
        submitted_dispute = client.post(f"/tickets/{dispute_id}/confirm", json={
            "session_id": "discount-dispute", "confirmed": True}).json()
        assert submitted_dispute["status"] == "submitted"
        repeated_dispute = client.post(f"/tickets/{dispute_id}/confirm", json={
            "session_id": "discount-dispute", "confirmed": True}).json()
        assert repeated_dispute["status"] == "already_submitted"
        assert repeated_dispute["ticket"]["ticket_id"] == dispute_id

        for order_id, request_type in (("A124", "exchange"), ("A125", "return"), ("A126", "return"), ("A900", "return")):
            response = client.post("/tickets/draft", json={
                "session_id": str(uuid4()), "order_id": order_id,
                "request_type": request_type, "reason": "申请售后"})
            assert response.status_code == 200
            rejected = response.json()
            assert rejected["status"] == "rejected" and rejected["ticket"] is None, rejected

        # 使用不同的服务端 buyer_id 读取或确认原工单时统一返回 404，不泄露工单详情。
        foreign_settings = Settings(model_mode="mock", fixtures_dir=ROOT / "data", data_dir=data_dir,
                                     demo_buyer_id="other-buyer")
        with TestClient(create_app(settings=foreign_settings)) as foreign_client:
            assert foreign_client.get(f"/tickets/{ticket_id}?session_id=foreign").status_code == 404
            assert foreign_client.post(f"/tickets/{ticket_id}/confirm", json={
                "session_id": "foreign", "confirmed": True}).status_code == 404

        # 第三个演示买家只能看到自己的 A901，不能通过订单号或聊天会话读取 A123。
        alice_settings = Settings(model_mode="mock", fixtures_dir=ROOT / "data", data_dir=data_dir / "alice",
                                  demo_buyer_id="alice-buyer")
        with TestClient(create_app(settings=alice_settings)) as alice_client:
            assert {item["order_id"] for item in alice_client.get("/demo/orders").json()} == {"A901"}
            assert alice_client.get(f"/tickets/{ticket_id}?session_id=alice").status_code == 404

    print(json.dumps({"status": "PASS", "cases": 14, "checked": [
        "eligible draft", "explicit confirmation", "same-session idempotency", "cross-session recovery",
        "cross-type conflict rejection", "cancelled confirmation rejection", "cancel then switch request type", "discount dispute gate and idempotency",
        "buyer isolation", "multiple buyer order visibility", "out-of-stock and missing-policy rejection",
        "expired and foreign-order rejection"
    ]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
