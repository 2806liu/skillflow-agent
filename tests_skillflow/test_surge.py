"""项目 B 质量事件固定案例；所有数据写入 .skillflow/surge-tests 的隔离库。"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import shutil
import sqlite3
import unittest
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from fastapi.testclient import TestClient

from skillflow.api import create_app
from skillflow.auth import CSRF_COOKIE
from skillflow.business.surge import BusinessRepository
from skillflow.config import ROOT, Settings
from skillflow.sqlite_store import SqliteStore
from scripts.surge_admin import seed
from tests_skillflow.test_auth import FakeIdentityProvider


class SurgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 固定样本只创建一次；与正在使用的 Alice/Bob 数据库隔离。
        cls.base = (ROOT / ".skillflow" / "surge-tests").resolve()
        cls.base.mkdir(parents=True, exist_ok=True)
        cls.root = (cls.base / ("unit-" + uuid4().hex)).resolve()
        cls.engines = []
        store = SqliteStore(cls.root)
        cls.repository = BusinessRepository(store.database_path)
        seed(cls.repository, 10000, 1000)
        cls.repository.change_incident("SIM-POWERBANK-001", actor="reviewer",
                                       action="publish", reason="固定案例审核通过")

    @classmethod
    def tearDownClass(cls):
        # 只清理本测试创建且验证位于 B 隔离根目录下的路径。
        for engine in cls.engines:
            engine.dispose()
        cls.repository.engine.dispose()
        if cls.root.is_relative_to(cls.base) and cls.root != cls.base:
            shutil.rmtree(cls.root)

    def client_for(self, buyer_id: str) -> TestClient:
        settings = Settings(data_dir=self.root, fixtures_dir=ROOT / "data", model_mode="mock",
                            auth_mode="demo", business_backend="sqlalchemy", demo_buyer_id=buyer_id)
        app = create_app(settings)
        self.engines.append(app.state.repository.engine)
        return TestClient(app)

    def test_migration_preserves_existing_project_b_state(self):
        """首次启用 SQLAlchemy 前保留旧会话快照，且新业务表可正常建立。"""
        directory = (self.base / ("migration-" + uuid4().hex)).resolve()
        try:
            store = SqliteStore(directory)
            store.write("session_before", {"owner": "demo-buyer"})
            repository = BusinessRepository(store.database_path)
            repository.engine.dispose()
            backups = list(directory.glob("skillflow.pre-business-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            with closing(sqlite3.connect(backups[0])) as backup:
                self.assertIsNotNone(backup.execute(
                    "SELECT value_json FROM kv WHERE name='session_before'").fetchone())
        finally:
            if directory.is_relative_to(self.base) and directory != self.base:
                shutil.rmtree(directory)

    def test_a_batch_decisions_and_alternative(self):
        """可信批次可判定范围；缺失或自报批次只能待核实，候选品仅来自库存目录。"""
        expected = ["AFFECTED", "NOT_AFFECTED", "UNVERIFIED", "UNVERIFIED", "AFFECTED"]
        for i, decision in enumerate(expected):
            with self.client_for(f"surge-buyer-{i:04d}") as client:
                response = client.post("/chat", json={"session_id": f"case-{i}",
                                                      "message": f"查询订单 A{100000+i} 的批次和替代品"})
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(result["result"]["incident"]["decision"], decision)
                if i in (2, 3, 4):
                    self.assertTrue(result["result"]["incident"]["manual_review"])
                if i == 0:
                    self.assertEqual(result["result"]["alternatives"][0]["sku"], "PB-SIM-B")
                    self.assertTrue(result["result"]["assess_after_sales"]["return_eligible"])
        # 用户自报来源不能仅凭 verified 标志误置就升级为可信批次。
        with self.repository.engine.begin() as conn:
            conn.exec_driver_sql("UPDATE business_order_items SET batch_verified=1 WHERE order_id='A100003'")
        try:
            self.assertEqual(self.repository.evaluate("A100003", "surge-buyer-0003")["decision"],
                             "UNVERIFIED")
        finally:
            with self.repository.engine.begin() as conn:
                conn.exec_driver_sql("UPDATE business_order_items SET batch_verified=0 WHERE order_id='A100003'")

    def test_alternatives_without_order_exclude_active_incident_sku(self):
        """直接询问替代品可得到目录候选，生效事件涉及的 SKU 不进入推荐。"""
        # 故意给受影响 SKU 填入库存，验证筛除依据是已生效事件而非库存为零。
        with self.repository.engine.begin() as conn:
            conn.exec_driver_sql("UPDATE business_products SET stock=5 WHERE sku='PB-SIM-A'")
        try:
            with self.client_for("surge-buyer-0000") as client:
                response = client.post("/chat", json={"session_id": "alternatives-only",
                                                       "message": "有哪些充电宝替代品？"})
                self.assertEqual(response.status_code, 200)
                choices = response.json()["result"]["alternatives"]
                self.assertEqual([item["sku"] for item in choices], ["PB-SIM-B"])
        finally:
            with self.repository.engine.begin() as conn:
                conn.exec_driver_sql("UPDATE business_products SET stock=0 WHERE sku='PB-SIM-A'")

    def test_incident_context_survives_service_restart(self):
        """事件咨询独立 KV 保存当前订单；重启后无订单号追问仍回到本人订单。"""
        with self.client_for("surge-buyer-0000") as client:
            first = client.post("/chat", json={"session_id": "incident-restart",
                                               "message": "查询订单 A100000 的充电宝批次"})
            self.assertEqual(first.status_code, 200)
        with self.client_for("surge-buyer-0000") as restarted:
            followup = restarted.post("/chat", json={"session_id": "incident-restart",
                                                    "message": "这个能退货吗？"})
            self.assertEqual(followup.status_code, 200)
            self.assertEqual(followup.json()["result"]["incident"]["order_id"], "A100000")

    def test_b_draft_requires_explicit_confirmation(self):
        """调查不建单；草稿不提交；确认后的历史只有一条正式确认。"""
        with self.client_for("surge-buyer-0010") as client:
            chat = client.post("/chat", json={"session_id": "draft", "message": "A100010 能退吗"}).json()
            self.assertNotIn("ticket", chat["result"])
            draft = client.post("/tickets/draft", json={"session_id": "draft", "order_id": "A100010",
                                                          "request_type": "return", "reason": "模拟质量事件退货"}).json()
            self.assertEqual(draft["status"], "draft_created")
            ticket_id = draft["ticket"]["ticket_id"]
            declined = client.post(f"/tickets/{ticket_id}/confirm", json={
                "session_id": "draft", "confirmed": False}).json()
            self.assertEqual(declined["ticket"]["status"], "draft")
            confirmed = client.post(f"/tickets/{ticket_id}/confirm", json={
                "session_id": "draft", "confirmed": True}).json()
            self.assertEqual(confirmed["ticket"]["status"], "submitted")
            again = client.post(f"/tickets/{ticket_id}/confirm", json={
                "session_id": "draft", "confirmed": True}).json()
            self.assertEqual(again["status"], "already_submitted")
            self.assertEqual([event["action"] for event in again["ticket_events"]].count("confirm"), 1)

    def test_c_concurrent_confirm_is_idempotent(self):
        """100 次并发重复确认仍只有一张有效工单、一条正式提交事件。"""
        with self.client_for("surge-buyer-0015") as client:
            draft = client.post("/tickets/draft", json={"session_id": "race", "order_id": "A100015",
                                                          "request_type": "return", "reason": "模拟质量事件退货"}).json()
        ticket_id = draft["ticket"]["ticket_id"]
        from skillflow.business.sqlite_tickets import SqliteTicketStore
        tickets = SqliteTicketStore(SqliteStore(self.root))
        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(pool.map(lambda _: tickets.confirm_incident(ticket_id, session_id="race",
                                                                        buyer_id="surge-buyer-0015"), range(100)))
        self.assertEqual(sum(changed for _, changed, _ in results), 1)
        self.assertEqual([item.action for item in tickets.events(tickets.get(ticket_id))].count("confirm"), 1)

    def test_d_cross_buyer_cannot_read_order(self):
        """演示身份无法获取其他买家的订单和批次事实。"""
        with self.client_for("surge-buyer-0000") as client:
            response = client.get("/orders/A100001")
            self.assertEqual(response.status_code, 404)
            self.assertNotIn("B20260090", response.text)

    def test_e_production_identity_and_run_isolation(self):
        """真实授权边界由服务端 OIDC 身份映射决定；越权订单和执行记录均返回 403。"""
        settings = Settings(data_dir=self.root, fixtures_dir=ROOT / "data", model_mode="mock",
                            auth_mode="production", business_backend="sqlalchemy",
                            oidc_issuer="https://identity.test", oidc_client_id="surge-test",
                            oidc_client_secret="test-secret", oidc_redirect_uri="https://testserver/auth/callback")
        app = create_app(settings, identity_provider=FakeIdentityProvider())
        self.engines.append(app.state.repository.engine)
        app.state.store.provision_identity(settings.oidc_issuer, "subject-zero", "surge-buyer-0000")
        app.state.store.provision_identity(settings.oidc_issuer, "subject-one", "surge-buyer-0001")

        def login(client, subject):
            begin = client.get("/auth/login", follow_redirects=False)
            state = parse_qs(urlparse(begin.headers["location"]).query)["state"][0]
            callback = client.post("/auth/callback", data={"state": state, "code": subject},
                                   follow_redirects=False)
            self.assertEqual(callback.status_code, 303)
            return {"X-CSRF-Token": client.cookies.get(CSRF_COOKIE)}

        with TestClient(app, base_url="https://testserver") as zero, TestClient(
                app, base_url="https://testserver") as one:
            headers_zero = login(zero, "subject-zero")
            headers_one = login(one, "subject-one")
            session_one = one.post("/sessions", headers=headers_one).json()["session_id"]
            self.assertEqual(zero.get("/orders/A100001").status_code, 403)
            self.assertEqual(zero.get("/orders/A100001").json()["detail"]["code"], "ORDER_FORBIDDEN")
            own = one.post("/chat", headers=headers_one, json={
                "session_id": session_one, "message": "A100001 批次"})
            self.assertEqual(own.status_code, 200, own.text)
            self.assertEqual(own.json()["result"]["incident"]["decision"], "NOT_AFFECTED")
            self.assertEqual(zero.get("/runs/" + own.json()["run_id"]).status_code, 403)
            session_zero = zero.post("/sessions", headers=headers_zero).json()["session_id"]
            forbidden = zero.post("/chat", headers=headers_zero, json={
                "session_id": session_zero, "message": "A100001 批次"})
            self.assertEqual(forbidden.status_code, 403)
            self.assertEqual(forbidden.json()["detail"]["code"], "ORDER_FORBIDDEN")

    def test_z_withdraw_blocks_confirmation(self):
        """草稿之后撤回事件时，事务复核拒绝提交并保留审计。"""
        with self.client_for("surge-buyer-0020") as client:
            draft = client.post("/tickets/draft", json={"session_id": "withdraw", "order_id": "A100020",
                                                          "request_type": "return", "reason": "模拟质量事件退货"}).json()
            ticket_id = draft["ticket"]["ticket_id"]
            self.repository.change_incident("SIM-POWERBANK-001", actor="reviewer",
                                            action="withdraw", reason="演示撤回")
            result = client.post(f"/tickets/{ticket_id}/confirm", json={
                "session_id": "withdraw", "confirmed": True}).json()
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(result["error_code"], "INCIDENT_CHANGED_OR_UNVERIFIED")
            self.assertEqual(result["ticket"]["status"], "draft")
            # 撤回后即使原草稿存在，也不能借重复申请绕过事件复核。
            retry = client.post("/tickets/draft", json={"session_id": "withdraw", "order_id": "A100020",
                                                        "request_type": "return", "reason": "重复申请"}).json()
            self.assertEqual(retry["status"], "rejected")
            outcome = client.post("/chat", json={"session_id": "withdraw", "message": "A100020 的批次"}).json()
            self.assertEqual(outcome["result"]["incident"]["decision"], "UNVERIFIED")


if __name__ == "__main__":
    unittest.main()
