"""B6 验收：在本地测试身份提供器下验证两买家授权、CSRF、会话失效和只读回放。"""

import asyncio
from contextlib import closing
from dataclasses import replace
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from fastapi.testclient import TestClient
import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from skillflow.api import create_app
from skillflow.auth import AUTH_COOKIE, AuthError, CSRF_COOKIE, OidcProvider
from skillflow.config import ROOT, Settings


class FakeIdentityProvider:
    """测试适配器只模拟已验证主体；实际 OIDC 的签名与 nonce 由生产适配器检查。"""

    async def authorization_url(self, state, nonce, challenge):
        return "https://identity.test/authorize?state=" + state

    async def exchange_code(self, code, verifier, nonce):
        return {"subject": code}

    async def end_session_url(self):
        """模拟已发现的 IdP 登出端点；浏览器是否确认退出由人工端到端验收。"""
        return "https://identity.test/logout?client_id=b6-test"


class AuthTests(unittest.TestCase):
    def setUp(self):
        # 所有测试数据都复制到项目 B 的临时目录，避免修改演示订单和政策。
        self.root = Path(tempfile.mkdtemp(prefix="b6-", dir=ROOT / ".skillflow"))
        fixtures = self.root / "fixtures"
        (fixtures / "policies").mkdir(parents=True)
        for name in ("buyers.json", "orders.json", "inventory.json", "price_adjustments.json"):
            shutil.copyfile(ROOT / "data" / name, fixtures / name)
        shutil.copyfile(ROOT / "data/policies/earbuds_v1.md", fixtures / "policies/earbuds_v1.md")
        settings = Settings(data_dir=self.root / "state", fixtures_dir=fixtures, model_mode="mock",
                            auth_mode="production", oidc_issuer="https://identity.test",
                            oidc_client_id="b6-test", oidc_client_secret="test-secret",
                            oidc_redirect_uri="https://testserver/auth/callback")
        self.settings = settings
        self.app = create_app(settings, identity_provider=FakeIdentityProvider())
        self.app.state.store.provision_identity(settings.oidc_issuer, "alice", "alice-buyer")
        self.app.state.store.provision_identity(settings.oidc_issuer, "bob", "other-buyer")
        self.alice = TestClient(self.app, base_url="https://testserver")
        self.bob = TestClient(self.app, base_url="https://testserver")

    def tearDown(self):
        self.alice.close()
        self.bob.close()
        shutil.rmtree(self.root)

    def login(self, client, subject):
        """完成一次授权码回调，返回随 Cookie 会话绑定的 CSRF 请求头。"""
        response = client.get("/auth/login", follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        state = parse_qs(urlparse(response.headers["location"]).query)["state"][0]
        response = client.post("/auth/callback", data={"state": state, "code": subject}, follow_redirects=False)
        self.assertEqual(response.status_code, 303, response.text)
        # 认证 Cookie 不可由前端脚本读取；CSRF Cookie 可读但仍须服务端双提交校验。
        cookies = response.headers.get_list("set-cookie")
        auth_cookie = next(value.lower() for value in cookies if value.startswith(AUTH_COOKIE + "="))
        csrf_cookie = next(value.lower() for value in cookies if value.startswith(CSRF_COOKIE + "="))
        self.assertIn("httponly", auth_cookie)
        self.assertIn("secure", auth_cookie)
        self.assertIn("samesite=lax", auth_cookie)
        self.assertIn("secure", csrf_cookie)
        self.assertNotIn("httponly", csrf_cookie)
        return {"X-CSRF-Token": client.cookies.get(CSRF_COOKIE)}

    def session(self, client, headers):
        response = client.post("/sessions", headers=headers)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["session_id"]

    def assert_forbidden(self, response, code):
        """越权必须返回 403 和稳定错误码，且不能包含其他买家的业务数据。"""
        self.assertEqual(response.status_code, 403, response.text)
        self.assertEqual(response.json(), {"detail": {"code": code, "message": "无权访问。"}})

    def test_request_monitoring_redacts_credentials_and_business_ids(self):
        """请求监控只保留模板、状态与耗时，401/403 也纳入统计而不泄漏查询参数。"""
        self.login(self.alice, "alice")
        before = self.app.state.request_metrics.snapshot()
        with self.assertLogs("uvicorn.error", level="INFO") as captured:
            healthy = self.alice.get("/health?secret=hidden-value")
            denied = self.alice.get("/orders/A900?token=hidden-value")
            unauthenticated = self.bob.get("/orders/A901?code=hidden-value")
        self.assertEqual((healthy.status_code, denied.status_code, unauthenticated.status_code), (200, 403, 401))
        self.assertEqual(len(healthy.headers["X-Request-ID"]), 32)
        # assertLogs 带 logger 前缀；只解析结构化 JSON 正文。
        records = [json.loads(line[line.index("{"):]) for line in captured.output]
        self.assertEqual({record["status"] for record in records}, {200, 401, 403})
        self.assertEqual({record["route"] for record in records}, {"/health", "/orders/{order_id}"})
        self.assertTrue(all(record["duration_ms"] >= 0 for record in records))
        self.assertNotIn("hidden-value", "\n".join(captured.output))
        self.assertNotIn("A900", "\n".join(captured.output))
        self.assertNotIn("A901", "\n".join(captured.output))
        after = self.app.state.request_metrics.snapshot()
        self.assertEqual(after["requests"] - before["requests"], 3)
        self.assertEqual(after["status_counts"]["4xx"] - before["status_counts"].get("4xx", 0), 2)
        self.assertIsNotNone(after["p95_ms"])

    def test_oidc_login_requires_credentials_when_sso_remains(self):
        """本应用退出后 Keycloak 可能仍有 SSO Cookie；新登录必须要求用户主动输入账号。"""
        provider = OidcProvider(self.settings)
        provider.metadata = {"issuer": self.settings.oidc_issuer,
                             "authorization_endpoint": "https://identity.test/authorize"}
        url = asyncio.run(provider.authorization_url("login-state", "login-nonce", "pkce-challenge"))
        params = parse_qs(urlparse(url).query)
        self.assertEqual(params["prompt"], ["login"])
        self.assertEqual(params["response_mode"], ["form_post"])
        self.assertEqual(params["code_challenge_method"], ["S256"])

    def test_oidc_logout_uses_discovered_same_origin_endpoint(self):
        """退出只能跳往发行者同源 HTTPS 登出端点，URL 不放身份令牌或客户端密钥。"""
        provider = OidcProvider(self.settings)
        provider.metadata = {"issuer": self.settings.oidc_issuer,
                             "end_session_endpoint": "https://identity.test/logout"}
        url = asyncio.run(provider.end_session_url())
        params = parse_qs(urlparse(url).query)
        self.assertEqual(urlparse(url).scheme + "://" + urlparse(url).netloc + urlparse(url).path,
                         "https://identity.test/logout")
        self.assertEqual(params["client_id"], ["b6-test"])
        # 回跳地址仅取自受信任的服务端回调设置，不接受请求端任意指定。
        self.assertEqual(params["post_logout_redirect_uri"], ["https://testserver/"])
        provider.metadata["end_session_endpoint"] = "https://other.test/logout"
        with self.assertRaises(AuthError):
            asyncio.run(provider.end_session_url())

    def test_oidc_container_connect_host_keeps_public_tls_identity(self):
        """容器只改后端 TCP 目标，仍使用公开 issuer 的 Host、SNI 和证书名称。"""
        provider = OidcProvider(replace(self.settings, oidc_connect_host="host.docker.internal"))

        async def check():
            def respond(request):
                self.assertEqual(str(request.url), "https://host.docker.internal/realms/myrealm/certs")
                self.assertEqual(request.headers["host"], "identity.test")
                self.assertEqual(request.extensions["sni_hostname"], "identity.test")
                return httpx.Response(200, json={"issuer": "https://identity.test"})

            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                result = await provider._provider_request(
                    client, "GET", "https://identity.test/realms/myrealm/certs")
                self.assertEqual(result.status_code, 200)
                # 发现文档不能把请求引向与公开 issuer 不同源的内部或外部地址。
                with self.assertRaises(AuthError):
                    await provider._provider_request(client, "GET", "https://other.test/certs")

        asyncio.run(check())
        with self.assertRaises(ValueError):
            replace(self.settings, oidc_connect_host="host.docker.internal/unsafe")

    def test_restart_preserves_login_ownership_ticket_and_run(self):
        """重建应用后复用已签发 Cookie，核对买家隔离、工单历史和执行记录仍在项目 B 数据库中。"""
        alice_headers = self.login(self.alice, "alice")
        bob_headers = self.login(self.bob, "bob")
        alice_session = self.session(self.alice, alice_headers)
        bob_session = self.session(self.bob, bob_headers)
        run = self.alice.post("/chat", headers=alice_headers, json={
            "session_id": alice_session, "message": "查询订单 A901"})
        self.assertEqual(run.status_code, 200, run.text)
        run_id = run.json()["run_id"]
        draft = self.alice.post("/tickets/draft", headers=alice_headers, json={
            "session_id": alice_session, "order_id": "A901", "request_type": "return",
            "reason": "质量问题申请退货"})
        self.assertEqual(draft.status_code, 200, draft.text)
        ticket_id = draft.json()["ticket"]["ticket_id"]
        confirmed = self.alice.post("/tickets/" + ticket_id + "/confirm", headers=alice_headers,
                                    json={"session_id": alice_session, "confirmed": True})
        self.assertEqual(confirmed.status_code, 200, confirmed.text)

        # 新应用实例使用同一数据目录，相当于服务重启；不重新登录以验证认证会话持久化。
        restarted = create_app(self.settings, identity_provider=FakeIdentityProvider())
        with TestClient(restarted, base_url="https://testserver") as alice_after, \
                TestClient(restarted, base_url="https://testserver") as bob_after:
            alice_after.cookies.update(self.alice.cookies)
            bob_after.cookies.update(self.bob.cookies)
            self.assertEqual(alice_after.get("/auth/me").json()["buyer_id"], "alice-buyer")
            self.assertEqual(bob_after.get("/auth/me").json()["buyer_id"], "other-buyer")
            self.assertEqual(alice_after.get("/sessions/" + alice_session).status_code, 200)
            self.assertEqual(bob_after.get("/sessions/" + bob_session).status_code, 200)
            self.assertEqual(alice_after.get("/orders/A901").status_code, 200)
            self.assertEqual(bob_after.get("/orders/A900").status_code, 200)
            self.assert_forbidden(bob_after.get("/orders/A901"), "ORDER_FORBIDDEN")
            self.assert_forbidden(alice_after.get("/orders/A900"), "ORDER_FORBIDDEN")
            self.assertEqual(alice_after.get("/runs/" + run_id).status_code, 200)
            self.assert_forbidden(bob_after.get("/runs/" + run_id), "RUN_FORBIDDEN")
            ticket = alice_after.get("/tickets/" + ticket_id, params={"session_id": alice_session})
            self.assertEqual(ticket.status_code, 200, ticket.text)
            self.assertEqual(ticket.json()["ticket"]["status"], "submitted")
            self.assert_forbidden(bob_after.get("/tickets/" + ticket_id,
                                                params={"session_id": bob_session}), "TICKET_FORBIDDEN")
            history = alice_after.get("/tickets/" + ticket_id + "/events",
                                      params={"session_id": alice_session})
            self.assertEqual([event["action"] for event in history.json()], ["draft_created", "confirm"])

    def test_auth_ownership_replay_and_revocation(self):
        self.assertEqual(self.alice.post("/chat", json={"session_id": "fake", "message": "A901"}).status_code, 401)
        alice_headers = self.login(self.alice, "alice")
        bob_headers = self.login(self.bob, "bob")
        a = self.session(self.alice, alice_headers)
        b = self.session(self.bob, bob_headers)
        self.assertEqual(self.alice.get("/orders/A901").status_code, 200)
        self.assert_forbidden(self.bob.get("/orders/A901"), "ORDER_FORBIDDEN")
        self.assert_forbidden(self.alice.get("/orders/A900"), "ORDER_FORBIDDEN")
        self.assert_forbidden(self.bob.get("/sessions/" + a), "SESSION_FORBIDDEN")
        self.assert_forbidden(self.bob.post("/chat", json={"session_id": a, "message": "A901"}, headers=bob_headers), "SESSION_FORBIDDEN")
        self.assert_forbidden(self.alice.post("/chat", json={"session_id": a, "message": "A900"}, headers=alice_headers), "ORDER_FORBIDDEN")
        self.assertEqual(self.alice.post("/chat", json={"session_id": a, "message": "A901"}).status_code, 403)
        # 流式聊天也必须在建立 run_id 前拒绝他人的订单，不能产生可回放记录。
        denied_stream = self.alice.post("/chat/stream", headers=alice_headers, json={
            "run_id": str(uuid4()), "session_id": a, "message": "查询订单 A900"})
        self.assert_forbidden(denied_stream, "ORDER_FORBIDDEN")
        self.assertEqual(self.alice.get("/orders/UNKNOWN").status_code, 404)
        self.assertEqual(self.alice.get("/sessions/" + str(uuid4())).status_code, 404)
        with closing(self.app.state.store.connect()) as conn:
            # 401、403 与跨买家 404 在进入 Planner 前结束，不留下运行或工单写入。
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM run_owners").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM ticket_events").fetchone()[0], 0)
        result = self.alice.post("/chat", json={"session_id": a, "message": "查询订单 A901"}, headers=alice_headers)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertNotIn(self.alice.cookies.get(AUTH_COOKIE), result.text)
        self.assertNotIn(alice_headers["X-CSRF-Token"], result.text)
        run_id = result.json()["run_id"]
        self.assertEqual(self.alice.get("/runs/" + run_id).status_code, 200)
        self.assert_forbidden(self.bob.get("/runs/" + run_id), "RUN_FORBIDDEN")
        # 普通聊天的 run 没有 SSE 资源，此路径即使知道 run_id 也应是 404。
        self.assertEqual(self.bob.get("/runs/" + run_id + "/stream", params={"session_id": a}).status_code, 404)
        self.assertEqual(self.bob.get("/demo/orders").status_code, 404)
        self.assertEqual(self.bob.get("/health").json(), {"status": "ok", "auth_mode": "production"})
        logout = self.alice.post("/auth/logout", headers=alice_headers, follow_redirects=False)
        self.assertEqual(logout.status_code, 200)
        self.assertEqual(logout.json()["logout_url"], "https://identity.test/logout?client_id=b6-test")
        self.assertEqual(self.alice.get("/runs/" + run_id).status_code, 401)
        self.assertEqual(self.bob.get("/sessions/" + b).status_code, 200)
        # 同一买家重新登录后仍可恢复自己已持久化的会话与执行记录。
        self.login(self.alice, "alice")
        self.assertEqual(self.alice.get("/sessions/" + a).status_code, 200)
        self.assertEqual(self.alice.get("/runs/" + run_id).status_code, 200)

    def test_ticket_and_stream_replay_isolation(self):
        alice_headers = self.login(self.alice, "alice")
        bob_headers = self.login(self.bob, "bob")
        a, b = self.session(self.alice, alice_headers), self.session(self.bob, bob_headers)
        draft = self.alice.post("/tickets/draft", headers=alice_headers, json={
            "session_id": a, "order_id": "A901", "request_type": "return", "reason": "质量问题申请退货"})
        self.assertEqual(draft.status_code, 200, draft.text)
        ticket_id = draft.json()["ticket"]["ticket_id"]
        self.assert_forbidden(self.bob.get("/tickets/" + ticket_id, params={"session_id": b}), "TICKET_FORBIDDEN")
        self.assert_forbidden(self.bob.get("/tickets/" + ticket_id + "/events", params={"session_id": b}), "TICKET_FORBIDDEN")
        self.assert_forbidden(self.bob.post("/tickets/" + ticket_id + "/confirm", headers=bob_headers,
                                       json={"session_id": b, "confirmed": True}), "TICKET_FORBIDDEN")
        self.assert_forbidden(self.bob.post("/tickets/" + ticket_id + "/cancel", headers=bob_headers,
                                       json={"session_id": b}), "TICKET_FORBIDDEN")
        events = self.alice.get("/tickets/" + ticket_id + "/events", params={"session_id": a}).json()
        self.assertEqual([event["action"] for event in events].count("draft_created"), 1)
        stream_id = str(uuid4())
        body = {"run_id": stream_id, "session_id": a, "message": "查询订单 A901"}
        self.assertEqual(self.alice.post("/chat/stream", headers=alice_headers, json=body).status_code, 200)
        self.assertEqual(self.alice.get("/runs/" + stream_id + "/stream", params={"session_id": a}).status_code, 200)
        self.assert_forbidden(self.bob.get("/runs/" + stream_id + "/stream", params={"session_id": a}), "RUN_FORBIDDEN")
        self.assert_forbidden(self.bob.post("/chat/stream", headers=bob_headers, json=body), "SESSION_FORBIDDEN")
        # Bob 用自己的会话抢占 Alice 已登记的 run_id，也必须按越权拒绝。
        collision = {"run_id": stream_id, "session_id": b, "message": "查询订单 A900"}
        self.assert_forbidden(self.bob.post("/chat/stream", headers=bob_headers, json=collision), "RUN_FORBIDDEN")

        # 同一买家跨会话重复申请和确认必须复用原工单，历史只记录一次创建。
        another = self.session(self.alice, alice_headers)
        again = self.alice.post("/tickets/draft", headers=alice_headers, json={
            "session_id": another, "order_id": "A901", "request_type": "return", "reason": "重复申请退货"})
        self.assertEqual(again.status_code, 200, again.text)
        self.assertEqual(again.json()["ticket"]["ticket_id"], ticket_id)
        for _ in range(2):
            confirmed = self.alice.post("/tickets/" + ticket_id + "/confirm", headers=alice_headers,
                                        json={"session_id": a, "confirmed": True})
            self.assertEqual(confirmed.status_code, 200, confirmed.text)
        events = self.alice.get("/tickets/" + ticket_id + "/events", params={"session_id": a}).json()
        self.assertEqual([event["action"] for event in events].count("draft_created"), 1)
        self.assertEqual([event["action"] for event in events].count("confirm"), 1)

    def test_production_requires_provider_and_old_ownerless_run_is_hidden(self):
        """缺少配置须启动失败；历史 KV 无归属时即使拿到 run_id 也不可读取。"""
        with self.assertRaises(ValueError):
            Settings(auth_mode="production", oidc_issuer="", oidc_client_id="", oidc_client_secret="")
        headers = self.login(self.alice, "alice")
        run_id = str(uuid4())
        self.app.state.store.write("run_" + run_id, {"answer": "legacy"})
        self.assertEqual(self.alice.get("/runs/" + run_id).status_code, 404)
        self.assertEqual(self.alice.post("/sessions", headers=headers).status_code, 200)

    def test_expired_session_and_unbound_identity(self):
        """未预先绑定的 IdP 主体不能继承演示买家；服务端到期立即拒绝。"""
        response = self.bob.get("/auth/login", follow_redirects=False)
        state = parse_qs(urlparse(response.headers["location"]).query)["state"][0]
        denied = self.bob.post("/auth/callback", data={"state": state, "code": "unbound"})
        self.assertEqual(denied.status_code, 401)
        headers = self.login(self.alice, "alice")
        session_id = self.session(self.alice, headers)
        with self.app.state.store.transaction() as conn:
            conn.execute("UPDATE auth_sessions SET expires_at=0 WHERE buyer_id=?", ("alice-buyer",))
        self.assertEqual(self.alice.get("/sessions/" + session_id).status_code, 401)
        self.assertEqual(self.alice.post("/tickets/draft", headers=headers, json={
            "session_id": session_id, "order_id": "A901", "request_type": "return", "reason": "过期尝试"}).status_code, 401)

    def test_oidc_adapter_verifies_signature_and_nonce(self):
        """真实生产适配器只接受可信 issuer、audience、RSA 签名和本次授权的 nonce。"""
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
        public["kid"] = "test-key"
        settings = Settings(auth_mode="production", model_mode="mock", oidc_issuer="https://identity.test",
                            oidc_client_id="b6-test", oidc_client_secret="test-secret",
                            oidc_redirect_uri="https://testserver/auth/callback")
        token = jwt.encode({"iss": settings.oidc_issuer, "aud": settings.oidc_client_id,
                            "sub": "alice", "iat": int(time.time()), "exp": int(time.time()) + 300,
                            "nonce": "correct"}, key, algorithm="RS256", headers={"kid": "test-key"})

        def handler(request):
            # 测试传输完全在内存中，未向真实身份提供方发送任何网络请求。
            if request.url.path.endswith("openid-configuration"):
                return httpx.Response(200, json={"issuer": settings.oidc_issuer,
                    "authorization_endpoint": "https://identity.test/authorize",
                    "token_endpoint": "https://identity.test/token", "jwks_uri": "https://identity.test/jwks"})
            if request.url.path == "/token":
                return httpx.Response(200, json={"id_token": token})
            return httpx.Response(200, json={"keys": [public]})

        original = httpx.AsyncClient
        def local_client(**kwargs):
            return original(transport=httpx.MockTransport(handler), **kwargs)

        with patch("skillflow.auth.httpx.AsyncClient", side_effect=local_client):
            provider = OidcProvider(settings)
            login_url = asyncio.run(provider.authorization_url("state", "correct", "challenge"))
            self.assertEqual(parse_qs(urlparse(login_url).query)["response_mode"], ["form_post"])
            self.assertEqual(asyncio.run(provider.exchange_code("test-code", "verifier", "correct"))["subject"], "alice")
            with self.assertRaises(AuthError):
                asyncio.run(provider.exchange_code("test-code", "verifier", "wrong"))


if __name__ == "__main__":
    unittest.main()
