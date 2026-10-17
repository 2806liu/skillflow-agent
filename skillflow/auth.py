"""项目 B 的身份入口：OIDC 验证后签发本地会话，业务代码只接收可信 Principal。

浏览器 Cookie 是认证凭证；聊天 session_id 只是会话索引。失败时不回退演示身份，
OIDC 密钥、令牌和 Cookie 也不进入 LangGraph 状态、Trace 或回答。
"""

import base64
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx


AUTH_COOKIE = "__Host-skillflow"
CSRF_COOKIE = "__Host-skillflow-csrf"


class AuthError(Exception):
    """认证来源或授权对象无效；API 将其映射为不泄漏细节的拒绝响应。"""


@dataclass(frozen=True)
class Principal:
    """每次请求独立生成的可信买家身份；session_hash 仅用于服务端校验 CSRF。"""

    buyer_id: str
    role: str = "buyer"
    session_hash: str = ""


class IdentityProvider(Protocol):
    """外部身份提供方只完成登录，不直接取得项目 B 的业务工具调用权。"""

    async def authorization_url(self, state: str, nonce: str, challenge: str) -> str: ...

    async def exchange_code(self, code: str, verifier: str, nonce: str) -> dict: ...

    async def end_session_url(self) -> str: ...


class OidcProvider:
    """从固定 HTTPS 发行方发现端点并验证 ID Token 的签名、受众和 nonce。"""

    def __init__(self, settings):
        # 生产启动即检查签名验证依赖，避免登录后才发现不能验证 ID Token。
        try:
            import jwt  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("生产 OIDC 登录需要安装项目 B 的 PyJWT[crypto] 依赖") from exc
        self.settings = settings
        self.metadata = None

    async def _provider_request(self, client: httpx.AsyncClient, method: str, public_url: str, data=None):
        """容器内连接宿主机时仍校验公开 issuer 的 TLS 名称；只重写传输地址，不改认证语义。"""
        connect_host = self.settings.oidc_connect_host
        if not connect_host:
            return await client.request(method, public_url, data=data)
        issuer_parts, endpoint_parts = urlsplit(self.settings.oidc_issuer), urlsplit(public_url)
        if endpoint_parts.scheme != "https" or endpoint_parts.netloc != issuer_parts.netloc:
            raise AuthError("身份提供方后端端点无效")
        connect_url = urlunsplit(("https", connect_host, endpoint_parts.path, endpoint_parts.query, ""))
        # Host 和 SNI 保留 localhost，证书仍按公开 issuer 严格验证；不能把信任检查关掉。
        request = client.build_request(method, connect_url, data=data,
                                       headers={"Host": issuer_parts.netloc},
                                       extensions={"sni_hostname": issuer_parts.hostname})
        return await client.send(request)

    async def _metadata(self):
        if self.metadata is None:
            url = self.settings.oidc_issuer.rstrip("/") + "/.well-known/openid-configuration"
            try:
                async with httpx.AsyncClient(timeout=10, trust_env=self.settings.oidc_trust_env) as client:
                    response = await self._provider_request(client, "GET", url)
                    response.raise_for_status()
                    metadata = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                # 提供方不可用时拒绝登录，不回退到演示身份或泄露端点响应正文。
                raise AuthError("身份提供方暂不可用") from exc
            if metadata.get("issuer") != self.settings.oidc_issuer or not all(
                str(metadata.get(key, "")).startswith("https://")
                for key in ("authorization_endpoint", "token_endpoint", "jwks_uri")
            ):
                raise AuthError("身份提供方配置无效")
            self.metadata = metadata
        return self.metadata

    async def authorization_url(self, state: str, nonce: str, challenge: str) -> str:
        """生成需主动重新认证的授权码 URL；state、nonce 和 PKCE 阻止回调串用。"""
        metadata = await self._metadata()
        # 本应用允许同一浏览器轮流验收 Alice/Bob；Keycloak SSO Cookie 仍在时，
        # prompt=login 要求重新认证，避免退出本应用后无提示地回到上一个买家。
        query = urlencode({"response_type": "code", "client_id": self.settings.oidc_client_id,
                           "redirect_uri": self.settings.oidc_redirect_uri, "scope": "openid",
                           "response_mode": "form_post", "prompt": "login",
                           "state": state, "nonce": nonce, "code_challenge": challenge,
                           "code_challenge_method": "S256"})
        return metadata["authorization_endpoint"] + "?" + query

    async def end_session_url(self) -> str:
        """返回 Keycloak 公布的浏览器登出地址；不把令牌或客户端密钥放入 URL。"""
        metadata = await self._metadata()
        endpoint = str(metadata.get("end_session_endpoint", ""))
        issuer_parts, endpoint_parts = urlsplit(self.settings.oidc_issuer), urlsplit(endpoint)
        # 浏览器即将离开本应用，只允许跳到已验证发行者的 HTTPS 同源登出端点。
        if endpoint_parts.scheme != "https" or endpoint_parts.netloc != issuer_parts.netloc:
            raise AuthError("身份提供方登出端点无效")
        separator = "&" if endpoint_parts.query else "?"
        # 回跳地址只取自服务端固定的 OIDC 回调配置，避免浏览器参数造成开放重定向。
        # Keycloak 客户端须登记该精确地址为 Valid Post Logout Redirect URI。
        callback_parts = urlsplit(self.settings.oidc_redirect_uri)
        if callback_parts.scheme != "https" or not callback_parts.netloc:
            raise AuthError("应用回跳地址无效")
        return endpoint + separator + urlencode({
            "client_id": self.settings.oidc_client_id,
            "post_logout_redirect_uri": f"{callback_parts.scheme}://{callback_parts.netloc}/",
        })

    async def exchange_code(self, code: str, verifier: str, nonce: str) -> dict:
        """用授权码换取身份，并拒绝伪造、过期或缺少买家映射声明的令牌。"""
        try:
            import jwt  # 生产依赖 PyJWT[crypto]；演示模式和测试提供器无需导入。
            metadata = await self._metadata()
            async with httpx.AsyncClient(timeout=10, trust_env=self.settings.oidc_trust_env) as client:
                token_response = await self._provider_request(client, "POST", metadata["token_endpoint"], data={
                    "grant_type": "authorization_code", "code": code,
                    "redirect_uri": self.settings.oidc_redirect_uri,
                    "client_id": self.settings.oidc_client_id,
                    "client_secret": self.settings.oidc_client_secret,
                    "code_verifier": verifier})
                token_response.raise_for_status()
                token = token_response.json()["id_token"]
                jwks_response = await self._provider_request(client, "GET", metadata["jwks_uri"])
                jwks_response.raise_for_status()
            header = jwt.get_unverified_header(token)
            if header.get("alg") != "RS256" or not header.get("kid"):
                raise AuthError("身份令牌算法无效")
            keys = [key for key in jwks_response.json().get("keys", []) if key.get("kid") == header["kid"]]
            if len(keys) != 1:
                raise AuthError("身份令牌密钥无效")
            public_key = jwt.PyJWK.from_dict(keys[0]).key
            claims = jwt.decode(token, public_key, algorithms=["RS256"],
                                audience=self.settings.oidc_client_id,
                                issuer=self.settings.oidc_issuer,
                                options={"require": ["exp", "iat", "sub"]})
            if not hmac.compare_digest(str(claims.get("nonce", "")), nonce):
                raise AuthError("身份令牌 nonce 不匹配")
            return {"subject": claims["sub"], "buyer_id": claims.get("buyer_id"),
                    "role": claims.get("role", "buyer")}
        except AuthError:
            raise
        except Exception as exc:
            raise AuthError("身份提供方验证失败") from exc


class AuthService:
    """管理项目 B 自有认证会话；所有买家身份均来自验证后的提供方或显式演示模式。"""

    def __init__(self, store, settings, provider=None):
        self.store, self.settings = store, settings
        self.provider = provider if provider is not None else (
            OidcProvider(settings) if settings.auth_mode == "production" else None)

    @staticmethod
    def digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    async def begin_login(self) -> str:
        """生成短时登录事务；只有一次有效回调能消耗 state。"""
        if self.provider is None:
            raise AuthError("演示模式没有外部登录")
        state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
        self.store.create_auth_flow(self.digest(state), nonce, verifier, time.time() + 300)
        return await self.provider.authorization_url(state, nonce, challenge)

    async def finish_login(self, state: str, code: str, repository) -> tuple[str, str, Principal]:
        """验证一次性回调和买家映射，签发不可从客户端推导的认证 Cookie。"""
        if self.provider is None or not state or not code:
            raise AuthError("登录回调无效")
        flow = self.store.consume_auth_flow(self.digest(state), time.time())
        if not flow:
            raise AuthError("登录事务已过期或使用")
        identity = await self.provider.exchange_code(code, flow["verifier"], flow["nonce"])
        # 令牌只证明外部主体；买家归属必须事先由项目 B 可信管理流程显式绑定。
        binding = self.store.identity_binding(self.settings.oidc_issuer, str(identity.get("subject", "")))
        buyer_id, role = (binding["buyer_id"], binding["role"]) if binding else (None, None)
        if (not identity.get("subject") or not isinstance(buyer_id, str)
                or not repository.has_buyer(buyer_id) or role not in {"buyer", "admin"}):
            raise AuthError("身份没有项目 B 的买家映射")
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        session_hash = self.digest(token)
        self.store.create_auth_session(session_hash, buyer_id, identity["subject"], role,
                                       self.digest(csrf), time.time() + self.settings.auth_session_seconds)
        return token, csrf, Principal(buyer_id, role, session_hash)

    def principal(self, request) -> Principal:
        """每个 HTTP 请求重新从服务端会话读取身份，退出和过期立即生效。"""
        if self.settings.auth_mode == "demo":
            return Principal(self.settings.demo_buyer_id, "admin")
        token = request.cookies.get(AUTH_COOKIE, "")
        if not token:
            raise AuthError("未登录")
        session_hash = self.digest(token)
        row = self.store.auth_session(session_hash, time.time())
        if not row:
            raise AuthError("认证会话已失效")
        return Principal(row["buyer_id"], row["role"], session_hash)

    def verify_csrf(self, request, principal: Principal):
        """写请求必须同时持有认证 Cookie 和绑定该会话的 CSRF 值。"""
        if self.settings.auth_mode == "demo":
            return
        cookie = request.cookies.get(CSRF_COOKIE, "")
        header = request.headers.get("X-CSRF-Token", "")
        row = self.store.auth_session(principal.session_hash, time.time())
        if not cookie or not header or not row or not hmac.compare_digest(cookie, header):
            raise AuthError("CSRF 校验失败")
        if not hmac.compare_digest(self.digest(header), row["csrf_hash"]):
            raise AuthError("CSRF 校验失败")

    def logout(self, principal: Principal):
        """撤销当前认证会话；保留业务工单和 Trace 的持久状态。"""
        if principal.session_hash:
            self.store.revoke_auth_session(principal.session_hash)

    async def end_session_url(self) -> str | None:
        """本地会话撤销后供浏览器结束 Keycloak SSO；无提供方时仅保留本地退出。"""
        if self.provider is None:
            return None
        return await self.provider.end_session_url()
