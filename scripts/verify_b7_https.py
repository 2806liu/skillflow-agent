"""只读检查项目 B 本机 HTTPS 和监控响应头；双买家数据由浏览器验收。"""

import argparse
from pathlib import Path
import re

import httpx


def main():
    """读取受信任的本机 CA，检查 Keycloak 发行者、SkillFlow 健康与未登录边界。"""
    parser = argparse.ArgumentParser(description="项目 B 本机 HTTPS 只读验收")
    parser.add_argument("--skillflow", default="https://localhost:8001")
    parser.add_argument("--issuer", default="https://localhost/realms/myrealm")
    parser.add_argument("--ca", type=Path, default=Path.home() / "AppData/Local/mkcert/rootCA.pem")
    args = parser.parse_args()
    if not args.ca.is_file():
        raise SystemExit("FAIL: 找不到本机 mkcert 根证书，请用 --ca 指定 rootCA.pem。")

    # 不继承系统代理，防止 localhost 请求被代理转发；证书必须由指定 CA 验证。
    with httpx.Client(verify=str(args.ca), trust_env=False, timeout=10, follow_redirects=False) as client:
        discovery = client.get(args.issuer.rstrip("/") + "/.well-known/openid-configuration")
        discovery.raise_for_status()
        metadata = discovery.json()
        if metadata.get("issuer") != args.issuer.rstrip("/"):
            raise SystemExit("FAIL: Keycloak issuer 与预期不一致。")
        # 浏览器重定向与服务端换令牌均必须继续使用 HTTPS，不能由发现文档降级。
        for endpoint in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
            if not metadata.get(endpoint, "").startswith("https://"):
                raise SystemExit(f"FAIL: Keycloak {endpoint} 不是 HTTPS。")

        health = client.get(args.skillflow.rstrip("/") + "/health")
        health.raise_for_status()
        if health.json() != {"status": "ok", "auth_mode": "production"}:
            raise SystemExit("FAIL: SkillFlow 未以 production 认证模式运行。")
        request_id = health.headers.get("X-Request-ID", "")
        if not re.fullmatch(r"[0-9a-f]{32}", request_id):
            raise SystemExit("FAIL: 当前 8001 服务尚未加载 B7 请求监控代码。")

        unauthenticated = client.get(args.skillflow.rstrip("/") + "/auth/me")
        if unauthenticated.status_code != 401 or not unauthenticated.headers.get("X-Request-ID"):
            raise SystemExit("FAIL: 未登录访问应返回 401，且应带请求 ID。")
    print("PASS: Keycloak HTTPS issuer、SkillFlow production 健康检查、B7 请求 ID、未登录 401。")
    print("下一步请在 Alice/Bob 浏览器窗口核对订单、工单和执行记录；本脚本不读取登录 Cookie。")


if __name__ == "__main__":
    main()
