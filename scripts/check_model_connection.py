"""检查项目 B 的模型接口连通性，不接触订单、政策索引或工单。

默认依次检查域名解析、直连 TCP/TLS、HTTP 接口；可选的 --model 和
--structured 才会发送小型模型请求。所有配置仅来自项目 B 的 .env，
输出会遮盖密钥，失败时按网络、认证、模型和结构化输出分层报告。
"""

import argparse
import asyncio
import os
import socket
import ssl
import sys
from pathlib import Path
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skillflow.config import Settings  # noqa: E402
from skillflow.planner import Planner  # noqa: E402
from skillflow.skills.registry import load_skills  # noqa: E402


def safe_error(error: BaseException, key: str) -> str:
    """只输出异常类别及简短原因，避免服务商错误文本意外回显密钥。"""
    parts = []
    current = error
    for _ in range(3):
        if current is None:
            break
        message = str(current).replace(key, "***") if key else str(current)
        parts.append(f"{type(current).__name__}: {message[:160]}")
        current = current.__cause__
    return " → ".join(parts)


def direct_checks(host: str, port: int, timeout: float) -> None:
    """检查本机直连路径；HTTP 客户端仍会另行测试系统代理路径。"""
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        print(f"DNS：成功（{len(addresses)} 个地址）")
    except OSError as error:
        print(f"DNS：失败（{type(error).__name__}）")
        return
    try:
        with socket.create_connection((host, port), timeout=timeout) as connection:
            print("TCP：成功")
            if port == 443:
                # 只做握手，不发送密钥；证书错误属于 TLS 层问题。
                with ssl.create_default_context().wrap_socket(connection, server_hostname=host):
                    print("TLS：成功")
    except (OSError, ssl.SSLError) as error:
        print(f"TCP/TLS：失败（{type(error).__name__}）")


def http_check(settings: Settings, trust_env: bool) -> bool:
    """分别测试继承系统代理和直连路径，定位代理设置导致的连接失败。"""
    endpoint = settings.base_url.rstrip("/") + "/models"
    label = "系统代理" if trust_env else "直连"
    try:
        with httpx.Client(timeout=min(settings.model_timeout, 10), follow_redirects=False,
                          trust_env=trust_env) as client:
            response = client.get(endpoint, headers={"Authorization": f"Bearer {settings.api_key}"})
        print(f"HTTP（{label}）：已连接，GET /models 返回 {response.status_code}")
        if response.status_code in {401, 403}:
            print("提示：接口可达，但密钥无效或无权访问模型列表。")
        elif response.status_code == 404:
            print("提示：接口可达，但代理可能没有实现 /models；可运行 --model 验证聊天接口。")
        return True
    except httpx.RequestError as error:
        print(f"HTTP（{label}）：连接失败；" + safe_error(error, settings.api_key))
        return False


def model_check(settings: Settings) -> bool:
    """仅在显式指定 --model 时发送最小聊天请求，区分连接与模型权限。"""
    endpoint = settings.base_url.rstrip("/") + "/chat/completions"
    payload = {"model": settings.model, "messages": [{"role": "user", "content": "请只回复 OK"}],
               "max_completion_tokens": 16}
    try:
        with httpx.Client(timeout=settings.model_timeout, follow_redirects=False,
                          trust_env=settings.model_trust_env) as client:
            response = client.post(endpoint, headers={"Authorization": f"Bearer {settings.api_key}"}, json=payload)
        if response.is_success:
            print(f"模型请求：成功（HTTP {response.status_code}）")
            return True
        try:
            detail = response.json().get("error", {})
            code = detail.get("code") or detail.get("type") or "未提供错误代码"
        except (ValueError, AttributeError):
            code = "响应不是标准 JSON 错误"
        print(f"模型请求：失败（HTTP {response.status_code}，错误代码：{code}）")
        return False
    except httpx.RequestError as error:
        print("模型请求：连接失败；" + safe_error(error, settings.api_key))
        return False


async def structured_check(settings: Settings) -> bool:
    """调用项目真实 DialogueDecision 路径，验证代理的工具调用结构化输出。"""
    try:
        decision = await Planner(settings, load_skills()).decide("查询订单 A123", {})
        print(f"结构化输出：成功（intent={decision.intent}）")
        return True
    except Exception as error:
        print("结构化输出：失败；" + safe_error(error, settings.api_key))
        return False


def main() -> int:
    """按网络、模型、结构化输出顺序检查；失败时返回非零退出码。"""
    parser = argparse.ArgumentParser(description="检查项目 B 模型接口，不打印 API 密钥")
    parser.add_argument("--model", action="store_true", help="发送一次最小模型请求，可能产生少量费用")
    parser.add_argument("--structured", action="store_true", help="测试项目实际使用的 DialogueDecision 结构化输出")
    args = parser.parse_args()
    settings = Settings()
    if settings.model_mode != "langchain_openai" or not settings.base_url or not settings.model or not settings.api_key:
        print("配置不完整：请检查项目 B 的 SKILLFLOW_MODEL_MODE、MODEL、BASE_URL、API_KEY。")
        return 4
    parsed = urlsplit(settings.base_url)
    if parsed.scheme != "https" or not parsed.hostname:
        print("配置错误：SKILLFLOW_BASE_URL 应为 HTTPS 接口地址。")
        return 4
    print(f"配置：model={settings.model}，base_url={settings.base_url}，key=已设置")
    proxy_names = [name for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
                   if os.getenv(name)]
    print(f"代理环境变量：{', '.join(proxy_names) if proxy_names else '未设置'}；项目 B 使用系统代理：{settings.model_trust_env}")
    direct_checks(parsed.hostname, parsed.port or 443, min(settings.model_timeout, 5))
    # 两条 HTTP 路径均独立测试，只有项目配置选定的路径决定退出状态。
    proxy_ok = http_check(settings, True)
    direct_ok = http_check(settings, False)
    if not (proxy_ok if settings.model_trust_env else direct_ok):
        return 2
    if (args.model or args.structured) and not model_check(settings):
        return 3
    if args.structured and not asyncio.run(structured_check(settings)):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
