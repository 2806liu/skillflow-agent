"""在项目 B 的独立演示服务上评测普通订单 Agent 的真实模型低并发表现。

默认只预览请求；显式传入 --execute 才会调用 /chat。请求均为只读调查，
每条案例使用独立会话，报告只保存业务结论与安全的执行摘要，不保存凭据。
"""

import argparse
import asyncio
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CA = Path.home() / "AppData" / "Local" / "mkcert" / "rootCA.pem"
DEFAULT_REPORT_DIR = ROOT / ".skillflow" / "real-model-acceptance" / "reports"

# 固定五类边界案例，每类四种问法。只问调查结果，不请求创建或确认工单。
PROMPTS = {
    "A123": [
        "查询订单 A123，结合店铺政策，调查能否退货",
        "请核对 A123 的订单和政策，说明退货资格",
        "调查订单 A123 是否符合退换条件，并说明依据",
        "A123 已签收，请查订单、政策和换货库存",
    ],
    "A124": [
        "查询订单 A124，结合店铺政策，调查能否换货",
        "请核对 A124 的换货资格与当前库存",
        "调查订单 A124 是否符合退换条件，并说明缺货影响",
        "A124 已签收，请查政策及是否有同款换货库存",
    ],
    "A125": [
        "查询订单 A125，结合店铺政策，调查能否退货",
        "请核对 A125 的退货资格；没有政策时说明依据不足",
        "调查订单 A125 是否符合退换条件，并列出政策依据",
        "A125 已签收，请查订单和适用的本地售后政策",
    ],
    "A126": [
        "查询订单 A126，结合店铺政策，调查能否退货",
        "请核对 A126 的签收时间和退货期限",
        "调查订单 A126 是否符合退换条件，并说明超期处理",
        "A126 已签收，请依据政策判断能否自动申请退货",
    ],
    "A999": [
        "查询订单 A999",
        "请核对是否存在我的订单 A999",
        "查一下 A999 的订单信息，查不到就直接说明",
        "调查订单 A999 是否可以退货；先确认订单是否存在",
    ],
}


def percentile(values: list[float], percent: int) -> float | None:
    """用最近秩方法计算延迟分位数；空样本返回 None。"""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * percent / 100) - 1)], 2)


def cases(limit: int) -> list[dict]:
    """生成固定顺序的只读案例，确保各并发级别使用相同负载。"""
    return [{"case_id": f"{order_id}-{index + 1}", "order_id": order_id, "message": prompt}
            for order_id, prompts in PROMPTS.items() for index, prompt in enumerate(prompts)][:limit]


def check_business(case: dict, body: dict) -> tuple[bool, list[str]]:
    """核对结构化订单、资格与库存；引用是否真正支持结论仍需人工复核。"""
    order_id = case["order_id"]
    result = body.get("result") or {}
    issues = []
    if result.get("ticket") or result.get("ticket_events"):
        issues.append("READ_ONLY_TICKET_WRITE")
    if order_id == "A999":
        if body.get("status") != "order_not_found":
            issues.append("EXPECTED_ORDER_NOT_FOUND")
        if result.get("get_order", {}).get("order"):
            issues.append("INVENTED_ORDER")
        return not issues, issues

    order = result.get("get_order", {}).get("order") or {}
    if order.get("order_id") != order_id:
        issues.append("WRONG_ORDER")
    expected_status = "insufficient_evidence" if order_id == "A125" else "completed"
    if body.get("status") != expected_status:
        issues.append("WRONG_STATUS")
    assessment = result.get("assess_after_sales") or {}
    if order_id == "A125":
        if assessment:
            issues.append("UNSUPPORTED_ELIGIBILITY")
        if result.get("retrieve_policy", {}).get("evidence"):
            issues.append("UNEXPECTED_POLICY_EVIDENCE")
        return not issues, issues

    if not body.get("citations"):
        issues.append("MISSING_POLICY_CITATION")
    if order_id in {"A123", "A124"} and assessment.get("return_eligible") is not True:
        issues.append("EXPECTED_RETURN_ELIGIBLE")
    if order_id in {"A123", "A124"} and assessment.get("exchange_eligible") is not True:
        issues.append("EXPECTED_EXCHANGE_ELIGIBLE")
    if order_id == "A123" and result.get("check_inventory", {}).get("available") != 5:
        issues.append("WRONG_INVENTORY")
    if order_id == "A124" and result.get("check_inventory", {}).get("available") != 0:
        issues.append("WRONG_INVENTORY")
    if order_id == "A126" and assessment.get("return_eligible") is not False:
        issues.append("EXPECTED_MANUAL_REVIEW")
    if order_id == "A126" and assessment.get("exchange_eligible") is not False:
        issues.append("EXPECTED_MANUAL_REVIEW")
    return not issues, issues


def summarize(rows: list[dict], wall_seconds: float, level: int) -> dict:
    """按完整 HTTP 响应耗时计算指标，区分容量拒绝、模型错误和业务错误。"""
    latencies = [row["latency_ms"] for row in rows if row.get("latency_ms") is not None]
    completed = [row for row in rows if row.get("business_ok")]
    completed_latencies = [row["latency_ms"] for row in completed]
    planned = [row for row in rows if row.get("plan_ok") is not None]
    attempted_tools = sum(row.get("tool_attempts", 0) for row in rows)
    successful_tools = sum(row.get("tool_successes", 0) for row in rows)
    expected_empty_tools = sum(row.get("expected_empty_tools", 0) for row in rows)
    graded_tools = attempted_tools - expected_empty_tools
    return {
        "concurrency": level, "requests": len(rows), "expected_business_results": len(completed),
        "expected_result_rate": round(len(completed) / len(rows), 4) if rows else 0,
        "plan_valid_rate": round(sum(row["plan_ok"] for row in planned) / len(planned), 4) if planned else None,
        "tool_success_rate": round(successful_tools / graded_tools, 4) if graded_tools else None,
        "tool_attempts": attempted_tools, "expected_empty_tools": expected_empty_tools,
        "http_429": sum(row.get("http_status") == 429 for row in rows),
        "http_errors": sum(row.get("http_status", 0) >= 500 for row in rows),
        "transport_errors": sum(bool(row.get("transport_error")) for row in rows),
        "unexpected_ticket_writes": sum("READ_ONLY_TICKET_WRITE" in row.get("issues", []) for row in rows),
        "wall_seconds": round(wall_seconds, 2),
        "throughput_rps": round(len(rows) / wall_seconds, 3) if wall_seconds else None,
        "completed_throughput_rps": round(len(completed) / wall_seconds, 3) if wall_seconds else None,
        "full_response_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95),
                             "p99": percentile(latencies, 99)},
        "completed_response_ms": {"p50": percentile(completed_latencies, 50),
                                  "p95": percentile(completed_latencies, 95),
                                  "p99": percentile(completed_latencies, 99)},
    }


async def run_level(client: httpx.AsyncClient, fixed_cases: list[dict], level: int) -> tuple[list[dict], dict]:
    """用独立会话施加有界并发；每个任务只发一次 /chat，不自动重试。"""
    semaphore = asyncio.Semaphore(level)

    async def one(case: dict) -> dict:
        async with semaphore:
            started = time.perf_counter()
            row = {"case_id": case["case_id"], "order_id": case["order_id"], "message": case["message"]}
            try:
                response = await client.post("/chat", json={"session_id": f"model-eval-{uuid4()}",
                                                             "message": case["message"]})
                row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                row["http_status"] = response.status_code
                if response.status_code != 200:
                    detail = response.json().get("detail", {}) if response.headers.get("content-type", "").startswith("application/json") else {}
                    row["error_code"] = detail.get("code") if isinstance(detail, dict) else f"HTTP_{response.status_code}"
                    row["issues"] = ["HTTP_ERROR"]
                    row["business_ok"] = False
                    return row
                body = response.json()
                result = body.get("result") or {}
                order = result.get("get_order", {}).get("order") or {}
                assessment = result.get("assess_after_sales") or {}
                row.update({"run_id": body.get("run_id"), "response_status": body.get("status"),
                            "model_mode": body.get("model_mode"), "answer": body.get("answer", ""),
                            "result_summary": {
                                "order_id": order.get("order_id"), "product": order.get("product"),
                                "delivered_days_ago": order.get("delivered_days_ago"),
                                "quality_issue": order.get("quality_issue"), "paid_cents": order.get("paid_cents"),
                                "return_eligible": assessment.get("return_eligible"),
                                "exchange_eligible": assessment.get("exchange_eligible"),
                                "inventory_available": result.get("check_inventory", {}).get("available"),
                            },
                            "citations": [{"document_id": item.get("document_id"), "filename": item.get("filename"),
                                           "snippet": item.get("snippet", "")} for item in body.get("citations", [])]})
                events = body.get("events") or []
                row["plan_ok"] = any(item.get("node") == "validate" and item.get("status") == "success" for item in events)
                tool_events = [item for item in events if item.get("node") == "execute" and item.get("tool")
                               and item.get("status") in {"success", "empty", "failed"}]
                row["tool_attempts"] = len(tool_events)
                row["tool_successes"] = sum(item["status"] == "success" for item in tool_events)
                # 缺政策和不存在订单是预期空结果，不应误算成外部工具故障。
                expected_empty = {"A125": "retrieve_policy", "A999": "get_order"}.get(case["order_id"])
                row["expected_empty_tools"] = sum(item["status"] == "empty" and item["tool"] == expected_empty
                                                  for item in tool_events)
                row["tools"] = [{"tool": item["tool"], "status": item["status"]} for item in tool_events]
                row["business_ok"], row["issues"] = check_business(case, body)
                if any(item["status"] == "empty" and item["tool"] != expected_empty for item in tool_events):
                    row["issues"].append("UNEXPECTED_EMPTY_TOOL")
                    row["business_ok"] = False
                if body.get("model_mode") != "langchain_openai":
                    row["issues"].append("WRONG_MODEL_MODE")
                    row["business_ok"] = False
                return row
            except (httpx.RequestError, ValueError, KeyError, TypeError) as error:
                row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                row["transport_error"] = type(error).__name__
                row["issues"] = ["TRANSPORT_OR_RESPONSE_ERROR"]
                row["business_ok"] = False
                return row

    started = time.perf_counter()
    rows = await asyncio.gather(*(one(case) for case in fixed_cases))
    return rows, summarize(rows, time.perf_counter() - started, level)


def main() -> int:
    """检查本机真实模型演示服务，执行分级负载并保存可审阅报告。"""
    parser = argparse.ArgumentParser(description="项目 B 普通订单真实模型低并发验收")
    parser.add_argument("--base-url", default="https://localhost:8003")
    parser.add_argument("--ca-file", type=Path, default=DEFAULT_CA)
    parser.add_argument("--levels", default="1,2,4", help="逗号分隔的并发级别；先测 1，再逐级升高")
    parser.add_argument("--limit", type=int, default=20, help="每级固定案例数，正式验收使用 20")
    parser.add_argument("--timeout", type=float, default=120, help="单次完整响应超时秒数")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--execute", action="store_true", help="实际调用模型；默认仅预览")
    args = parser.parse_args()
    try:
        levels = [int(item) for item in args.levels.split(",")]
    except ValueError:
        parser.error("--levels 必须是逗号分隔的正整数")
    if not levels or any(level < 1 or level > 16 for level in levels) or not 1 <= args.limit <= 20 or args.timeout <= 0:
        parser.error("并发级别须在 1–16，案例数须在 1–20，超时须大于 0")
    parsed = urlsplit(args.base_url)
    if parsed.scheme != "https" or parsed.hostname not in {"localhost", "127.0.0.1"} or parsed.username or parsed.password:
        parser.error("只允许使用本机 HTTPS 演示服务")
    fixed_cases = cases(args.limit)
    print(f"案例数：{len(fixed_cases)}；并发级别：{levels}；预计 /chat 调用：{len(fixed_cases) * len(levels)} 次")
    if not args.execute:
        for case in fixed_cases:
            print(f"{case['case_id']}: {case['message']}")
        print("仅预览，未调用模型；确认后加 --execute。")
        return 0
    if not args.ca_file.is_file():
        parser.error(f"找不到本机 CA 证书：{args.ca_file}")
    # 先确保报告可写，再产生模型调用；中断时已完成的级别也能保留。
    report_dir = args.report_dir.resolve()
    if not report_dir.is_relative_to(ROOT):
        parser.error("报告只能写入项目 B 目录")
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"报告目录不可写：{type(error).__name__}；未调用模型。")
        return 2
    path = report_dir / f"real-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}.json"

    async def execute() -> dict:
        timeout = httpx.Timeout(args.timeout)
        async with httpx.AsyncClient(base_url=args.base_url, verify=str(args.ca_file), trust_env=False,
                                     timeout=timeout, limits=httpx.Limits(max_connections=max(levels))) as client:
            health_response = await client.get("/health")
            health_response.raise_for_status()
            health = health_response.json()
            # 阻止误向生产身份服务、Mock 模型或充电宝确定性路径发负载。
            if not (health.get("auth_mode") == "demo" and health.get("model_mode") == "langchain_openai"
                    and health.get("model_configured") is True and health.get("business_backend") == "mock"):
                raise RuntimeError("目标必须是项目 B 的 demo + langchain_openai + mock 业务库服务")
            report = {"generated_at": datetime.now(timezone.utc).isoformat(), "base_url": args.base_url,
                      "state": "running",
                      "auth_mode": health["auth_mode"], "model_mode": health["model_mode"],
                      "business_backend": health["business_backend"], "cases_per_level": len(fixed_cases),
                      "manual_review_required": ["回答事实是否正确", "引用是否支持各项结论"], "levels": []}
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            for level in levels:
                rows, summary = await run_level(client, fixed_cases, level)
                report["levels"].append({"summary": summary, "cases": rows})
                path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"并发 {level}: 正确 {summary['expected_business_results']}/{summary['requests']}，"
                      f"成功响应 P95 {summary['completed_response_ms']['p95']} ms，"
                      f"完成吞吐 {summary['completed_throughput_rps']} 次/秒，"
                      f"429={summary['http_429']}，传输错误={summary['transport_errors']}")
            report["state"] = "complete"
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return report

    try:
        report = asyncio.run(execute())
    except (httpx.HTTPError, RuntimeError, ValueError, OSError) as error:
        print(f"验收未开始或中断：{type(error).__name__}；请检查 8003 的健康状态、证书和模型服务。")
        return 2
    print(f"报告：{path}")
    print("业务与计划可先看自动统计；事实、引用正确率须人工复核，性能门槛须在基线后固定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
