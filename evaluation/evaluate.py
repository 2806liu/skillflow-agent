"""在项目 B 的演示数据上复测固定案例，输出可核对的业务与引用指标。"""

import json
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi.testclient import TestClient

from skillflow.api import create_app
from skillflow.config import Settings


ROOT = Path(__file__).resolve().parent.parent
CASES = Path(__file__).with_name("cases.json")
REPORT = Path(__file__).with_name("results.json")


def evaluate_case(client, case):
    """校验任务终态、业务字段和原文引用，避免只统计 HTTP 200。"""
    started = perf_counter()
    response = client.post("/chat", json={"session_id": str(uuid4()), "message": case["message"]})
    latency_ms = round((perf_counter() - started) * 1000, 2)
    response.raise_for_status()
    actual = response.json()
    errors = []
    if actual["status"] != case["status"]:
        errors.append(f"status: {actual['status']} != {case['status']}")
    execution = [event for event in actual["events"] if event["node"] == "execute" and event["status"] != "skipped"]
    if case["status"] == "completed":
        assessment = actual["result"].get("assess_after_sales", {})
        inventory = actual["result"].get("check_inventory", {})
        if assessment.get("return_eligible") != case["return_eligible"]:
            errors.append("退货资格与预期不符")
        if inventory.get("available") != case["inventory"]:
            errors.append("库存与预期不符")
        if not actual["citations"] or not assessment.get("citations_used"):
            errors.append("缺少政策引用")
        for citation in actual["citations"]:
            source = client.get(f"/policies/{citation['document_id']}/source")
            # 引用必须能回到本项目政策原文，且引用片段确实存在。
            if source.status_code != 200 or citation["snippet"] not in source.text:
                errors.append("引用无法回溯原文")
            if f"[{citation['citation_number']}]" not in actual["answer"]:
                errors.append("答案未标记引用")
    elif case["status"] == "insufficient_evidence":
        # 缺政策不得继续做资格判定。
        if "assess_after_sales" in actual["result"]:
            errors.append("无政策仍输出资格判断")
    elif case["status"] in {"order_not_found", "clarify", "unsupported"}:
        if actual["citations"] or "assess_after_sales" in actual["result"]:
            errors.append("非完成路径输出了政策结论")
    return {"id": case["id"], "expected_status": case["status"], "actual_status": actual["status"],
            "passed": not errors, "errors": errors, "latency_ms": latency_ms,
            "citation_count": len(actual["citations"]),
            "tool_attempts": len(execution),
            "tool_failed": sum(event["status"] == "failed" for event in execution),
            "tool_empty": sum(event["status"] == "empty" for event in execution)}


def main():
    """固定 Mock 模式与 B 自有目录；热身请求不进入指标分母。"""
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    settings = Settings(model_mode="mock", fixtures_dir=ROOT / "data", data_dir=ROOT / ".skillflow" / "evaluation")
    client = TestClient(create_app(settings=settings))
    for _ in range(2):
        client.post("/chat", json={"session_id": str(uuid4()), "message": "查询订单 A123"}).raise_for_status()
    results = [evaluate_case(client, case) for case in cases]
    completed = [item for item in results if item["expected_status"] == "completed"]
    sorted_latency = sorted(item["latency_ms"] for item in results)
    # 使用最近秩定义 P95：第 ceil(0.95*N) 个耗时，不插值。
    p95 = sorted_latency[ceil(0.95 * len(sorted_latency)) - 1]
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "environment": "single-process FastAPI TestClient; Mock Planner; JSON orders; local Markdown/BM25 policy",
              "method": "2 warm-up requests excluded; 22 sequential independent sessions; nearest-rank P95",
              "metrics": {"cases": len(results), "passed": sum(item["passed"] for item in results),
                          "completed_decisions": len(completed),
                          "citation_verified_decisions": sum(item["passed"] for item in completed),
                          "tool_attempts": sum(item["tool_attempts"] for item in results),
                          "tool_failed": sum(item["tool_failed"] for item in results),
                          "tool_empty": sum(item["tool_empty"] for item in results),
                          "p95_latency_ms": p95},
              "cases": results}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    if report["metrics"]["passed"] != len(results):
        raise SystemExit("固定案例有失败，详见 evaluation/results.json")


if __name__ == "__main__":
    main()
