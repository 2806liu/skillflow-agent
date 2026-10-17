"""S1 可重复验收入口：隔离测试配置，运行脚本模型并输出本项目内 JSON 报告。"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import unittest


ROOT = Path(__file__).resolve().parent.parent


def main():
    """仅覆盖当前测试进程的环境，不修改 .env、不启动服务或调用真实 LLM。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-regression", action="store_true", help="同时回归旧聊天、SSE 和确定性事件路径")
    parser.add_argument("--report", type=Path, default=ROOT / "evaluation/agent_runtime/s1/latest.json")
    args = parser.parse_args()
    report_path = args.report.resolve()
    if not report_path.is_relative_to(ROOT):
        parser.error("验收报告必须保存在项目 B 内")

    # 必须早于业务模块导入，避免本机生产 OIDC 或旧真实模型配置污染离线回归。
    os.environ.update({
        "SKILLFLOW_AUTH_MODE": "demo", "SKILLFLOW_MODEL_MODE": "mock",
        "SKILLFLOW_AGENT_ENGINE": "legacy", "SKILLFLOW_AGENT_MODEL_MODE": "disabled",
        "SKILLFLOW_DEMO_BUYER_ID": "demo-buyer", "SKILLFLOW_BUSINESS_BACKEND": "mock",
        "SKILLFLOW_CHAT_LOCK_MODE": "session", "SKILLFLOW_MAX_ACTIVE_CHATS": "8",
        "SKILLFLOW_MAX_WAITING_CHATS": "64", "SKILLFLOW_AGENT_MAX_ROUNDS": "8",
        "SKILLFLOW_AGENT_MAX_TOOL_ATTEMPTS": "12", "SKILLFLOW_AGENT_MAX_MODEL_REQUESTS": "10",
        "SKILLFLOW_AGENT_MODEL_TIMEOUT": "30", "SKILLFLOW_AGENT_ACTIVE_TIMEOUT": "120",
    })
    sys.path.insert(0, str(ROOT))
    modules = ["tests_skillflow.test_agent_foundation"]
    if args.include_regression:
        modules.extend(["tests_skillflow.test_agent", "tests_skillflow.test_streaming", "tests_skillflow.test_surge"])
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    started = perf_counter()
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    report = {
        "stage": "S1", "recorded_at": datetime.now(timezone.utc).isoformat(),
        "execution_mode": "simulation", "real_model_called": False,
        "scope": modules, "tests_run": result.testsRun,
        "failures": [test.id() for test, _ in result.failures],
        "errors": [test.id() for test, _ in result.errors], "skipped": len(result.skipped),
        "successful": result.wasSuccessful(), "elapsed_seconds": round(perf_counter() - started, 3),
        "dependencies": {name: version(name) for name in ("langgraph", "langchain-core", "pydantic")},
        "limitations": ["不证明真实模型 Tool Calling", "未实现 S4 图检查点恢复", "不是容量或延迟性能验收"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("S1_REPORT=" + str(report_path))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
