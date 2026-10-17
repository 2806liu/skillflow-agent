"""S3 隔离自动验收：默认全部使用模型替身，不请求外部 LLM；网页实测另存记录。"""

from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import sys
import unittest


def main():
    """固定测试环境；报告只写 B，绝不把替身测试标为真实模型性能证据。"""
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-regression", action="store_true")
    args = parser.parse_args()
    os.environ.update({"SKILLFLOW_AUTH_MODE": "demo", "SKILLFLOW_MODEL_MODE": "mock",
        "SKILLFLOW_AGENT_ENGINE": "legacy", "SKILLFLOW_AGENT_MODEL_MODE": "disabled",
        "SKILLFLOW_DEMO_BUYER_ID": "demo-buyer", "SKILLFLOW_BUSINESS_BACKEND": "mock",
        "SKILLFLOW_CHAT_LOCK_MODE": "session", "SKILLFLOW_MAX_ACTIVE_CHATS": "8", "SKILLFLOW_MAX_WAITING_CHATS": "64",
        "SKILLFLOW_AGENT_MAX_ROUNDS": "8", "SKILLFLOW_AGENT_MAX_TOOL_ATTEMPTS": "12",
        "SKILLFLOW_AGENT_MAX_MODEL_REQUESTS": "10", "SKILLFLOW_AGENT_MODEL_TIMEOUT": "30", "SKILLFLOW_AGENT_ACTIVE_TIMEOUT": "120",
        "SKILLFLOW_AGENT_CONTEXT_TTL_SECONDS": "1800"})
    # S3 短期目标衔接作为新图能力纳入同一回归入口，避免分离后遗漏连续换单用例。
    modules = ["tests_skillflow.test_agent_loop", "tests_skillflow.test_agent_conversation"]
    if args.include_regression:
        modules += ["tests_skillflow.test_agent_skills", "tests_skillflow.test_agent_foundation", "tests_skillflow.test_agent",
                    "tests_skillflow.test_streaming", "tests_skillflow.test_surge"]
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    report = {"stage": "S3", "recorded_at": datetime.now(timezone.utc).isoformat(), "real_model_called": False,
              "tests_run": result.testsRun, "passed": result.wasSuccessful(), "scope": modules,
              "failures": [t.id() for t, _ in result.failures], "errors": [t.id() for t, _ in result.errors]}
    output = root / "evaluation/agent_runtime/s3"
    output.mkdir(parents=True, exist_ok=True)
    (output / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "latest.md").write_text("# S3 自动验收\n\n模型替身＋实际本地工具；真实模型网页实测见 browser-acceptance.md，短期上下文补验见 context-browser-acceptance.md。\n\n"
        + f"测试数：{result.testsRun}；全部通过：{result.wasSuccessful()}。\n", encoding="utf-8")
    print("S3_REPORT=" + str(output / "latest.md"))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
