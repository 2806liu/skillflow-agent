"""S5 隔离验收：真实 MCP 协议、模型替身、实际检查点；可包含 S1—S4 回归。"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
import unittest


def main():
    """固定测试进程配置，报告单独归档，不覆盖前序阶段历史报告。"""
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-regression', action='store_true')
    args = parser.parse_args()
    os.environ.update({'SKILLFLOW_AUTH_MODE': 'demo', 'SKILLFLOW_MODEL_MODE': 'mock',
        'SKILLFLOW_AGENT_ENGINE': 'legacy', 'SKILLFLOW_AGENT_MODEL_MODE': 'disabled',
        'SKILLFLOW_AGENT_S4_ENABLED': 'false', 'SKILLFLOW_AGENT_S5_ENABLED': 'false',
        'SKILLFLOW_AGENT_CONFIRMATION_SECONDS': '180', 'SKILLFLOW_AGENT_MCP_QUERY_TIMEOUT': '10',
        'SKILLFLOW_AGENT_MCP_STARTUP_TIMEOUT': '10', 'SKILLFLOW_DEMO_BUYER_ID': 'demo-buyer',
        'SKILLFLOW_BUSINESS_BACKEND': 'mock', 'SKILLFLOW_CHAT_LOCK_MODE': 'session',
        'SKILLFLOW_MAX_ACTIVE_CHATS': '8', 'SKILLFLOW_MAX_WAITING_CHATS': '64',
        'SKILLFLOW_AGENT_MAX_ROUNDS': '8', 'SKILLFLOW_AGENT_MAX_TOOL_ATTEMPTS': '12',
        'SKILLFLOW_AGENT_MAX_MODEL_REQUESTS': '10', 'SKILLFLOW_AGENT_MODEL_TIMEOUT': '30',
        'SKILLFLOW_AGENT_ACTIVE_TIMEOUT': '120', 'SKILLFLOW_AGENT_CONTEXT_TTL_SECONDS': '1800'})
    modules = ['tests_skillflow.test_agent_mcp']
    if args.include_regression:
        modules += ['tests_skillflow.test_agent_resume', 'tests_skillflow.test_agent_loop',
            'tests_skillflow.test_agent_conversation', 'tests_skillflow.test_agent_skills',
            'tests_skillflow.test_agent_foundation', 'tests_skillflow.test_agent',
            'tests_skillflow.test_streaming', 'tests_skillflow.test_surge']
    started = time.perf_counter()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in modules)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    report = {'stage': 'S5', 'recorded_at': datetime.now(timezone.utc).isoformat(),
        'real_model_called': False, 'mcp_transport': 'real_stdio_subprocess',
        'versions': {name: version(name) for name in ['mcp', 'langchain-mcp-adapters', 'langchain-core', 'langgraph']},
        'tests_run': result.testsRun, 'passed': result.wasSuccessful(), 'elapsed_seconds': round(time.perf_counter() - started, 3),
        'failures': [t.id() for t, _ in result.failures], 'errors': [t.id() for t, _ in result.errors], 'modules': modules}
    output = root / 'evaluation/agent_runtime/s5'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'latest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'latest.md').write_text('# S5 自动验收\n\n实际 stdio MCP 子进程与 Adapter，模型为替身；不是容量测试。\n\n'
        + f"测试数：{result.testsRun}；全部通过：{result.wasSuccessful()}；套件耗时 {report['elapsed_seconds']} 秒。\n", encoding='utf-8')
    print('S5_REPORT=' + str(output / 'latest.md'))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
