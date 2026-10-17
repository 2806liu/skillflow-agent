"""客服表达与聊天等待反馈改进的回归入口；与 S5 历史报告分开归档。"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import unittest


def main():
    """固定隔离测试配置；真实模型网页验收另存，不以自动测试冒充 UI 观察。"""
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    os.environ.update({'SKILLFLOW_AUTH_MODE':'demo', 'SKILLFLOW_MODEL_MODE':'mock',
        'SKILLFLOW_AGENT_ENGINE':'legacy', 'SKILLFLOW_AGENT_MODEL_MODE':'disabled',
        'SKILLFLOW_AGENT_S4_ENABLED':'false', 'SKILLFLOW_AGENT_S5_ENABLED':'false',
        'SKILLFLOW_AGENT_CONFIRMATION_SECONDS':'180', 'SKILLFLOW_AGENT_CONTEXT_TTL_SECONDS':'1800',
        'SKILLFLOW_AGENT_MCP_QUERY_TIMEOUT':'10', 'SKILLFLOW_AGENT_MCP_STARTUP_TIMEOUT':'10',
        'SKILLFLOW_DEMO_BUYER_ID':'demo-buyer', 'SKILLFLOW_BUSINESS_BACKEND':'mock',
        'SKILLFLOW_CHAT_LOCK_MODE':'session', 'SKILLFLOW_MAX_ACTIVE_CHATS':'8', 'SKILLFLOW_MAX_WAITING_CHATS':'64',
        'SKILLFLOW_AGENT_MAX_ROUNDS':'8', 'SKILLFLOW_AGENT_MAX_TOOL_ATTEMPTS':'12',
        'SKILLFLOW_AGENT_MAX_MODEL_REQUESTS':'10', 'SKILLFLOW_AGENT_MODEL_TIMEOUT':'30',
        'SKILLFLOW_AGENT_ACTIVE_TIMEOUT':'120'})
    modules = ['tests_skillflow.test_customer_response', 'tests_skillflow.test_agent_mcp',
        'tests_skillflow.test_agent_resume', 'tests_skillflow.test_agent_loop',
        'tests_skillflow.test_agent_conversation', 'tests_skillflow.test_agent_skills',
        'tests_skillflow.test_agent_foundation', 'tests_skillflow.test_agent',
        'tests_skillflow.test_streaming', 'tests_skillflow.test_surge']
    started = time.perf_counter()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in modules)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    report = {'stage':'customer_experience', 'recorded_at':datetime.now(timezone.utc).isoformat(),
        'tests_run':result.testsRun, 'passed':result.wasSuccessful(), 'real_model_called':False,
        'elapsed_seconds':round(time.perf_counter()-started,3),
        'failures':[t.id() for t,_ in result.failures], 'errors':[t.id() for t,_ in result.errors], 'modules':modules}
    output = root / 'evaluation/agent_runtime/customer-experience'
    output.mkdir(parents=True, exist_ok=True)
    (output/'latest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'latest.md').write_text('# 客服表达与等待反馈：自动回归\n\n'
        '模型为替身；MCP、业务库和 Saver 实际执行。聊天等待状态通过网页另行核验。\n\n'
        +f"测试数 {result.testsRun}；全部通过 {result.wasSuccessful()}；整轮 {report['elapsed_seconds']} 秒。\n", encoding='utf-8')
    print('CUSTOMER_EXPERIENCE_REPORT='+str(output/'latest.md'))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
