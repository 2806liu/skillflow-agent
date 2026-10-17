"""S6 自动联合验收；真实模型与Keycloak网页、性能报告另存，不能混算成功率。"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import unittest


def main():
    """复用前序固定测试环境，增加生产身份与组合工具测试；不写原部署库。"""
    root=Path(__file__).resolve().parent.parent
    sys.path.insert(0,str(root))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--s6-only',action='store_true',help='只运行S6身份联合与评测器测试，单独存报告，不覆盖全量回归')
    args=parser.parse_args()
    # 设置全局测试默认，S6单项再显式启用生产模式，避免原部署变量泄漏到回归。
    os.environ.update({'SKILLFLOW_AUTH_MODE':'demo','SKILLFLOW_AGENT_S6_ENABLED':'false',
        # api模块的默认app也只能初始化测试目录，不能触碰原部署库的表结构。
        'SKILLFLOW_DATA_DIR':str(root/'.skillflow/agent-s6-runner'),
        'SKILLFLOW_MODEL_MODE':'mock','SKILLFLOW_AGENT_ENGINE':'legacy','SKILLFLOW_AGENT_MODEL_MODE':'disabled',
        'SKILLFLOW_AGENT_S4_ENABLED':'false','SKILLFLOW_AGENT_S5_ENABLED':'false',
        'SKILLFLOW_BUSINESS_BACKEND':'mock','SKILLFLOW_DEMO_BUYER_ID':'demo-buyer',
        'SKILLFLOW_FIXTURES_DIR':str(root/'data'),'SKILLFLOW_AGENT_LOGISTICS_FILE':str(root/'data/logistics/delivery.json')})
    modules=['tests_skillflow.test_agent_s6','tests_skillflow.test_agent_s6_evaluation',
        'tests_skillflow.test_agent_retry_progress','tests_skillflow.test_auth','tests_skillflow.test_customer_response',
        'tests_skillflow.test_agent_mcp','tests_skillflow.test_agent_resume','tests_skillflow.test_agent_loop',
        'tests_skillflow.test_agent_conversation','tests_skillflow.test_agent_skills',
        'tests_skillflow.test_agent_foundation','tests_skillflow.test_agent','tests_skillflow.test_streaming',
        'tests_skillflow.test_surge']
    if args.s6_only: modules=modules[:3]
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in modules)
    start=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    report={'stage':'S6','scope':'s6_only' if args.s6_only else 'full_regression',
        'recorded_at':datetime.now(timezone.utc).isoformat(),'tests_run':result.testsRun,
        'passed':result.wasSuccessful(),'real_model_called':False,'real_keycloak_called':False,
        'elapsed_seconds':round(time.perf_counter()-start,3),
        'failures':[t.id() for t,_ in result.failures],'errors':[t.id() for t,_ in result.errors]}
    out=root/'evaluation/agent_runtime/s6';out.mkdir(parents=True,exist_ok=True)
    # 每轮保留原始结果，latest仅为便捷入口，不能覆盖失败历史当作首次成功。
    prefix='followup' if args.s6_only else 'regression'
    (out/(prefix+'-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.json')).write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    latest='followup-latest' if args.s6_only else 'latest'
    (out/(latest+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/(latest+'.md')).write_text('# S6 自动联合回归\n\n身份提供方与模型为替身，业务库、MCP协议和图Saver实际执行。\n\n'
        +f"测试 {result.testsRun} 项；全部通过 {result.wasSuccessful()}；耗时 {report['elapsed_seconds']} 秒。\n",encoding='utf-8')
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(main())
