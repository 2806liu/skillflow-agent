"""S4 隔离验收：模型替身＋真实执行图 Saver＋实际业务事务，不请求外部模型。"""

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time
import unittest


def main():
    """固定当前进程环境；报告与 S3 并列保存，不覆盖历史真实模型证据。"""
    root=Path(__file__).resolve().parent.parent
    sys.path.insert(0,str(root))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-regression',action='store_true')
    args=parser.parse_args()
    os.environ.update({'SKILLFLOW_AUTH_MODE':'demo','SKILLFLOW_MODEL_MODE':'mock','SKILLFLOW_AGENT_ENGINE':'legacy',
        'SKILLFLOW_AGENT_MODEL_MODE':'disabled','SKILLFLOW_AGENT_S4_ENABLED':'false','SKILLFLOW_AGENT_CONFIRMATION_SECONDS':'180',
        'SKILLFLOW_DEMO_BUYER_ID':'demo-buyer','SKILLFLOW_BUSINESS_BACKEND':'mock','SKILLFLOW_CHAT_LOCK_MODE':'session',
        'SKILLFLOW_MAX_ACTIVE_CHATS':'8','SKILLFLOW_MAX_WAITING_CHATS':'64','SKILLFLOW_AGENT_MAX_ROUNDS':'8',
        'SKILLFLOW_AGENT_MAX_TOOL_ATTEMPTS':'12','SKILLFLOW_AGENT_MAX_MODEL_REQUESTS':'10',
        'SKILLFLOW_AGENT_MODEL_TIMEOUT':'30','SKILLFLOW_AGENT_ACTIVE_TIMEOUT':'120','SKILLFLOW_AGENT_CONTEXT_TTL_SECONDS':'1800'})
    modules=['tests_skillflow.test_agent_resume']
    if args.include_regression:
        modules+=['tests_skillflow.test_agent_loop','tests_skillflow.test_agent_conversation','tests_skillflow.test_agent_skills',
            'tests_skillflow.test_agent_foundation','tests_skillflow.test_agent','tests_skillflow.test_streaming','tests_skillflow.test_surge']
    started=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    report={'stage':'S4','recorded_at':datetime.now(timezone.utc).isoformat(),'real_model_called':False,
        'tests_run':result.testsRun,'passed':result.wasSuccessful(),'elapsed_seconds':round(time.perf_counter()-started,3),
        'scope':modules,'failures':[t.id() for t,_ in result.failures],'errors':[t.id() for t,_ in result.errors]}
    output=root/'evaluation/agent_runtime/s4';output.mkdir(parents=True,exist_ok=True)
    (output/'latest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'latest.md').write_text('# S4 自动验收\n\n模型替身、实际 LangGraph 检查点和 B 本地工单事务；不是模型容量测试。\n\n'
        +f"测试数：{result.testsRun}；全部通过：{result.wasSuccessful()}；套件耗时 {report['elapsed_seconds']} 秒。\n",encoding='utf-8')
    print('S4_REPORT='+str(output/'latest.md'))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(main())
