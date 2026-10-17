"""只读归档原8001浏览器四轮补验，不调用模型、业务工具或写入业务数据库。"""

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
from uuid import UUID, uuid4


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation.agent_runtime.s6_cases import judge


CASES = {
    'combined': ('affected', ['order', 'incident', 'logistics']),
    'inherit': ('unaffected', ['order', 'incident', 'logistics']),
    'switch': ('normal', ['order']),
    'switched_followup': ('no_stock', ['order']),
}


def collect(runs):
    """仅接受显式run_id；从原库读真实响应与消息，完整性和跨轮目标单独判定。"""
    manifest = json.loads((ROOT / '.skillflow/s6-fixtures/s6-cases.json').read_text(encoding='utf-8'))
    report = {'scope': 's6_original_browser_four_turn_followup',
              'recorded_at': datetime.now(timezone.utc).isoformat(),
              'source': 'user_browser_actions_then_readonly_original_database',
              'business_data': 'project_b_simulated', 'capacity_test': False,
              'export_reexecutes_business': False, 'records': [], 'issues': [], 'passed': False}
    database = ROOT / '.skillflow/skillflow.sqlite3'
    # 显式关闭宿主机连接，避免 Windows 文件句柄干扰容器内 SQLite WAL。
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        for case, run_id in runs.items():
            row = db.execute('SELECT state_json FROM agent_runs WHERE run_id=?', (run_id,)).fetchone()
            saved = db.execute('SELECT value_json FROM kv WHERE name=?', ('run_' + run_id,)).fetchone()
            if not row or not saved:
                report['issues'].append(case + ':RUN_NOT_TERMINAL_OR_NOT_FOUND')
                continue
            state, body = json.loads(row[0]), json.loads(saved[0])
            result = body['result']
            buyer = result['metadata']['buyer_id']
            kind, requirements = CASES[case]
            oid = next(r['order_id'] for r in manifest['cases'] if r['kind'] == kind and r['buyer_id'] == buyer)
            ok, issues, security = judge({'kind': kind, 'requirements': requirements}, body, buyer, oid)
            calls = [c['call_id'] for message in state['messages'] for c in message.get('tool_calls', [])]
            observed = [m['tool_call_id'] for m in state['messages'] if m['role'] == 'tool']
            paired = (len(calls) == len(set(calls)) == len(observed) == len(set(observed))
                      and set(calls) == set(observed))
            real = result['metadata']['model_mode'] == 'langchain_openai' and result['budget']['model_requests'] > 0
            if not paired:
                issues.append('MESSAGE_PAIRING_INCOMPLETE')
            if not real:
                issues.append('REAL_MODEL_RECORD_REQUIRED')
            report['records'].append({'case': case, 'correct': ok and paired and real,
                'issues': issues, 'security': security, 'all_tool_calls_paired': paired,
                'real_model_called': real,
                'user_messages': [m['content'] for m in state['messages'] if m['role'] == 'user'],
                'response': body})
    by = {r['case']: r['response'] for r in report['records']}
    if len(by) != len(CASES):
        report['issues'].append('FOUR_TURNS_NOT_COMPLETE')
    else:
        ordered = [by[name] for name in CASES]
        if len({b['session_id'] for b in ordered}) != 1:
            report['issues'].append('SAME_SESSION_REQUIRED')
        if len({b['result']['metadata']['buyer_id'] for b in ordered}) != 1:
            report['issues'].append('SAME_BUYER_REQUIRED')
        if len({b['result']['metadata']['task_id'] for b in ordered}) != 4:
            report['issues'].append('FOUR_FRESH_TASKS_REQUIRED')
        for child, parent in [('inherit', 'combined'), ('switched_followup', 'switch')]:
            if by[child]['result']['task_goal']['continuation_of_task_id'] != by[parent]['result']['metadata']['task_id']:
                report['issues'].append(child + ':WRONG_CONTINUATION_SOURCE')
        for name in ('combined', 'switch'):
            if by[name]['result']['conversation_context']['action'] != 'new':
                report['issues'].append(name + ':EXPLICIT_NEW_GOAL_REQUIRED')
        for child, parent in [('inherit', 'combined'), ('switched_followup', 'switch')]:
            if set(by[child]['result']['result_refs']) & set(by[parent]['result']['result_refs']):
                report['issues'].append(child + ':EVIDENCE_REUSED')
    report['passed'] = not report['issues'] and all(r['correct'] for r in report['records'])
    return report


def main():
    """每次导出另存JSON，部分证据和失败也保留，不覆盖旧人工验收记录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='combined/inherit/switch/switched_followup=run_id')
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        case, run_id = item.split('=', 1)
        if case not in CASES or case in runs:
            parser.error('场景名无效或重复')
        runs[case] = str(UUID(run_id))
    report = collect(runs)
    output = ROOT / 'evaluation/agent_runtime/s6' / (
        'browser-four-turn-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:6] + '.json')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'path': str(output), 'records': len(report['records']),
                      'passed': report['passed'], 'issues': report['issues']}, ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
