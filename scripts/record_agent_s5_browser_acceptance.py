"""只读导出已经在 S5 网页核对的运行；不新发模型、MCP 或业务写入。"""

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


def main():
    """从独立 B 演示库导出真实消息及配对证据，不根据预期编造结果。"""
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='case=run_id')
    args = parser.parse_args()
    database = root / '.skillflow/agent-s5-demo/skillflow.sqlite3'
    rows = []
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        for item in args.run:
            case, run_id = item.split('=', 1)
            response = json.loads(db.execute('SELECT value_json FROM kv WHERE name=?', ('run_' + run_id,)).fetchone()[0])
            if response['result']['stage'] != 'S5':
                raise ValueError('S5_RECORD_REQUIRED')
            state = json.loads(db.execute('SELECT state_json FROM agent_runs WHERE run_id=?', (run_id,)).fetchone()[0])
            calls = [c['call_id'] for m in state['messages'] for c in m.get('tool_calls', [])]
            observed = [m['tool_call_id'] for m in state['messages'] if m['role'] == 'tool']
            paired = len(calls) == len(set(calls)) == len(observed) == len(set(observed)) and set(calls) == set(observed)
            rows.append({'case': case, 'response': response, 'messages': state['messages'],
                         'all_tool_calls_paired': paired, 'checkpoint_id': response['result']['task']['checkpoint_id']})
        tickets = db.execute('SELECT count(*) FROM tickets').fetchone()[0]
    report = {'stage': 'S5', 'recorded_at': datetime.now(timezone.utc).isoformat(),
        'source': 'browser_ui_verified_then_readonly_database_export', 'real_model_called': True,
        'mcp_transport': 'real_stdio_subprocess', 'business_data': 'project_b_simulated',
        'capacity_test': False, 'tickets_in_demo_database': tickets, 'records': rows}
    output = root / 'evaluation/agent_runtime/s5'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'browser-acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    text = '# S5 真实模型网页验收\n\n实际在 8007 网页提交并核对；真实模型、真实 MCP 协议，固定模拟业务数据。不是容量测试。\n\n'
    text += '| 场景 | 状态 / 原因 | 模型请求 | 工具尝试 | 配对 | run_id |\n|---|---|---:|---:|---|---|\n'
    for row in rows:
        response = row['response']; r = response['result']
        text += f"| {row['case']} | {response['status']} / {r['stop_reason']} | {r['budget']['model_requests']} | {r['budget']['tool_attempts']} | {row['all_tool_calls_paired']} | {response['run_id']} |\n"
    for row in rows:
        response = row['response']; r = response['result']
        text += '\n## ' + row['case'] + '\n\n' + response['answer'] + '\n\n'
        text += 'Skill：' + json.dumps(r['loaded_skills'], ensure_ascii=False) + '\n\n'
        text += '实际工具链：' + ' → '.join(v['name'] + ':' + v['observation']['status'] for v in r['tool_records']) + '\n'
    text += f'\n演示库工单数：{tickets}。正常/空物流由实际 MCP 返回；越权须在本地订单边界拦截。超时、服务退出、坏响应及中断刷新由自动隔离故障测试覆盖。\n'
    text += '\n## 发现与修正\n\nA900 首次已被 ORDER_FORBIDDEN 拦截，未进入 MCP 读取；完成门只显示证据不足，提示不清晰。修正 S5 完成门后，重测明确返回 ORDER_FORBIDDEN 并说明当前买家无权访问。首次记录保留，不宣称全部首次成功。\n'
    text += '\n![最终 S5 网页结果与工具证据](browser-acceptance.jpg)\n'
    (output / 'browser-acceptance.md').write_text(text, encoding='utf-8')
    print(output / 'browser-acceptance.md')


if __name__ == '__main__':
    main()
