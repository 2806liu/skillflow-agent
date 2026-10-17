"""只读导出已由网页核对的 S4 运行及数据库终态，不发模型请求或新增业务操作。"""

import argparse
from contextlib import closing
from datetime import datetime,timezone
import json
from pathlib import Path
import sqlite3


def main():
    """分清累计预算与本 run 尝试；待确认的未完成调用明确标注，不冒充工具成功。"""
    root=Path(__file__).resolve().parent.parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='append',default=[],help='case=run_id，可重复')
    args=parser.parse_args()
    database=(root/'.skillflow/agent-s4-demo/skillflow.sqlite3').resolve()
    records=[]
    with closing(sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)) as db:
        db.row_factory=sqlite3.Row
        for item in args.run:
            case,run_id=item.split('=',1)
            raw=db.execute('SELECT value_json FROM kv WHERE name=?',('run_'+run_id,)).fetchone()
            if not raw:
                raise SystemExit('运行尚未保存：'+run_id)
            response=json.loads(raw[0]);r=response['result']
            if r.get('stage')!='S4':
                raise SystemExit('不是 S4 记录：'+run_id)
            raw=db.execute('SELECT state_json FROM agent_runs WHERE run_id=?',(run_id,)).fetchone()
            state=json.loads(raw[0])
            calls=[v['call_id'] for m in state['messages'] for v in m.get('tool_calls',[])]
            observations=[m['tool_call_id'] for m in state['messages'] if m['role']=='tool']
            unmatched=sorted(set(calls)-set(observations))
            paired=len(calls)==len(set(calls)) and not (set(observations)-set(calls)) and len(observations)==len(set(observations))
            pending=state['constraints'].get('_pending_calls',[])
            index=state['constraints'].get('_call_index',0)
            allowed=[c['call_id'] for c in pending[index:]] if state['status'] in {'waiting_confirmation','interrupted'} else []
            paired=paired and unmatched==sorted(allowed)
            attempts={row['kind']:row['n'] for row in db.execute('SELECT kind,count(*) n FROM agent_attempts WHERE run_id=? GROUP BY kind',(run_id,))}
            records.append({'case':case,'response':response,'messages':state['messages'],
                'tool_messages_or_pending_matched':paired,'pending_tool_call_ids':unmatched,'run_attempts':attempts,
                'trusted_tool_state_saved':bool(state['constraints'].get('_tool_state'))})
        tickets=[dict(row) for row in db.execute('SELECT ticket_id,order_id,request_type,status FROM tickets ORDER BY created_at')]
        submits=db.execute("SELECT count(*) FROM ticket_events WHERE action='confirm' AND result='submitted'").fetchone()[0]
        receipts=db.execute("SELECT count(*) FROM agent_operation_receipts WHERE operation_id LIKE '%:submit:%'").fetchone()[0]
    report={'recorded_at':datetime.now(timezone.utc).isoformat(),'source':'browser_ui_manual_verification',
        'business_data':'project_b_simulated','capacity_test':False,'records':records,'tickets':tickets,
        'formal_submit_events':submits,'submit_operation_receipts':receipts}
    output=root/'evaluation/agent_runtime/s4';output.mkdir(parents=True,exist_ok=True)
    (output/'browser-acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    text='# S4 真实模型网页验收\n\n2026-10-16。实际在独立 8006 网页提交并检查结果；随后只读导出 B 数据库记录。\n\n'
    text+='新建调查/申请使用 B 配置的外部真实模型；确认恢复段可以不请求模型。本轮为模拟订单与工单，不执行退款，也不是模型容量或成功率评测。\n\n'
    text+='| 场景 | 状态/原因 | 累计模型请求 | 本 run 模型尝试 | 本 run 工具尝试 | run_id |\n|---|---|---:|---:|---:|---|\n'
    for row in records:
        response=row['response'];r=response['result'];attempt=row['run_attempts']
        text+=f"| {row['case']} | {response['status']} / {r['stop_reason']} | {r['budget']['model_requests']} | {attempt.get('model',0)+attempt.get('model_decision',0)} | {attempt.get('tool',0)} | {response['run_id']} |\n"
    text+=f'\n正式提交事件 {submits} 条，提交操作回执 {receipts} 条。工单终态：\n\n'+json.dumps(tickets,ensure_ascii=False,indent=2)+'\n'
    for row in records:
        r=row['response']['result']
        text+='\n## '+row['case']+'\n\n'+row['response']['answer']+'\n\n'
        text+='任务：'+r['metadata']['task_id']+'；工具消息或待确认配对：'+str(row['tool_messages_or_pending_matched'])+'。\n\n'
        text+='Skill：'+json.dumps(r['loaded_skills'],ensure_ascii=False)+'。\n\n'
        text+='工具链：'+' → '.join(v['name'] for v in r['tool_records'])+'。\n\n'
        if r.get('pending_action'):
            text+='待确认卡片：'+json.dumps(r['pending_action'],ensure_ascii=False)+'。\n'
    text+='\n## 发现与修正\n\nA700 首轮加载多个 Skill 后，把普通政策未登记当成事件申请的阻碍，最终 APPLICATION_NOT_COMPLETED，未建单。已明确 return_exchange / quality_incident SOP 的适用条件，并改进完成门反馈；重测用可信事件资格创建草稿，保留首次未完成记录。S4 价保/优惠 Skill 的写工具范围也在自动测试中发现遗漏并补齐，最终四类工单回归通过。\n'
    text+='\nA127 首次 analyze_discount 使用不存在的证据引用，运行时返回 EVIDENCE_NOT_FOUND；模型读取观察后改用真实 evidence.ref，后续计算和草稿创建成功。此工具拒绝保留在同任务 Trace，不隐藏额外尝试。优惠争议草稿留供人工查看，未在本轮正式提交。\n'
    text+='\n## 边界\n\n- 刷新只加载任务；服务重启后明确确认才继续同一任务。待确认时允许 submit_ticket 尚无 ToolMessage，直到用户确认并执行业务命令才返回原调用 ID 的结果。\n- A123 在卡片有效期内完成跨服务重启后的提交；A700 实际超过 3 分钟后首次点击只核验续卡，随后明确短句确认才提交。两种结果分别记录，不把 A123 的有效期内提交说成过期续卡。\n- 正常重启与网页功能由本报告记录；事务提交窗口、中途取消、竞争、预算和版本不兼容由自动故障注入测试覆盖。\n- 本报告保留所测运行，不以少量功能案例宣称真实模型 P95 或并发容量。\n'
    (output/'browser-acceptance.md').write_text(text,encoding='utf-8')
    print(str(output/'browser-acceptance.md'))


if __name__=='__main__':
    main()
