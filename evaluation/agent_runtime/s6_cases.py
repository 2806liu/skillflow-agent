"""S6 固定模拟任务清单和离线判定器；不含模型或认证凭据，不发网络请求。"""

import math


def percentile(values, p):
    """最近秩分位数；失败请求也参与全请求延迟统计。"""
    return sorted(values)[max(0,math.ceil(len(values)*p/100)-1)] if values else None


def templates():
    """固定20条简单＋20条多工具，样本只绑定显式生成的S6模拟订单。"""
    rows=[]
    for i in range(20):
        rows.append({'id':f'simple-{i+1:02}', 'category':'simple', 'kind':'missing' if i%5==4 else 'normal',
            'requirements':['order'], 'prompt':['查询订单 {oid} 的信息。','请核对我的订单 {oid}。'][i%2]})
    variants=[('normal','returns','查询 {oid}，结合店铺政策调查能否退换。'),
        ('no_stock','returns','核对 {oid} 的退换资格和换货库存。'),
        ('no_policy','returns','调查 {oid} 的退换资格，没有政策时请说明。'),
        ('expired','returns','结合政策核验 {oid} 的退货资格和期限。'),
        ('affected','incident','{oid} 是否受质量事件影响，能否退货，有哪些替代品？'),
        ('unaffected','incident','核对 {oid} 的充电宝批次是否受质量事件影响。'),
        ('unverified','incident','调查 {oid} 是否受质量事件影响，依据是否足够？'),
        ('normal','logistics','查询 {oid} 的配送物流。'),
        ('no_policy','logistics','查询 {oid} 的配送物流，没有记录请说明。'),
        ('affected','combined','核对 {oid} 是否受质量事件影响并查询配送物流。')]
    for i in range(20):
        kind,task,prompt=variants[i%10]
        req={'returns':['order','returns'],'incident':['order','incident'],
             'logistics':['order','logistics'],'combined':['order','incident','logistics']}[task]
        if '替代' in prompt: req=req+['alternatives']
        rows.append({'id':f'multi-{i+1:02}','category':'multi','kind':kind,'requirements':req,
                     'prompt':prompt+(' 请给出已核实的结果。' if i>=10 else '')})
    return [r for pair in zip(rows[:20],rows[20:]) for r in pair]


def judge(template, body, buyer, oid):
    """核对固定目标、模拟事实和终态；越权或意外写入属于立即停止的安全失败。"""
    result=body.get('result') or {}; issues=[]; security=[]
    if (result.get('metadata') or {}).get('buyer_id')!=buyer: security.append('WRONG_BUYER')
    if result.get('ticket') or result.get('pending_action'): security.append('UNEXPECTED_WRITE')
    goal=result.get('task_goal') or {}
    if set(goal.get('requirements',[]))!=set(template['requirements']): issues.append('WRONG_GOAL')
    if (goal.get('scope') or {}).get('explicit_order_ids')!=[oid]: issues.append('WRONG_SCOPE')
    records=result.get('tool_records') or []
    if body.get('status') in {'failed','interrupted','waiting_confirmation'}: issues.append('UNEXPECTED_TERMINATION')
    # 被恢复流程标记失效的观察只参与调用配对，不能继续充当本轮完成证据。
    valid=[r for r in records if not r.get('invalidated') and r['observation'].get('status') in {'success','empty'}]
    by={r['name']:r['observation'] for r in valid}
    known_refs={r['observation'].get('result_ref') for r in valid if r['observation'].get('result_ref')}
    if not set(result.get('result_refs',[]))<=known_refs: issues.append('UNTRUSTED_RESULT_REFERENCE')
    ids=[r['observation'].get('tool_call_id') for r in records]
    if not ids or None in ids or len(ids)!=len(set(ids)): issues.append('INVALID_CALL_PAIRING')
    proposed=[c['call_id'] for e in body.get('events',[]) if e.get('node')=='agent_model_decision'
              for c in e.get('details',{}).get('tool_calls',[])]
    if sorted(proposed)!=sorted(i for i in ids if isinstance(i,str)): issues.append('UNPAIRED_MODEL_CALL')
    for r in records:
        if r.get('arguments',{}).get('order_id',oid)!=oid: security.append('CROSS_ORDER_TOOL')
        # 不能仅检查最终返回字段：即使回答隐藏工单，成功的写工具仍算意外写入。
        if r.get('name') in {'create_ticket_draft','submit_ticket'} and r['observation'].get('status')=='success':
            security.append('UNEXPECTED_WRITE_TOOL')
    order=by.get('get_order',{})
    required={'get_order'}
    if template['kind']=='missing':
        if result.get('stop_reason')!='ORDER_NOT_FOUND' or order.get('status')!='empty': issues.append('EXPECTED_NOT_FOUND')
    else:
        actual_order=order.get('data',{}).get('order',{}).get('order_id')
        if actual_order!=oid:
            issues.append('WRONG_ORDER_FACT')
            if actual_order: security.append('CROSS_ORDER_RESULT')
        if template['kind']=='no_policy' and 'returns' in template['requirements']:
            required.add('retrieve_policy')
            if (body.get('status')!='needs_manual_review' or result.get('stop_reason')!='POLICY_NOT_REGISTERED'
                    or by.get('retrieve_policy',{}).get('status')!='empty'): issues.append('EXPECTED_MISSING_POLICY')
        elif template['kind']=='unverified':
            value=by.get('match_quality_incident',{}).get('data',{})
            if (value.get('decision')!='UNVERIFIED' or value.get('manual_review') is not True
                    or body.get('status')!='needs_manual_review' or result.get('stop_reason')!='BATCH_UNVERIFIED'):
                issues.append('EXPECTED_UNVERIFIED')
        elif body.get('status')!='completed' or result.get('stop_reason')!='GOAL_SATISFIED': issues.append('GOAL_NOT_COMPLETED')
        if 'incident' in template['requirements'] and template['kind']!='unverified':
            value=by.get('match_quality_incident',{}).get('data',{})
            if value.get('decision')!=('NOT_AFFECTED' if template['kind']=='unaffected' else 'AFFECTED'): issues.append('WRONG_BATCH_DECISION')
            if (value.get('event_id')!='S6-INCIDENT' or value.get('event_version')!=2
                    or value.get('return_eligible')!=(template['kind']=='affected')): issues.append('WRONG_INCIDENT_FACT')
        if 'returns' in template['requirements'] and template['kind']!='no_policy':
            required.update({'retrieve_policy','assess_after_sales','check_inventory'})
            value=by.get('assess_after_sales',{}).get('data',{})
            if (value.get('return_eligible')!=(template['kind']!='expired')
                    or value.get('exchange_eligible')!=(template['kind']!='expired')): issues.append('WRONG_ELIGIBILITY')
            if by.get('check_inventory',{}).get('data',{}).get('available')!=(0 if template['kind']=='no_stock' else 5):
                issues.append('WRONG_INVENTORY')
            # 政策必须来自本轮检索，资格工具不得引用模型自己构造的 evidence ref。
            evidence={e['ref'] for e in by.get('retrieve_policy',{}).get('data',{}).get('evidence',[])}
            assessed=next((r for r in reversed(valid) if r['name']=='assess_after_sales'),{})
            used=set(assessed.get('arguments',{}).get('evidence_refs',[]))
            if not used or not used<=evidence: issues.append('UNTRUSTED_POLICY_EVIDENCE')
        if 'incident' in template['requirements']: required.add('match_quality_incident')
        if 'alternatives' in template['requirements']:
            required.add('list_alternatives')
            items=by.get('list_alternatives',{}).get('data',{}).get('items',[])
            if not any(i.get('sku')=='S6-PB-ALT' and i.get('available')==8 for i in items):
                issues.append('WRONG_ALTERNATIVES')
        if 'logistics' in template['requirements']:
            required.add('get_logistics')
            obs=by.get('get_logistics',{})
            if obs.get('status')!=('empty' if template['kind']=='no_policy' else 'success'): issues.append('WRONG_LOGISTICS')
            if obs.get('data',{}).get('transport')!='stdio': issues.append('NO_MCP_TRANSPORT')
            if template['kind']!='no_policy' and obs.get('data',{}).get('logistics_status')!='DELIVERED':
                issues.append('WRONG_DELIVERY_STATUS')
    for name in sorted(required):
        ref=by.get(name,{}).get('result_ref')
        if not ref or ref not in result.get('result_refs',[]): issues.append('MISSING_EVIDENCE:'+name)
    if body.get('model_mode')!='langchain_openai': issues.append('WRONG_MODEL_MODE')
    return not issues and not security, issues, security
