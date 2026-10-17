"""已登录客户端的S6联合补验：只读业务查询和跨买家拒绝，不创建或提交工单。"""

from uuid import uuid4

from .s6_cases import judge


async def check_joint(clients, buyers, manifest, report, save):
    """由正常OIDC登录后的两个客户端顺序执行，逐项留证；失败后不进入正式容量。"""
    lookup = {(r['buyer_id'], r['kind']): r['order_id'] for r in manifest['cases']}
    report.update(state='running', queries=[], denials=[], issues=[], passed=False,
                  requests_sent=0, model_requests_observed=0, model_called=False)

    def headers(client):
        """CSRF只随同源请求发送，不把认证Cookie或CSRF写入结果。"""
        return {'X-CSRF-Token': client.cookies.get('__Host-skillflow-csrf', '')}

    async def request(client, method, path, **kwargs):
        """只允许原应用的相对路径；禁止跟随重定向把敏感请求转交其他地址。"""
        return await client.request(method, path, headers=headers(client), follow_redirects=False, **kwargs)

    sessions = []
    for client in clients:
        response = await request(client, 'POST', '/sessions')
        response.raise_for_status()
        sessions.append(response.json()['session_id'])
    cases = [
        (0, 'combined', 'affected', ['order', 'incident', 'logistics'], '{oid} 是否受质量事件影响并查询配送物流？'),
        (0, 'inherit', 'unaffected', ['order', 'incident', 'logistics'], '{oid}'),
        (0, 'switch', 'normal', ['order'], '只查 {oid} 的订单状态'),
        (0, 'switched_followup', 'no_stock', ['order'], '{oid}'),
        (1, 'bob_returns', 'normal', ['order', 'returns'], '核对 {oid} 的退换资格和换货库存。'),
        (1, 'bob_combined', 'affected', ['order', 'incident', 'logistics'], '{oid} 是否受质量事件影响并查询配送物流？'),
    ]
    bodies = {}
    for person, name, kind, requirements, prompt in cases:
        buyer, client = buyers[person], clients[person]
        oid = lookup[(buyer, kind)]
        row = {'case': name, 'buyer_id': buyer, 'order_id': oid, 'message': prompt.format(oid=oid)}
        report['queries'].append(row)
        # 发送后若超时，模型是否调用未知；联合样本与正式容量的计数分别保存。
        report['requests_sent'] += 1
        if report['model_called'] is False:
            report['model_called'] = None
        save()
        response = await request(client, 'POST', '/chat', json={'session_id': sessions[person], 'message': row['message']})
        row['http_status'] = response.status_code
        if response.status_code != 200:
            row.update(correct=False, issues=['HTTP_ERROR'])
        else:
            body = response.json()
            consumed = (body.get('result', {}).get('budget') or {}).get('model_requests', 0)
            if consumed > 0:
                report['model_called'] = True
                report['model_requests_observed'] += consumed
            ok, issues, security = judge({'kind': kind, 'requirements': requirements}, body, buyer, oid)
            row.update(correct=ok, issues=issues, security=security, response=body)
            bodies[name] = body
            parent = {'inherit': 'combined', 'switched_followup': 'switch'}.get(name)
            # 新task关联前次目的，结果引用必须重新取得；显式切换必须被识别为新目标。
            if parent:
                if body['result']['task_goal']['continuation_of_task_id'] != bodies[parent]['result']['metadata']['task_id']:
                    row['issues'].append('WRONG_CONTINUATION_SOURCE')
                if set(body['result']['result_refs']) & set(bodies[parent]['result']['result_refs']):
                    row['issues'].append('REUSED_EVIDENCE')
            elif name in {'combined', 'switch'} and body['result']['conversation_context']['action'] != 'new':
                row['issues'].append('EXPLICIT_NEW_GOAL_REQUIRED')
            row['correct'] = ok and not row['issues'] and not security
        save()
        if not row['correct']:
            report['state'] = 'query_failed'
            return False
    # 两个方向都用真实登录身份探测新增入口；被探测任务均已完成，不携带提交动作或确认卡片。
    for person, foreign_case in [(0, 'bob_combined'), (1, 'combined')]:
        client, session = clients[person], sessions[person]
        foreign = bodies[foreign_case]
        task = foreign['result']['task']
        checks = [
            ('session', 'GET', '/sessions/' + foreign['session_id'], {}),
            ('task_list', 'GET', '/agent/tasks', {'params': {'session_id': foreign['session_id']}}),
            ('task', 'GET', '/agent/tasks/' + task['task_id'], {'params': {'session_id': session}}),
            ('run', 'GET', '/runs/' + foreign['run_id'], {}),
            ('resume', 'POST', '/agent/tasks/' + task['task_id'] + '/resume', {'json': {
                'session_id': session, 'request_id': str(uuid4()), 'expected_version': task['version'],
                'checkpoint_id': task['checkpoint_id'], 'decision': 'continue'}}),
            ('order', 'POST', '/chat', {'json': {'session_id': session,
                'message': '查询订单 ' + lookup[(buyers[1-person], 'normal')]}}),
        ]
        for name, method, path, kwargs in checks:
            response = await request(client, method, path, **kwargs)
            row = {'buyer_id': buyers[person], 'entry': name, 'http_status': response.status_code,
                   'correct': response.status_code == 403}
            report['denials'].append(row)
            save()
            if not row['correct']:
                report.update(state='stopped_security_failure')
                report['issues'].append('CROSS_BUYER_' + name.upper() + '_NOT_REJECTED')
                return False
    report.update(state='complete', passed=True)
    save()
    return True
