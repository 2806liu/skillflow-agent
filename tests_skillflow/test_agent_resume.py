"""S4 验收：脚本模型输出原生调用，实际执行图 Saver、业务事务与 HTTP 恢复。"""

import asyncio
from contextlib import closing
from dataclasses import replace
import json
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from tests_skillflow import test_agent_loop as fixtures
from skillflow.api import create_app
from skillflow.agent.contracts import ToolCall, ModelReply, MessageEnvelope


class ApplicationModel(fixtures.TestModel):
    """只用于离线验收：先复用观察驱动的调查，再按真实草稿结果提出提交。"""

    async def interpret_goal(self,text):
        """替身也输出独立意图对象；不绕过服务端原文和单号约束。"""
        from skillflow.agent.goals import application_intent
        proposal=await super().interpret_goal(text)
        app=application_intent(text)
        return proposal.model_copy(update={'requested_action':'apply','request_type':app['request_type']}) if app else proposal

    async def generate(self,messages,tool_specs):
        reply=await super().generate(messages,tool_specs)
        goal=json.loads(messages[0].content.split('固定目标：')[1].split('\n可加载技能目录：')[0])
        app=goal.get('application')
        if reply.message.tool_calls or not app:
            return reply
        calls={c.call_id:c for m in messages for c in m.tool_calls}
        observations={calls[m.tool_call_id].name:json.loads(m.content) for m in messages if m.role=='tool'}
        if 'create_ticket_draft' not in observations:
            name,args='create_ticket_draft',{**app,'reason':'用户本轮明确申请'}
        elif observations['create_ticket_draft']['status']=='success' and 'submit_ticket' not in observations:
            name,args='submit_ticket',{'ticket_id':observations['create_ticket_draft']['data']['ticket']['ticket_id']}
        else:
            return reply
        return ModelReply(message=MessageEnvelope(role='assistant',content='',tool_calls=[ToolCall(call_id=uuid4().hex,name=name,arguments=args)]))


class ResumeTests(fixtures.unittest.TestCase):
    """每项使用项目 B 内独立数据库，测试完仅清理自己的测试目录。"""

    setUp=fixtures.LoopTests.setUp
    tearDown=fixtures.LoopTests.tearDown
    ask=fixtures.LoopTests.ask

    def s4(self,limits=None):
        """显式启用 S4，替换模型避免任何外部请求。"""
        self.client.close();self.app.state.repository.engine.dispose()
        self.model=ApplicationModel()
        # patcher 本身没有 return_value，更新已安装 mock 的返回对象。
        import skillflow.agent.adapters.chat_model as gateway
        gateway.ChatModelGateway.return_value=self.model
        self.settings=replace(self.settings,agent_s4_enabled=True,**({'agent_limits':limits} if limits else {}))
        self.app=create_app(self.settings);self.client=TestClient(self.app)

    def body(self,result,decision='confirm',**changes):
        """卡片、版本与检查点都从服务端响应取得，不能自行编造可恢复状态。"""
        task=result['result']['task'];card=task['pending_action'] or {}
        return {'session_id':result['session_id'],'request_id':str(uuid4()),'expected_version':task['version'],
            'checkpoint_id':task['checkpoint_id'],'decision':decision,
            'confirmation_id':card.get('confirmation_id'),'action_hash':card.get('action_hash'),**changes}

    def resume(self,result,decision='confirm',**changes):
        response=self.client.post('/agent/tasks/'+result['result']['task']['task_id']+'/resume',json=self.body(result,decision,**changes))
        self.assertEqual(response.status_code,200,response.text)
        return response.json()

    def test_real_saver_pause_submit_same_task_new_run_and_budget(self):
        self.s4()
        first=self.ask('为 A123 申请退货')
        self.assertEqual(first['status'],'waiting_confirmation',first)
        self.assertTrue((self.root/'agent_checkpoints.sqlite3').exists())
        card=first['result']['pending_action']
        self.assertEqual(card['request_type'],'return')
        self.assertEqual(self.app.state.tickets.get(card['ticket_id'])['status'],'draft')
        second=self.resume(first)
        self.assertEqual(second['result']['stop_reason'],'TICKET_SUBMITTED',second)
        self.assertEqual(first['result']['metadata']['task_id'],second['result']['metadata']['task_id'])
        self.assertNotEqual(first['run_id'],second['run_id'])
        self.assertGreater(second['result']['budget']['tool_attempts'],first['result']['budget']['tool_attempts'])
        self.assertEqual(self.app.state.tickets.get(card['ticket_id'])['status'],'submitted')

    def test_inquiry_and_followup_never_inherit_write_intent(self):
        self.s4()
        first=self.ask('A123 能否退货')
        self.assertEqual(first['status'],'completed',first)
        self.assertIsNone(first['result']['task_goal']['application'])
        first=self.ask('为 A123 申请退货')
        second=self.ask('A124')
        self.assertIsNone(second['result']['task_goal']['application'])
        with closing(self.app.state.store.connect()) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM tickets').fetchone()[0],1)

    def test_restart_then_confirmation_uses_saved_tool_state(self):
        self.s4();first=self.ask('为 A123 申请退货')
        self.client.close();self.app.state.repository.engine.dispose()
        self.app=create_app(self.settings);self.client=TestClient(self.app)
        before=self.model.calls
        result=self.resume(first)
        self.assertEqual(result['result']['stop_reason'],'TICKET_SUBMITTED',result)
        self.assertEqual(self.model.calls,before)

    def test_expiry_renews_three_minute_card_without_submission(self):
        self.s4();now=[1000.0];self.app.state.agent_service.ledger.clock=lambda:now[0]
        first=self.ask('为 A123 申请退货')
        self.assertEqual(first['result']['pending_action']['expires_at'],1180)
        now[0]=1180
        second=self.resume(first)
        self.assertEqual(second['status'],'waiting_confirmation',second)
        self.assertNotEqual(first['result']['pending_action']['confirmation_id'],second['result']['pending_action']['confirmation_id'])
        self.assertEqual(second['result']['pending_action']['expires_at'],1360)
        self.assertEqual(self.app.state.tickets.get(first['result']['pending_action']['ticket_id'])['status'],'draft')
        with closing(self.app.state.store.connect()) as db:
            row=db.execute('SELECT status FROM agent_confirmations WHERE confirmation_id=?',
                (first['result']['pending_action']['confirmation_id'],)).fetchone()
            self.assertEqual(row[0],'invalidated')
        third=self.resume(second)
        self.assertEqual(third['result']['stop_reason'],'TICKET_SUBMITTED',third)

    def test_fact_change_new_card_then_explicit_second_confirmation(self):
        self.s4();first=self.ask('为 A123 申请退货')
        from skillflow.business.surge import Product
        with self.app.state.repository.sessions.begin() as db:
            db.get(Product,'earbuds-black').stock=4
        second=self.resume(first)
        self.assertEqual(second['status'],'waiting_confirmation',second)
        self.assertNotEqual(first['result']['pending_action']['facts_hash'],second['result']['pending_action']['facts_hash'])
        self.assertEqual(self.resume(second)['result']['stop_reason'],'TICKET_SUBMITTED')

    def test_eligibility_changed_keeps_draft_and_stops(self):
        self.s4();first=self.ask('为 A123 申请换货')
        from skillflow.business.surge import Product
        with self.app.state.repository.sessions.begin() as db:db.get(Product,'earbuds-black').stock=0
        second=self.resume(first)
        self.assertEqual(second['status'],'needs_manual_review',second)
        self.assertEqual(second['result']['stop_reason'],'APPLICATION_NOT_ELIGIBLE')
        self.assertFalse(second['result']['task']['resumable'])
        self.assertEqual(self.app.state.tickets.get(first['result']['pending_action']['ticket_id'])['status'],'draft')
        with closing(self.app.state.store.connect()) as db:
            self.assertEqual(db.execute('SELECT status FROM agent_confirmations WHERE confirmation_id=?',
                (first['result']['pending_action']['confirmation_id'],)).fetchone()[0],'invalidated')
        self.assertEqual(second['result']['tool_records'][-1]['observation']['status'],'rejected')

    def test_idempotent_request_and_stale_version(self):
        self.s4();first=self.ask('为 A123 申请退货')
        path='/agent/tasks/'+first['result']['task']['task_id']+'/resume';body=self.body(first)
        second=self.client.post(path,json=body)
        self.assertEqual(second.status_code,200,second.text)
        self.assertEqual(self.client.post(path,json=body).json(),second.json())
        body['request_id']=str(uuid4())
        self.assertEqual(self.client.post(path,json=body).status_code,409)
        with closing(self.app.state.store.connect()) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM ticket_events WHERE action='confirm'").fetchone()[0],1)

    def test_wrong_card_and_cross_buyer_are_rejected(self):
        self.s4();first=self.ask('为 A123 申请退货')
        path='/agent/tasks/'+first['result']['task']['task_id']+'/resume'
        self.assertEqual(self.client.post(path,json=self.body(first,confirmation_id='wrong')).status_code,409)
        svc=self.app.state.agent_service
        with self.assertRaisesRegex(ValueError,'TASK_FORBIDDEN'):
            asyncio.run(svc.detail(first['result']['task']['task_id'],'other-buyer','s3'))

    def test_quality_withdrawal_blocks_submission(self):
        self.s4()
        from evaluation.agent_runtime.s2_fixtures import seed_quality_demo
        seed_quality_demo(self.app.state.repository)
        first=self.ask('为 A700 充电宝质量事件申请退货')
        self.assertEqual(first['status'],'waiting_confirmation',first)
        self.app.state.repository.change_incident('S2-INCIDENT',actor='reviewer',action='withdraw',reason='测试撤回')
        second=self.resume(first)
        self.assertEqual(second['status'],'needs_manual_review',second)
        self.assertNotEqual(second['result']['stop_reason'],'TICKET_SUBMITTED')

    def test_crash_after_committed_transaction_replays_receipt_once(self):
        """事务先成功、图检查点未保存：恢复读取同事务回执，不新增提交事件。"""
        self.s4();first=self.ask('为 A123 申请退货')
        async def crash(point,state):
            if point=='after_submit_transaction':
                raise asyncio.CancelledError()
        self.app.state.agent_service.failpoint=crash
        second=self.resume(first)
        self.assertEqual(second['status'],'interrupted',second)
        ticket_id=first['result']['pending_action']['ticket_id']
        self.assertEqual(self.app.state.tickets.get(ticket_id)['status'],'submitted')
        self.app.state.agent_service.failpoint=None
        third=self.resume(second,'continue')
        self.assertEqual(third['result']['stop_reason'],'TICKET_SUBMITTED',third)
        with closing(self.app.state.store.connect()) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM ticket_events WHERE action='confirm'").fetchone()[0],1)
            self.assertEqual(db.execute("SELECT count(*) FROM agent_operation_receipts WHERE operation_id LIKE '%:submit:%'").fetchone()[0],1)

    def test_tool_receipt_ahead_of_checkpoint_and_model_attempt_charge(self):
        """已完成的 load_skill 不重复扣工具预算；未知模型响应的重试仍计请求。"""
        self.s4();count=[0]
        async def crash(point,state):
            if point=='after_tool_receipt' and count[0]==0:
                count[0]+=1
                raise asyncio.CancelledError()
        self.app.state.agent_service.failpoint=crash
        first=self.ask('为 A123 申请退货')
        self.assertEqual(first['status'],'interrupted',first)
        used=first['result']['budget']['tool_attempts']
        self.app.state.agent_service.failpoint=None
        second=self.resume(first,'continue')
        self.assertEqual(second['status'],'waiting_confirmation',second)
        self.assertEqual(used,1)
        self.assertEqual(second['result']['budget']['tool_attempts'],6)
        self.assertTrue(any(e['node']=='agent_tool_receipt_replay' for e in second['events']))

    def test_model_response_crash_does_not_reset_reserved_calls(self):
        self.s4();count=[0]
        async def crash(point,state):
            if point=='after_model_response' and count[0]==0:
                count[0]+=1
                raise asyncio.CancelledError()
        self.app.state.agent_service.failpoint=crash
        first=self.ask('为 A123 申请退货')
        self.assertEqual(first['status'],'interrupted',first)
        self.app.state.agent_service.failpoint=None
        second=self.resume(first,'continue')
        self.assertEqual(second['status'],'waiting_confirmation',second)
        self.assertEqual(second['result']['budget']['model_requests'],9)

    def test_terminal_budget_failure_cannot_resume(self):
        from skillflow.agent.config import BudgetLimits
        self.s4(BudgetLimits(max_model_requests=1))
        first=self.ask('为 A123 申请退货')
        self.assertEqual(first['status'],'failed',first)
        self.assertFalse(first['result']['task']['resumable'])
        path='/agent/tasks/'+first['result']['task']['task_id']+'/resume'
        self.assertEqual(self.client.post(path,json=self.body(first,'continue')).status_code,409)

    def test_sse_replay_does_not_renew_or_execute(self):
        self.s4();from tests_skillflow.test_streaming import frames
        job={'session_id':'s4-stream','run_id':str(uuid4()),'message':'为 A123 申请退货'}
        initial=frames(self.client.post('/chat/stream',json=job).text)
        result=initial[-1][2]
        self.assertEqual(result['status'],'waiting_confirmation',result)
        calls=self.model.calls
        self.assertEqual(frames(self.client.post('/chat/stream',json=job).text),initial)
        self.assertEqual(self.model.calls,calls)
        second=self.resume(result)
        replay=frames(self.client.get('/runs/'+second['run_id']+'/stream',params={'session_id':job['session_id']}).text)
        self.assertEqual(replay[-1][2],second)
        self.assertEqual(self.model.calls,calls)

    def test_hundred_concurrent_duplicate_confirm_requests_submit_once(self):
        """固定幂等竞争验收，不将此本地测试宣称为真实模型吞吐。"""
        self.s4();first=self.ask('为 A123 申请退货')
        path='/agent/tasks/'+first['result']['task']['task_id']+'/resume'
        body=self.body(first)
        # 一个 TestClient 门户中的异步并发使用 ASGITransport，避免跨事件循环共享锁。
        import httpx
        async def request_many():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),base_url='http://testserver') as client:
                return await asyncio.gather(*(client.post(path,json=body) for _ in range(100)))
        responses=asyncio.run(request_many())
        self.assertTrue(all(r.status_code in {200,429} for r in responses),[r.text for r in responses if r.status_code not in {200,429}])
        accepted=[r.json() for r in responses if r.status_code==200]
        self.assertTrue(accepted)
        self.assertEqual(len({r['run_id'] for r in accepted}),1)
        with closing(self.app.state.store.connect()) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM ticket_events WHERE action='confirm'").fetchone()[0],1)
        # 准入拒绝的重复请求随后重发，也返回同一提交结果。
        self.assertEqual(self.client.post(path,json=body).json()['run_id'],accepted[0]['run_id'])

    def test_expired_process_lease_requires_explicit_continue_and_new_card(self):
        """模拟进程在图暂停后、生命周期更新前退出，GET 仅显示，不自动执行。"""
        self.s4();first=self.ask('为 A123 申请退货');task=first['result']['task']
        svc=self.app.state.agent_service;now=svc.ledger.clock()
        with self.app.state.store.transaction() as db:
            db.execute("UPDATE agent_lifecycle SET status='running',lease_owner='dead-process',lease_until=?,segment_started=? WHERE task_id=?",
                (now-1,now-3,task['task_id']))
        detail=self.client.get('/agent/tasks/'+task['task_id'],params={'session_id':'s3'}).json()
        self.assertEqual(detail['status'],'interrupted',detail)
        first['result']['task']=detail
        second=self.resume(first,'continue')
        self.assertEqual(second['status'],'waiting_confirmation',second)
        self.assertNotEqual(second['result']['pending_action']['confirmation_id'],first['result']['pending_action']['confirmation_id'])

    def test_durable_cas_allows_only_one_competing_resume(self):
        self.s4();first=self.ask('为 A123 申请退货');task=first['result']['task']
        ledger=self.app.state.agent_service.ledger
        def claim(_):
            try:
                return ledger.claim_resume(task['task_id'],'demo-buyer','s3',task['version'],task['checkpoint_id'],str(uuid4()),str(uuid4()))
            except ValueError as exc:
                return str(exc)
        with ThreadPoolExecutor(max_workers=8) as pool:
            values=list(pool.map(claim,range(20)))
        self.assertEqual(sum(isinstance(v,tuple) and v[1] for v in values),1,values)
        self.assertTrue(all(isinstance(v,tuple) or v=='TASK_BUSY' for v in values),values)

    def test_skill_content_change_is_not_silently_restored(self):
        self.s4();first=self.ask('为 A123 申请退货')
        registry=self.app.state.agent_service.tools.skills
        doc=registry._documents['return_exchange']
        registry._documents['return_exchange']=doc.model_copy(update={'content_hash':'changed'})
        second=self.resume(first)
        self.assertEqual(second['result']['stop_reason'],'SKILL_VERSION_INCOMPATIBLE',second)
        self.assertFalse(second['result']['task']['resumable'])

    def test_change_application_invalidates_old_card_without_cancelling_draft(self):
        self.s4();first=self.ask('为 A123 申请退货')
        second=self.ask('为 A123 申请改成换货')
        path='/agent/tasks/'+first['result']['task']['task_id']+'/resume'
        self.assertEqual(self.client.post(path,json=self.body(first)).status_code,409)
        self.assertEqual(self.app.state.tickets.get(first['result']['pending_action']['ticket_id'])['status'],'draft')

    def test_explicit_short_confirmation_and_ambiguous_text_rejection(self):
        self.s4();first=self.ask('为 A123 申请退货')
        path='/agent/tasks/'+first['result']['task']['task_id']+'/resume'
        self.assertEqual(self.client.post(path,json=self.body(first,source_text='如果确认以后呢')).status_code,422)
        result=self.client.post(path,json=self.body(first,source_text='确认提交')).json()
        self.assertEqual(result['result']['stop_reason'],'TICKET_SUBMITTED',result)

    def test_quoted_conditional_or_negative_requests_do_not_authorize_draft(self):
        self.s4()
        for text in ['A123 能否申请退货','请解释为 A123 申请退货是什么意思','如果 A123 申请退货会怎样','A123 不要申请退货']:
            result=self.ask(text,str(uuid4()))
            self.assertIsNone((result['result']['task_goal'] or {}).get('application'),result)
        with closing(self.app.state.store.connect()) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM tickets').fetchone()[0],0)

    def test_price_and_discount_share_command_boundary(self):
        """价保与优惠使用既有处理器计算，提交仍经过同一确认与回执事务。"""
        self.s4()
        for message,kind in [('为 A123 申请价保补差','price_adjustment'),('为 A127 申请优惠争议','discount_dispute')]:
            first=self.ask(message,kind)
            self.assertEqual(first['status'],'waiting_confirmation',first)
            self.assertEqual(first['result']['pending_action']['request_type'],kind)
            self.assertEqual(self.resume(first)['result']['stop_reason'],'TICKET_SUBMITTED')

    def test_multiple_tasks_keep_separate_cards_and_scope(self):
        self.s4();first=self.ask('为 A123 申请退货');second=self.ask('为 A127 申请优惠争议')
        tasks=self.client.get('/agent/tasks',params={'session_id':'s3'}).json()['tasks']
        self.assertEqual(sum(t['status']=='waiting_confirmation' for t in tasks),2,tasks)
        self.assertEqual(self.resume(first)['result']['stop_reason'],'TICKET_SUBMITTED')
        self.assertEqual(self.resume(second)['result']['stop_reason'],'TICKET_SUBMITTED')

    def test_application_uses_trusted_order_category_for_incident(self):
        self.s4()
        from evaluation.agent_runtime.s2_fixtures import seed_quality_demo
        seed_quality_demo(self.app.state.repository)
        first=self.ask('为 A700 申请退货')
        self.assertEqual(first['status'],'waiting_confirmation',first)
        self.assertIn('incident',first['result']['task_goal']['requirements'])
        self.assertNotIn('returns',first['result']['task_goal']['requirements'])

    def test_out_of_stock_and_outside_window_stop_without_forced_draft(self):
        self.s4()
        for message,code in [('为 A124 申请换货','INSUFFICIENT_STOCK'),('为 A126 申请退货','APPLICATION_NOT_ELIGIBLE')]:
            first=self.ask(message,code)
            self.assertEqual(first['status'],'needs_manual_review',first)
            self.assertEqual(first['result']['stop_reason'],code)
        with closing(self.app.state.store.connect()) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM tickets').fetchone()[0],0)

    def test_draft_transaction_ahead_of_checkpoint_does_not_create_twice(self):
        self.s4();count=[0]
        async def crash(point,state):
            if point=='after_draft_transaction' and count[0]==0:
                count[0]+=1
                raise asyncio.CancelledError()
        self.app.state.agent_service.failpoint=crash
        first=self.ask('为 A123 申请退货')
        self.assertEqual(first['status'],'interrupted',first)
        self.app.state.agent_service.failpoint=None
        second=self.resume(first,'continue')
        self.assertEqual(second['status'],'waiting_confirmation',second)
        with closing(self.app.state.store.connect()) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM tickets').fetchone()[0],1)
            self.assertEqual(db.execute("SELECT count(*) FROM ticket_events WHERE action='draft_created'").fetchone()[0],1)

    def test_registry_change_is_not_silently_restored(self):
        self.s4();first=self.ask('为 A123 申请退货')
        registry=self.app.state.agent_service.tools.registry
        definition=registry._definitions['get_order']
        registry._definitions['get_order']=replace(definition,description='新版本 Schema 元数据')
        second=self.resume(first)
        self.assertEqual(second['result']['stop_reason'],'CHECKPOINT_INCOMPATIBLE',second)
