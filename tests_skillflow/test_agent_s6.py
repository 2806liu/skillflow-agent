"""S6 新登录入口的联合回归：模型与 IdP 替身，真实 MCP、业务事务和图 Saver。"""

import asyncio
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import httpx

from skillflow.api import create_app
from skillflow.auth import CSRF_COOKIE
from skillflow.config import ROOT, Settings
from skillflow.sqlite_store import SqliteStore
from skillflow.business.surge import BusinessRepository
from skillflow.business.s6_samples import prepare_samples
from skillflow.agent.contracts import MessageEnvelope, ModelReply, ToolCall
from tests_skillflow.test_agent_resume import ApplicationModel
from tests_skillflow.test_auth import FakeIdentityProvider


class CompositeModel(ApplicationModel):
    """按真实调用 ID 配对观察，组合多个 Skill；仅用于可重复回归，绝不生产回退。"""

    async def generate(self, messages, tool_specs):
        self.calls += 1
        goal = json.loads(messages[0].content.split('固定目标：')[1].split('\n可加载技能目录：')[0])
        calls = {c.call_id: c for m in messages for c in m.tool_calls}
        observations = {calls[m.tool_call_id].name: json.loads(m.content) for m in messages if m.role == 'tool'}
        loaded = {c.arguments['name'] for c in calls.values() if c.name == 'load_skill'}
        needed = [skill for req, skill in [('returns','return_exchange'), ('incident','quality_incident'),
                   ('logistics','logistics_tracking')] if req in goal['requirements']] or ['order_tracking']
        def reply(items):
            """一次模型响应可含多个独立加载调用，执行器仍逐个验证与反馈。"""
            return ModelReply(message=MessageEnvelope(role='assistant', content='' if items else '调查结束',
                tool_calls=[ToolCall(call_id=uuid4().hex, name=n, arguments=a) for n,a in items]))
        missing = [s for s in needed if s not in loaded]
        if missing:
            return reply([('load_skill', {'name': s, 'reason': '完成本次目标'}) for s in missing])
        ids = goal['scope']['explicit_order_ids']
        if not ids:
            return reply([] if 'list_my_orders' in observations else [('list_my_orders', {'keyword': goal['scope']['keyword']})])
        oid = ids[0]
        if 'get_order' not in observations:
            return reply([('get_order', {'order_id': oid})])
        if observations['get_order']['status'] != 'success':
            return reply([])
        for req, tools in [('returns', ['retrieve_policy','assess_after_sales','check_inventory']),
                           ('incident',['match_quality_incident']), ('alternatives',['list_alternatives']),
                           ('logistics',['get_logistics'])]:
            if req not in goal['requirements']:
                continue
            for tool in tools:
                if tool in observations:
                    continue
                args = {'order_id': oid}
                if tool == 'retrieve_policy': args['query'] = '退换期限与条件'
                if tool == 'assess_after_sales':
                    evidence = observations['retrieve_policy']['data'].get('evidence', [])
                    if not evidence: break
                    args['evidence_refs'] = [e['ref'] for e in evidence]
                if tool == 'check_inventory' and 'assess_after_sales' not in observations: break
                return reply([(tool, args)])
        application = goal.get('application')
        if application and 'create_ticket_draft' not in observations:
            return reply([('create_ticket_draft', {**application, 'reason':'本轮明确申请'})])
        if application and observations.get('create_ticket_draft',{}).get('status') == 'success' and 'submit_ticket' not in observations:
            return reply([('submit_ticket', {'ticket_id': observations['create_ticket_draft']['data']['ticket']['ticket_id']})])
        return reply([])


class IntegratedTests(unittest.IsolatedAsyncioTestCase):
    """所有HTTP请求经真实认证/CSRF，两个客户端在同一lifespan与事件循环中运行。"""

    async def asyncSetUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='s6-test-', dir=ROOT/'.skillflow'))
        store = SqliteStore(self.root)
        repo = BusinessRepository(store.database_path, ROOT/'data')
        self.manifest = prepare_samples(repo, ROOT/'data', self.root/'fixtures', ['alice-buyer','other-buyer'])
        repo.engine.dispose()
        settings = Settings(data_dir=self.root, fixtures_dir=self.root/'fixtures', auth_mode='production',
            oidc_issuer='https://identity.test', oidc_client_id='s6', oidc_client_secret='test-only',
            oidc_redirect_uri='https://testserver/auth/callback', business_backend='sqlalchemy',
            agent_engine='tool_loop', agent_model_mode='langchain_openai', agent_s4_enabled=True,
            agent_s5_enabled=True, agent_s6_enabled=True, agent_logistics_file=self.root/'fixtures/logistics.json',
            model='test-only', api_key='test-only')
        self.settings = settings
        self.model = CompositeModel()
        self.mock = patch('skillflow.agent.adapters.chat_model.ChatModelGateway', return_value=self.model)
        self.mock.start()
        self.app = create_app(settings, identity_provider=FakeIdentityProvider())
        for subject,buyer in [('alice','alice-buyer'),('bob','other-buyer')]:
            self.app.state.store.provision_identity(settings.oidc_issuer, subject, buyer)
        self.life = self.app.router.lifespan_context(self.app)
        await self.life.__aenter__()
        self.alice = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='https://testserver')
        self.bob = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='https://testserver')
        for client,subject in [(self.alice,'alice'),(self.bob,'bob')]:
            begin = await client.get('/auth/login')
            state = parse_qs(urlparse(begin.headers['location']).query)['state'][0]
            done = await client.post('/auth/callback', data={'state':state,'code':subject})
            self.assertEqual(done.status_code,303)

    async def asyncTearDown(self):
        await self.alice.aclose(); await self.bob.aclose()
        await self.life.__aexit__(None,None,None)
        self.app.state.repository.engine.dispose()
        self.mock.stop()
        shutil.rmtree(self.root)

    def oid(self, kind, buyer='alice-buyer'):
        """测试使用实际生成的订单清单，不假定 A123 属于登录买家。"""
        return next(r['order_id'] for r in self.manifest['cases'] if r['kind']==kind and r['buyer_id']==buyer)

    async def session(self, client):
        response = await client.post('/sessions', headers={'X-CSRF-Token':client.cookies.get(CSRF_COOKIE)})
        self.assertEqual(response.status_code,200,response.text)
        return response.json()['session_id']

    async def ask(self, client, message, session=None):
        session = session or await self.session(client)
        response = await client.post('/chat', json={'session_id':session,'message':message},
            headers={'X-CSRF-Token':client.cookies.get(CSRF_COOKIE)})
        self.assertEqual(response.status_code,200,response.text)
        return response.json()

    async def test_login_capabilities_samples_preserve_original_owners(self):
        me = (await self.alice.get('/auth/me')).json()
        self.assertTrue(me['runtime']['agent_s6_enabled'])
        self.assertEqual(len(me['runtime']['examples']),8)
        self.assertNotIn(self.oid('normal','other-buyer'),json.dumps(me['runtime']['examples']))
        # 结构化物流示例也必须按当前登录买家过滤，切换到 Bob 后不能沿用 Alice 的订单。
        for client,buyer in [(self.alice,'alice-buyer'),(self.bob,'other-buyer')]:
            examples=(await client.get('/auth/me')).json()['runtime']['examples']
            expected={r['order_id'] for r in self.manifest['cases'] if r['buyer_id']==buyer}
            self.assertEqual({r['order_id'] for r in examples},expected)
            self.assertEqual(next(r['order_id'] for r in examples if r['kind']=='normal'),self.oid('normal',buyer))
            self.assertEqual(next(r['order_id'] for r in examples if r['kind']=='no_policy'),self.oid('no_policy',buyer))
        self.assertEqual(self.app.state.repository.owner_of_order('A123'),'demo-buyer')
        self.assertEqual(self.app.state.repository.owner_of_order('A901'),'alice-buyer')
        before = len(self.manifest['cases'])
        self.assertEqual(len(prepare_samples(self.app.state.repository, ROOT/'data',self.root/'fixtures',
            ['alice-buyer','other-buyer'])['cases']),before)

    async def test_event_and_mcp_combination_and_followup_fresh_task(self):
        session = await self.session(self.alice)
        r = await self.ask(self.alice,f"{self.oid('affected')} 是否受质量事件影响并查询配送物流？",session)
        self.assertEqual(r['status'],'completed',r)
        names = [x['name'] for x in r['result']['tool_records']]
        self.assertIn('get_logistics',names); self.assertIn('match_quality_incident',names)
        self.assertTrue(r['result']['task']['checkpoint_id'])
        from evaluation.agent_runtime.s6_cases import judge
        ok,issues,security=judge({'kind':'affected','requirements':['order','incident','logistics']},
                                 r,'alice-buyer',self.oid('affected'))
        self.assertTrue(ok,(issues,security))
        self.assertGreaterEqual(r['result']['timing']['admission_wait_ms'],0)
        follow = await self.ask(self.alice,self.oid('unaffected'),session)
        self.assertEqual(follow['status'],'completed',follow)
        self.assertNotEqual(r['result']['task']['task_id'],follow['result']['task']['task_id'])
        self.assertIn('logistics',follow['result']['task_goal']['requirements'])
        # 裸单号只继承组合目标，每笔订单重新获得事件和物流证据，不复用上单结论。
        self.assertEqual(follow['result']['task_goal']['continuation_of_task_id'],r['result']['task']['task_id'])
        ok,issues,security=judge({'kind':'unaffected','requirements':['order','incident','logistics']},
                                 follow,'alice-buyer',self.oid('unaffected'))
        self.assertTrue(ok,(issues,security))
        self.assertFalse(set(r['result']['result_refs']) & set(follow['result']['result_refs']))
        # 明确切换为只查订单后，下次单号续问不再混入事件或物流目标。
        switched=await self.ask(self.alice,f"只查 {self.oid('normal')} 订单状态",session)
        next_order=await self.ask(self.alice,self.oid('no_stock'),session)
        for item,kind in [(switched,'normal'),(next_order,'no_stock')]:
            ok,issues,security=judge({'kind':kind,'requirements':['order']},item,'alice-buyer',self.oid(kind))
            self.assertTrue(ok,(issues,security))
            self.assertEqual([row['name'] for row in item['result']['tool_records']],['load_skill','get_order'])

    async def test_policy_and_mcp_combination(self):
        r = await self.ask(self.alice,f"调查 {self.oid('normal')} 退换资格并查询配送物流。")
        self.assertEqual(r['status'],'completed',r)
        self.assertTrue({'assess_after_sales','check_inventory','get_logistics'} <= {x['name'] for x in r['result']['tool_records']})

    async def test_joint_preflight_driver_checks_two_directions_without_writes(self):
        """补验脚本经过实际认证入口与图；六轮只读查询、双向12项越权拒绝不得建单。"""
        from evaluation.agent_runtime.s6_joint import check_joint
        report={}
        passed=await check_joint([self.alice,self.bob],self.manifest['buyers'],self.manifest,report,lambda:None)
        self.assertTrue(passed,report)
        self.assertEqual(len(report['queries']),6)
        self.assertEqual(len(report['denials']),12)
        self.assertEqual(report['requests_sent'],6)
        self.assertGreater(report['model_requests_observed'],0)
        with closing(self.app.state.store.connect()) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM tickets').fetchone()[0],0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM agent_confirmations').fetchone()[0],0)

    async def test_frozen_40_templates_with_real_tools_and_offline_judge(self):
        """校准正式评测器：40条模板经过实际业务/MCP/完成门，模型替身不算真实容量。"""
        from evaluation.agent_runtime.s6_cases import templates, judge
        cases=templates()
        self.assertEqual(len(cases),40)
        report={'scope':'offline_template_calibration','recorded_at':datetime.now(timezone.utc).isoformat(),
            'real_model_called':False,'real_keycloak_called':False,'cases':[],'passed':False}
        try:
            for index,case in enumerate(cases):
                client,buyer=(self.alice,'alice-buyer') if index%4<2 else (self.bob,'other-buyer')
                oid='A69999999' if case['kind']=='missing' else self.oid(case['kind'],buyer)
                # 每条独立会话，防止校准时短期目标继承掩盖模板本身的目标错误。
                result=await self.ask(client,case['prompt'].format(oid=oid))
                ok,issues,security=judge(case,result,buyer,oid)
                detail=result['result']
                report['cases'].append({'case_id':case['id'],'buyer_id':buyer,'order_id':oid,
                    'status':result['status'],'stop_reason':detail['stop_reason'],
                    'skills':detail['loaded_skills'],'tools':[r['name'] for r in detail['tool_records']],
                    'result_refs':detail['result_refs'],'correct':ok,'issues':issues,'security':security})
                self.assertTrue(ok,(case['id'],issues,security,detail['stop_reason']))
        finally:
            # 单独归档逐案例证据，成功与失败均保留；不覆盖真实模型容量或全量回归。
            report['passed']=len(report['cases'])==40 and all(r['correct'] for r in report['cases'])
            output=ROOT/'evaluation/agent_runtime/s6';output.mkdir(parents=True,exist_ok=True)
            name='offline-template-calibration-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:6]+'.json'
            (output/name).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

    async def resume_task(self,result,decision='confirm'):
        """每次续办都使用服务端最新卡片、版本与当前Alice会话的CSRF。"""
        task=result['result']['task'];card=task['pending_action'] or {}
        response=await self.alice.post('/agent/tasks/'+task['task_id']+'/resume',
            headers={'X-CSRF-Token':self.alice.cookies.get(CSRF_COOKIE)},json={
                'session_id':result['session_id'],'request_id':str(uuid4()),'expected_version':task['version'],
                'checkpoint_id':task['checkpoint_id'],'decision':decision,
                'confirmation_id':card.get('confirmation_id'),'action_hash':card.get('action_hash')})
        self.assertEqual(response.status_code,200,response.text)
        return response.json()

    async def test_authenticated_expiry_fact_change_and_restart_receipt(self):
        """新身份链路串联过期续卡、事实变化再确认及提交后崩溃，数据库只能提交一次。"""
        from skillflow.business.surge import Product
        now=[time.time()]
        self.app.state.agent_service.ledger.clock=lambda:now[0]
        first=await self.ask(self.alice,f"为 {self.oid('normal')} 申请退货")
        self.assertEqual(first['status'],'waiting_confirmation',first)
        ticket_id=first['result']['pending_action']['ticket_id']
        now[0]+=180
        renewed=await self.resume_task(first)
        self.assertEqual(renewed['status'],'waiting_confirmation',renewed)
        self.assertNotEqual(first['result']['pending_action']['confirmation_id'],renewed['result']['pending_action']['confirmation_id'])
        self.assertEqual(self.app.state.tickets.get(ticket_id)['status'],'draft')
        with self.app.state.repository.sessions.begin() as db:
            db.get(Product,'S6-EAR-BLACK').stock=4
        changed=await self.resume_task(renewed)
        self.assertEqual(changed['status'],'waiting_confirmation',changed)
        self.assertNotEqual(renewed['result']['pending_action']['facts_hash'],changed['result']['pending_action']['facts_hash'])
        self.assertEqual(self.app.state.tickets.get(ticket_id)['status'],'draft')
        async def crash(point,state):
            """仅在测试注入取消，覆盖业务提交已成功、图状态未保存的崩溃窗口。"""
            if point=='after_submit_transaction': raise asyncio.CancelledError()
        self.app.state.agent_service.failpoint=crash
        interrupted=await self.resume_task(changed)
        self.assertEqual(interrupted['status'],'interrupted',interrupted)
        self.assertEqual(self.app.state.tickets.get(ticket_id)['status'],'submitted')
        # 重建应用与实际Saver；现有登录Cookie继续由同一业务库校验，不重新伪造身份。
        cookies=[httpx.Cookies(c.cookies) for c in (self.alice,self.bob)]
        await self.alice.aclose();await self.bob.aclose()
        await self.life.__aexit__(None,None,None)
        self.app.state.repository.engine.dispose()
        self.app=create_app(self.settings,identity_provider=FakeIdentityProvider())
        self.life=self.app.router.lifespan_context(self.app);await self.life.__aenter__()
        self.alice,self.bob=[httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
            base_url='https://testserver',cookies=c) for c in cookies]
        continued=await self.resume_task(interrupted,'continue')
        self.assertEqual(continued['result']['stop_reason'],'TICKET_SUBMITTED',continued)
        self.assertEqual(continued['result']['task']['task_id'],first['result']['task']['task_id'])
        self.assertGreaterEqual(continued['result']['budget']['tool_attempts'],interrupted['result']['budget']['tool_attempts'])
        with closing(self.app.state.store.connect()) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM ticket_events WHERE action='confirm'").fetchone()[0],1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM agent_operation_receipts WHERE operation_id LIKE '%:submit:%'").fetchone()[0],1)

    async def test_authenticated_mcp_failure_is_local_degradation(self):
        """物流断连仍可展示本人订单；故障不得假装成功，政策功能继续可用。"""
        await self.app.state.agent_service.mcp_manager.aclose()
        logistics=await self.ask(self.alice,f"查询 {self.oid('normal')} 的配送物流")
        self.assertEqual(logistics['status'],'needs_manual_review',logistics)
        self.assertEqual(logistics['result']['stop_reason'],'MCP_UNAVAILABLE')
        self.assertIn(self.oid('normal'),logistics['answer'])
        self.assertEqual((await self.alice.get('/health')).json()['status'],'degraded')
        policy=await self.ask(self.bob,f"核验 {self.oid('normal','other-buyer')} 的退换资格")
        self.assertEqual(policy['status'],'completed',policy)

    async def test_cross_buyer_tasks_runs_resume_csrf_and_order(self):
        r = await self.ask(self.alice,f"为 {self.oid('normal')} 申请退货")
        self.assertEqual(r['status'],'waiting_confirmation',r)
        task = r['result']['task']; card=task['pending_action']; bs=await self.session(self.bob)
        url='/agent/tasks/'+task['task_id']
        self.assertEqual((await self.bob.get(url,params={'session_id':bs})).status_code,403)
        self.assertEqual((await self.bob.get('/runs/'+r['run_id'])).status_code,403)
        body={'session_id':bs,'request_id':str(uuid4()),'expected_version':task['version'],
              'checkpoint_id':task['checkpoint_id'],'decision':'confirm',
              'confirmation_id':card['confirmation_id'],'action_hash':card['action_hash']}
        self.assertEqual((await self.bob.post(url+'/resume',json=body)).status_code,403)
        self.assertEqual((await self.bob.post(url+'/resume',json=body,
            headers={'X-CSRF-Token':self.bob.cookies.get(CSRF_COOKIE)})).status_code,403)
        self.assertEqual((await self.bob.post('/chat',json={'session_id':bs,'message':self.oid('normal')},
            headers={'X-CSRF-Token':self.bob.cookies.get(CSRF_COOKIE)})).status_code,403)
        body['session_id']=r['session_id']
        committed = await self.alice.post(url+'/resume',json=body,headers={'X-CSRF-Token':self.alice.cookies.get(CSRF_COOKIE)})
        self.assertEqual(committed.status_code,200,committed.text)
        self.assertEqual(committed.json()['status'],'completed',committed.text)

    async def test_two_buyers_four_sessions_isolation(self):
        jobs=[]
        for client,buyer in [(self.alice,'alice-buyer'),(self.bob,'other-buyer')]*2:
            jobs.append(self.ask(client,f"查询 {self.oid('normal',buyer)} 的配送物流"))
        rows=await asyncio.gather(*jobs)
        self.assertEqual(len({r['session_id'] for r in rows}),4)
        for r in rows: self.assertEqual(r['status'],'completed',r)

    async def test_hundred_authenticated_duplicate_confirmations(self):
        """新身份入口的100请求竞争；允许明确准入拒绝，正式提交只能发生一次。"""
        r=await self.ask(self.alice,f"为 {self.oid('normal')} 申请退货")
        self.assertEqual(r['status'],'waiting_confirmation',r)
        task=r['result']['task']; card=task['pending_action']
        body={'session_id':r['session_id'],'request_id':str(uuid4()),'expected_version':task['version'],
            'checkpoint_id':task['checkpoint_id'],'decision':'confirm',
            'confirmation_id':card['confirmation_id'],'action_hash':card['action_hash']}
        url='/agent/tasks/'+task['task_id']+'/resume'
        headers={'X-CSRF-Token':self.alice.cookies.get(CSRF_COOKIE)}
        responses=await asyncio.gather(*(self.alice.post(url,json=body,headers=headers) for _ in range(100)))
        self.assertTrue(all(x.status_code in {200,429} for x in responses))
        accepted=[x.json() for x in responses if x.status_code==200]
        self.assertTrue(accepted)
        self.assertEqual(len({x['run_id'] for x in accepted}),1)
        with closing(self.app.state.store.connect()) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM ticket_events WHERE action='confirm'").fetchone()[0],1)
        calls=self.model.calls
        replay=await self.alice.get('/runs/'+accepted[0]['run_id']+'/stream',params={'session_id':r['session_id']})
        self.assertEqual(replay.status_code,200)
        self.assertEqual(self.model.calls,calls)

    async def test_infrastructure_40_samples(self):
        """只测 warm MCP 及本地恢复加载，禁止把无模型测试冒充完整任务延迟。"""
        r=await self.ask(self.alice,f"查询 {self.oid('normal')} 的配送物流")
        mcp,restore=[],[]
        for _ in range(40):
            start=time.perf_counter()
            await self.app.state.agent_service.mcp_manager.call('alice-buyer',self.oid('normal'))
            mcp.append(time.perf_counter()-start)
            start=time.perf_counter()
            from evaluation.agent_runtime.s6_bench import load_recovery_state
            loaded=await load_recovery_state(self.app.state.agent_service,r['result']['task']['task_id'],'alice-buyer',r['session_id'])
            self.assertTrue(loaded['tool_receipts'])
            restore.append(time.perf_counter()-start)
        self.assertLessEqual(sorted(mcp)[37],1)
        self.assertLessEqual(sorted(restore)[37],2)
        output=ROOT/'evaluation/agent_runtime/s6'
        output.mkdir(parents=True,exist_ok=True)
        (output/'infrastructure.json').write_text(json.dumps({'model':'substitute','samples':40,
            'mcp_warm_seconds':mcp,'checkpoint_read_seconds':restore,
            'restore_scope':'ownership,version,Saver,Skill hash,evidence,tool receipts; no graph execution',
            'passed':True}),encoding='utf-8')
