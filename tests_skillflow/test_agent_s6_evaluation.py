"""S6评测器自身的反例与调度测试；HTTP全部为替身，不能作为真实模型容量成绩。"""

import asyncio
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import httpx

from evaluation.agent_runtime.s6_cases import judge, percentile, templates
from scripts import agent_s6_real_model_acceptance as bench


def sample_body(incident=False):
    """构造固定可信观察形状，用删证据、改结果等反例验证判定器会拒绝假成功。"""
    oid='A610000'
    tools=[('get_order',{'order':{'order_id':oid}}, {})]
    if incident:
        tools += [('match_quality_incident',{'decision':'AFFECTED','event_id':'S6-INCIDENT',
            'event_version':2,'return_eligible':True},{}),
            ('list_alternatives',{'items':[{'sku':'S6-PB-ALT','available':8}]},{})]
        requirements=['order','incident','alternatives']
    else:
        tools += [('retrieve_policy',{'evidence':[{'ref':'e1'}]},{}),
            ('assess_after_sales',{'return_eligible':True,'exchange_eligible':True},{'evidence_refs':['e1']}),
            ('check_inventory',{'available':5},{})]
        requirements=['order','returns']
    records=[]
    for i,(name,data,arguments) in enumerate(tools):
        records.append({'name':name,'arguments':{'order_id':oid,**arguments},'observation':{
            'status':'success','data':data,'tool_call_id':f'call{i}','result_ref':f'r{i}'}})
    body={'status':'completed','model_mode':'langchain_openai','result':{
        'metadata':{'buyer_id':'alice-buyer'},'task_goal':{'requirements':requirements,
            'scope':{'explicit_order_ids':[oid]}},'stop_reason':'GOAL_SATISFIED',
        'tool_records':records,'result_refs':[r['observation']['result_ref'] for r in records]},
        'events':[{'node':'agent_model_decision','details':{'tool_calls':[
            {'call_id':r['observation']['tool_call_id']} for r in records]}}]}
    return {'kind':'affected' if incident else 'normal','requirements':requirements},body


class GraderTests(unittest.TestCase):
    """完成率按业务证据判断，不能只相信completed字符串或快速拒绝。"""

    def test_frozen_count_and_nearest_rank(self):
        rows=templates()
        self.assertEqual(len({r['id'] for r in rows}),40)
        self.assertEqual(sum(r['category']=='simple' for r in rows),20)
        self.assertEqual(percentile(list(range(1,21)),95),19)
        self.assertIsNone(percentile([],95))

    def test_positive_and_inventory_missing_wrong_or_invalidated(self):
        template,body=sample_body()
        self.assertTrue(judge(template,body,'alice-buyer','A610000')[0])
        for mutation in ['missing','wrong','invalidated']:
            changed=deepcopy(body)
            row=changed['result']['tool_records'][-1]
            if mutation=='missing': changed['result']['tool_records'].pop()
            elif mutation=='wrong': row['observation']['data']['available']=0
            else: row['invalidated']=True
            self.assertFalse(judge(template,changed,'alice-buyer','A610000')[0],mutation)

    def test_alternatives_and_event_version_must_match_fixed_facts(self):
        template,body=sample_body(True)
        self.assertTrue(judge(template,body,'alice-buyer','A610000')[0])
        for index,field,value in [(1,'event_version',1),(2,'items',[])]:
            changed=deepcopy(body)
            changed['result']['tool_records'][index]['observation']['data'][field]=value
            self.assertFalse(judge(template,changed,'alice-buyer','A610000')[0])

    def test_policy_evidence_and_result_refs_cannot_be_omitted_or_forged(self):
        template,body=sample_body()
        for mutation in ['empty_refs','forged_refs','forged_policy']:
            changed=deepcopy(body)
            if mutation=='empty_refs': changed['result']['result_refs']=[]
            elif mutation=='forged_refs': changed['result']['result_refs'].append('made-up')
            else: changed['result']['tool_records'][2]['arguments']['evidence_refs']=['made-up']
            self.assertFalse(judge(template,changed,'alice-buyer','A610000')[0],mutation)

    def test_security_and_pairing_failures(self):
        template,body=sample_body()
        for mutation in ['buyer','order','write','hidden_write','foreign_result','pairing']:
            changed=deepcopy(body)
            if mutation=='buyer': changed['result']['metadata']['buyer_id']='other-buyer'
            elif mutation=='order': changed['result']['tool_records'][0]['arguments']['order_id']='A900'
            elif mutation=='write': changed['result']['ticket']={'status':'draft'}
            elif mutation=='hidden_write': changed['result']['tool_records'][0]['name']='submit_ticket'
            elif mutation=='foreign_result': changed['result']['tool_records'][0]['observation']['data']['order']['order_id']='A900'
            else: changed['events']=[]
            ok,issues,security=judge(template,changed,'alice-buyer','A610000')
            self.assertFalse(ok)
            self.assertTrue(security if mutation!='pairing' else issues)


class LoginTransportTests(unittest.IsolatedAsyncioTestCase):
    """只在内存中模拟原OIDC跳转，证明地址在发出请求前校验，不读取真实密码。"""

    async def test_external_redirect_is_blocked_before_next_request(self):
        seen=[]
        def handler(request):
            seen.append(request.url.host)
            return httpx.Response(302,headers={'location':'https://external.invalid/collect'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(RuntimeError):
                await bench.local_request(client,'GET','https://localhost:8001/auth/login')
        self.assertEqual(seen,['localhost'])

    async def test_cross_origin_307_does_not_forward_form(self):
        seen=[]
        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(307,headers={'location':'https://localhost:8001/auth/callback'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(RuntimeError):
                await bench.local_request(client,'POST','https://localhost/realms/myrealm/login-actions/authenticate',
                    data={'username':'test-only','password':'test-only'})
        self.assertEqual(len(seen),1)

    async def test_password_form_cannot_post_to_application(self):
        seen=[]
        def handler(request):
            seen.append(request.method)
            return httpx.Response(200,text='<form action="https://localhost:8001/auth/callback"><input name="username"></form>')
        async with httpx.AsyncClient(base_url='https://localhost:8001',transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(RuntimeError): await bench.login(client,'test-only','test-only')
        self.assertEqual(seen,['GET'])

    async def test_normal_login_and_wrong_password_single_attempt(self):
        for accepted in [True,False]:
            passwords=[]
            def handler(request):
                path=request.url.path
                if path=='/auth/login':
                    return httpx.Response(303,headers={'location':'https://localhost/realms/myrealm/login'})
                if path.endswith('/login') or (path.endswith('/authenticate') and not accepted):
                    if request.method=='POST': passwords.append(request.content)
                    return httpx.Response(200,text='<form action="/realms/myrealm/login-actions/authenticate"><input name="username"><input type="password" name="password"></form>')
                if path.endswith('/authenticate'):
                    passwords.append(request.content)
                    return httpx.Response(200,text='<form action="https://localhost:8001/auth/callback"><input name="code" value="test-only-code"><input name="state" value="test-only-state"></form>')
                if path=='/auth/callback': return httpx.Response(303,headers={'location':'/'})
                if path=='/auth/me': return httpx.Response(200,json={'buyer_id':'alice-buyer','runtime':{'agent_s6_enabled':True}})
                return httpx.Response(200,text='done')
            async with httpx.AsyncClient(base_url='https://localhost:8001',transport=httpx.MockTransport(handler)) as client:
                if accepted: self.assertEqual(await bench.login(client,'test-only','test-only'),'alice-buyer')
                else:
                    with self.assertRaises(RuntimeError): await bench.login(client,'test-only','test-only')
            self.assertEqual(len(passwords),1)


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    """验证120条调度、失败保留与安全停止；内存HTTP和判定替身不会写正式容量记录。"""

    async def asyncSetUp(self):
        self.project=Path(__file__).resolve().parents[1]
        self.root=Path(tempfile.mkdtemp(prefix='s6-evaluation-test-',dir=self.project/'.skillflow'))

    async def asyncTearDown(self):
        """只清理本测试在项目B内创建的临时目录。"""
        self.assertTrue(self.root.resolve().is_relative_to((self.project/'.skillflow').resolve()))
        shutil.rmtree(self.root)

    async def execute(self,mode):
        """评测器按原流程运行，但所有身份、HTTP、模型证据均在测试里替换。"""
        output_root=self.root/mode;output_root.mkdir()
        people=['alice-buyer','other-buyer'];sent=[];sessions=[]
        kinds={r['kind'] for r in templates()}-{'missing'}
        manifest={'buyers':people,'cases':[{'buyer_id':b,'kind':k,'order_id':f'A{j+610000}'}
            for j,(b,k) in enumerate((b,k) for b in people for k in sorted(kinds))]}
        client_type=httpx.AsyncClient
        async def handler(request):
            await asyncio.sleep(0)
            if request.url.path=='/sessions':
                if mode=='session_error': return httpx.Response(503)
                sid=f's{len(sessions)}';sessions.append(sid)
                return httpx.Response(200,json={'session_id':sid})
            payload=json.loads(request.content);sent.append(payload)
            if mode=='worker_error' and len(sessions)==7:
                raise RuntimeError('test-only-worker-failure')
            return httpx.Response(200,json={'result':{'budget':{'model_requests':1}}})
        def factory(**kwargs):
            """保留真实httpx客户端的请求行为，只把网络传输替换为内存服务器。"""
            return client_type(**kwargs,transport=httpx.MockTransport(handler))
        async def login(client,username,password):
            if mode=='login_error': raise RuntimeError('test-only')
            return people[0 if username=='alice' else 1]
        def grade(*args):
            return mode not in {'performance','security'}, ['TEST_FAILURE'] if mode=='performance' else [], ['TEST_SECURITY'] if mode=='security' else []
        args=SimpleNamespace(ca_file=self.root/'unused.pem',users=['alice','bob'])
        with patch.object(bench,'ROOT',output_root),patch.object(bench,'simulated_cases',return_value=manifest), \
             patch.object(bench.ssl,'create_default_context',return_value=True), \
             patch.object(bench.httpx,'AsyncClient',side_effect=factory),patch.object(bench,'login',side_effect=login), \
             patch.object(bench,'judge',side_effect=grade),redirect_stdout(io.StringIO()):
            if mode.endswith('_error'):
                with self.assertRaises(Exception): await bench.run(args,['test-only','test-only'])
                code=2
            else: code=await bench.run(args,['test-only','test-only'])
        paths=list((output_root/'evaluation/agent_runtime/s6').glob('real-model-*.json'))
        self.assertEqual(len(paths),1)
        self.assertTrue(paths[0].with_suffix('.md').exists())
        return code,json.loads(paths[0].read_text(encoding='utf-8')),sent,sessions

    async def test_all_levels_and_120_requests(self):
        code,report,sent,sessions=await self.execute('normal')
        self.assertEqual(code,0)
        self.assertEqual(report['state'],'complete')
        self.assertEqual(report['requests_sent'],120)
        self.assertEqual(len(sent),120);self.assertEqual(len(sessions),7)
        for level in report['levels']:
            self.assertEqual({r['category'] for r in level['rows']},{'simple','multi'})
            self.assertEqual(len(level['rows']),40)

    async def test_performance_failure_continues_remaining_levels(self):
        code,report,sent,_=await self.execute('performance')
        self.assertEqual(code,1);self.assertFalse(report['passed'])
        self.assertEqual(len(sent),120);self.assertEqual(len(report['levels']),3)

    async def test_security_stops_and_partial_samples_fail(self):
        code,report,sent,_=await self.execute('security')
        self.assertEqual(code,1);self.assertFalse(report['passed'])
        self.assertEqual(report['state'],'stopped_security_failure')
        self.assertEqual(len(sent),1);self.assertEqual(len(report['levels']),1)

    async def test_login_and_session_errors_leave_incomplete_report(self):
        for mode in ['login_error','session_error']:
            _,report,sent,_=await self.execute(mode)
            self.assertEqual(report['state'],'incomplete')
            self.assertFalse(report['passed']);self.assertFalse(report['model_called'])
            self.assertEqual(len(sent),0)

    async def test_worker_failure_keeps_previous_levels_and_cancels_peers(self):
        _,report,sent,_=await self.execute('worker_error')
        self.assertEqual(report['state'],'incomplete');self.assertFalse(report['passed'])
        self.assertEqual(len(report['levels']),3)
        self.assertEqual(len(report['levels'][0]['rows']),40)
        self.assertEqual(len(report['levels'][1]['rows']),40)
        self.assertLess(len(sent),120)
        self.assertFalse(any(t.get_coro().__name__=='worker' for t in asyncio.all_tasks()))
