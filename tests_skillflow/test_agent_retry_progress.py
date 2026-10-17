"""真实失败回归：前置条件补齐后可再次调用，成功调用与无进展重复仍被限制。"""

import asyncio
import json
import unittest
from uuid import uuid4

from skillflow.agent.contracts import MessageEnvelope, ModelReply, ToolCall
from tests_skillflow import test_agent_loop as loop_fixtures
from tests_skillflow import test_agent_s6 as s6_fixtures


def install_out_of_order_model(model):
    """模型先提库存再提资格；观察拒绝后补齐资格，再使用新调用ID查库存。"""
    original=model.generate
    injected=[False]
    async def generate(messages,specs):
        calls={c.call_id:c for m in messages for c in m.tool_calls}
        observed={calls[m.tool_call_id].name:json.loads(m.content) for m in messages if m.role=='tool'}
        if (observed.get('assess_after_sales',{}).get('status')=='success'
                and observed.get('check_inventory',{}).get('status')=='rejected'):
            model.calls+=1
            oid=next(c.arguments['order_id'] for c in calls.values() if c.name=='get_order')
            return ModelReply(message=MessageEnvelope(role='assistant',content='',tool_calls=[
                ToolCall(call_id=uuid4().hex,name='check_inventory',arguments={'order_id':oid})]))
        reply=await original(messages,specs)
        assessment=next((c for c in reply.message.tool_calls if c.name=='assess_after_sales'),None)
        if assessment and not injected[0]:
            injected[0]=True
            premature=ToolCall(call_id=uuid4().hex,name='check_inventory',arguments={'order_id':assessment.arguments['order_id']})
            return reply.model_copy(update={'message':reply.message.model_copy(update={'tool_calls':[premature,*reply.message.tool_calls]})})
        return reply
    model.generate=generate


def assert_inventory_recovered(test,result,prior_events=()):
    """按业务结果断言：先拒绝错误顺序，后成功查询且库存仍为0；所有调用必须配对。"""
    test.assertEqual(result['status'],'completed',result)
    rows=result['result']['tool_records']
    inventory=[r['observation'] for r in rows if r['name']=='check_inventory']
    test.assertEqual([r['status'] for r in inventory],['rejected','success'])
    test.assertEqual(inventory[0]['error_code'],'PRECONDITION_MISSING')
    test.assertEqual(inventory[1]['data']['available'],0)
    test.assertIn('缺货',result['answer'])
    # 恢复结果包含整项任务的工具记录，而events仅为当前执行段；配对需合并前一段事件。
    proposed=[c['call_id'] for e in [*prior_events,*result['events']] if e['node']=='agent_model_decision'
        for c in e['details'].get('tool_calls',[])]
    test.assertEqual(sorted(proposed),sorted(r['observation']['tool_call_id'] for r in rows))


class S3ProgressTests(unittest.TestCase):
    """无检查点图也应遵守相同进展规则，不继承其他测试类整套用例。"""
    setUp=loop_fixtures.LoopTests.setUp
    tearDown=loop_fixtures.LoopTests.tearDown
    ask=loop_fixtures.LoopTests.ask

    def test_precondition_correction_in_native_loop(self):
        install_out_of_order_model(self.model)
        assert_inventory_recovered(self,self.ask('调查 A124 的退换资格'))


class RepeatBoundaryTests(unittest.TestCase):
    """权限、跨订单、失效证据与重复成功不能被其他工具的进展绕过。"""
    def test_progress_boundary_and_bounded_temporary_retry(self):
        from skillflow.agent.repeat_guard import duplicate_blocked
        call=ToolCall(call_id='new-call',name='check_inventory',arguments={'order_id':'A124'})
        def row(name,status='success',code=None,oid='A124',**extra):
            """测试观察保持生产记录形状；不提供实际业务授权。"""
            return {'name':name,'arguments':{'order_id':oid},
                'observation':{'status':status,'error_code':code,**extra}}
        required=('get_order','assess_after_sales')
        history=[row('get_order'),row('check_inventory','rejected','PRECONDITION_MISSING')]
        self.assertTrue(duplicate_blocked(call,history,required))
        self.assertFalse(duplicate_blocked(call,[*history,row('assess_after_sales')],required))
        self.assertTrue(duplicate_blocked(call,[*history,row('assess_after_sales',oid='A123')],required))
        self.assertTrue(duplicate_blocked(call,[*history,row('retrieve_policy')],required))
        invalidated=row('assess_after_sales');invalidated['invalidated']=True
        self.assertTrue(duplicate_blocked(call,[*history,invalidated],required))
        self.assertTrue(duplicate_blocked(call,[row('get_order'),row('check_inventory','rejected','ORDER_FORBIDDEN'),row('assess_after_sales')],required))
        self.assertTrue(duplicate_blocked(call,[row('check_inventory'),row('get_order'),row('assess_after_sales')],required))
        self.assertFalse(duplicate_blocked(call,[*history,row('check_inventory','rejected','DUPLICATE_NO_PROGRESS'),row('assess_after_sales')],required))
        temporary=row('check_inventory','failed','TOOL_TIMEOUT',retryable=True)
        self.assertFalse(duplicate_blocked(call,[temporary],required))
        self.assertTrue(duplicate_blocked(call,[temporary,temporary],required))


class S6ProgressTests(unittest.IsolatedAsyncioTestCase):
    """生产身份入口、实际MCP生命周期与Saver，模拟模型的错误顺序可稳定重现。"""
    asyncSetUp=s6_fixtures.IntegratedTests.asyncSetUp
    asyncTearDown=s6_fixtures.IntegratedTests.asyncTearDown
    oid=s6_fixtures.IntegratedTests.oid
    session=s6_fixtures.IntegratedTests.session
    ask=s6_fixtures.IntegratedTests.ask
    resume_task=s6_fixtures.IntegratedTests.resume_task

    async def test_precondition_correction_in_persistent_loop(self):
        install_out_of_order_model(self.model)
        result=await self.ask(self.alice,f"调查 {self.oid('no_stock')} 的退换资格")
        assert_inventory_recovered(self,result)

    async def test_two_turn_selection_and_corrected_tool_order(self):
        """先问无单号调查，再选择候选；新任务重新取证，错序调用可在预算内纠正。"""
        install_out_of_order_model(self.model)
        session=await self.session(self.alice)
        first=await self.ask(self.alice,'查询我的耳机订单，调查能否退换',session)
        self.assertEqual(first['status'],'clarify',first)
        self.assertIn(self.oid('no_stock'),first['answer'])
        self.assertNotIn(self.oid('normal','other-buyer'),first['answer'])
        self.assertFalse(any(r['name']=='get_order' for r in first['result']['tool_records']))
        second=await self.ask(self.alice,'选择 '+self.oid('no_stock'),session)
        assert_inventory_recovered(self,second)
        self.assertEqual(second['result']['task_goal']['continuation_of_task_id'],first['result']['metadata']['task_id'])
        self.assertNotEqual(second['result']['metadata']['task_id'],first['result']['metadata']['task_id'])

    async def test_receipt_replay_preserves_dependency_progress(self):
        """资格已产生回执但图未保存时中断，恢复后应承认进展，且不重放业务执行。"""
        install_out_of_order_model(self.model)
        triggered=[False]
        async def crash(point,state):
            rows=state.constraints.get('tool_records',[])
            if (point=='after_tool_receipt' and not triggered[0] and rows
                    and rows[-1]['name']=='check_inventory'
                    and rows[-1]['observation'].get('error_code')=='PRECONDITION_MISSING'):
                triggered[0]=True
                raise asyncio.CancelledError()
        self.app.state.agent_service.failpoint=crash
        first=await self.ask(self.alice,f"调查 {self.oid('no_stock')} 的退换资格")
        self.assertEqual(first['status'],'interrupted',first)
        self.app.state.agent_service.failpoint=None
        second=await self.resume_task(first,'continue')
        assert_inventory_recovered(self,second,first['events'])
        self.assertEqual(second['result']['task']['task_id'],first['result']['task']['task_id'])
        self.assertTrue(any(e['node']=='agent_tool_receipt_replay' for e in second['events']))
