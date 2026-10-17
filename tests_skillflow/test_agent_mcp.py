"""S5：真实 stdio 子进程与 Adapter、模型替身、实际图检查点；不请求外部模型。"""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import unittest
from uuid import uuid4

from fastapi.testclient import TestClient

from skillflow.agent.adapters.mcp_session import McpSessionManager, McpFailure
from skillflow.agent.adapters.mcp_tools import LogisticsBusinessAdapter
from skillflow.agent.contracts import MessageEnvelope, ModelReply, ToolCall
from skillflow.agent.context import checked_context
from skillflow.api import create_app
from skillflow.config import ROOT
from tests_skillflow import test_agent_loop as fixtures
from tests_skillflow import test_agent_resume as resume_fixtures


class LogisticsModel(fixtures.TestModel):
    """从原生观察选择物流调用，保留错误后一次显式重试；不替代生产模型。"""
    async def generate(self, messages, tool_specs):
        self.calls += 1
        goal = json.loads(messages[0].content.split('固定目标：')[1].split('\n可加载技能目录：')[0])
        calls = {c.call_id: c for m in messages for c in m.tool_calls}
        observed = [(i, calls[m.tool_call_id].name, json.loads(m.content))
                    for i, m in enumerate(messages) if m.role == 'tool']
        prior = {name: result for _, name, result in observed}
        oid = goal['scope']['explicit_order_ids'][0]
        if 'load_skill' not in prior:
            name, args = 'load_skill', {'name': 'logistics_tracking', 'reason': '查询订单配送物流'}
        elif 'get_order' not in prior:
            name, args = 'get_order', {'order_id': oid}
        elif prior['get_order']['status'] != 'success':
            name, args = None, {}
        else:
            notice = max([i for i, m in enumerate(messages) if '前一执行段物流观察' in m.content] or [-1])
            logistics = [r for i, n, r in observed if n == 'get_logistics' and i > notice]
            if not logistics or (len(logistics) < 2 and logistics[-1]['retryable']):
                name, args = 'get_logistics', {'order_id': oid}
            else:
                name, args = None, {}
        return ModelReply(message=MessageEnvelope(role='assistant', content='' if name else '调查结束',
            tool_calls=[ToolCall(call_id=uuid4().hex, name=name, arguments=args)] if name else []))


class McpProtocolTests(unittest.IsolatedAsyncioTestCase):
    """每项通过 SDK 建立真实服务子进程；不把内存函数调用称为 MCP 证据。"""
    async def asyncSetUp(self):
        self.manager = McpSessionManager(ROOT / 'data/logistics/delivery.json')
        self.addAsyncCleanup(self.manager.aclose)

    def fault(self, mode, **kwargs):
        """故障服务单独位于 tests，正式服务不接受这些开关。"""
        self.manager.command = [sys.executable, '-B', '-m', 'tests_skillflow.mcp_fault_server', '--mode', mode]
        for k, v in kwargs.items():
            setattr(self.manager, k, v)

    def adapter(self, buyer='demo-buyer'):
        """协议测试的身份来自可信测试上下文，不从模型参数取得。"""
        return LogisticsBusinessAdapter(type('BoundLocal', (), {'buyer_id': buyer})(), self.manager)

    async def test_discovery_conversion_normal_empty_and_server_ownership(self):
        await self.manager.start()
        self.assertEqual(self.manager.health()['status'], 'ready')
        found = await self.adapter().invoke('get_logistics', {'order_id': 'A123'})
        empty = await self.adapter().invoke('get_logistics', {'order_id': 'A125'})
        denied = await self.adapter().invoke('get_logistics', {'order_id': 'A900'})
        self.assertEqual(found.data['mcp_tool'], 'logistics_get_logistics')
        self.assertEqual(found.data['logistics_status'], 'DELIVERED')
        self.assertEqual(empty.status, 'empty')
        self.assertEqual(denied.error_code, 'MCP_ACCESS_DENIED')
        self.assertFalse(denied.data)
        self.assertEqual(self.manager.starts, 1)

    async def test_single_timeout_does_not_close_other_query(self):
        self.fault('slow')
        await self.manager.start()
        self.manager.query_timeout = .2
        slow, fast = await asyncio.gather(*(self.adapter().invoke('get_logistics', {'order_id': oid})
                                          for oid in ['A123', 'A124']))
        self.assertEqual(slow.error_code, 'MCP_TIMEOUT')
        self.assertTrue(slow.retryable)
        self.assertEqual(fast.status, 'success')
        await asyncio.sleep(.65)  # 让迟到响应完成，验证它不会污染下一查询。
        self.assertEqual((await self.adapter().invoke('get_logistics', {'order_id': 'A124'})).status, 'success')
        self.assertEqual(self.manager.starts, 1)

    async def test_child_exit_is_not_implicitly_replayed_and_next_query_reconnects_once(self):
        self.fault('exit')
        await self.manager.start()
        failed = await self.adapter().invoke('get_logistics', {'order_id': 'A123'})
        self.assertEqual(failed.error_code, 'MCP_UNAVAILABLE')
        self.assertEqual(self.manager.calls, 1)
        self.manager.command = [sys.executable, '-B', '-m', 'skillflow.integrations.logistics_mcp.server']
        results = await asyncio.gather(*(self.adapter().invoke('get_logistics', {'order_id': 'A123'}) for _ in range(5)))
        self.assertTrue(all(r.status == 'success' for r in results))
        self.assertEqual(self.manager.starts, 2)

    async def test_startup_failure_and_cooldown_do_not_fork_per_request(self):
        self.fault('startup_exit')
        await self.manager.start()
        self.assertEqual(self.manager.health()['status'], 'unavailable')
        results = await asyncio.gather(*(self.adapter().invoke('get_logistics', {'order_id': 'A123'}) for _ in range(5)))
        self.assertTrue(all(r.error_code == 'MCP_UNAVAILABLE' for r in results))
        self.assertEqual(self.manager.starts, 1)
        self.assertGreater(self.manager._next_attempt - time.monotonic(), 8)

    async def test_schema_drift_is_rejected_before_any_tool_execution(self):
        self.fault('schema')
        await self.manager.start()
        self.assertEqual(self.manager.last_error, 'MCP_SCHEMA_INVALID')
        result = await self.adapter().invoke('get_logistics', {'order_id': 'A123'})
        self.assertEqual(result.error_code, 'MCP_SCHEMA_INVALID')
        self.assertFalse(result.retryable)
        self.assertEqual(self.manager.calls, 0)

    async def test_malformed_response_and_wrong_order_are_not_retryable(self):
        for mode in ['malformed', 'wrong_order']:
            manager = McpSessionManager(ROOT / 'data/logistics/delivery.json', command=[sys.executable,
                '-B', '-m', 'tests_skillflow.mcp_fault_server', '--mode', mode])
            try:
                await manager.start()
                adapter = LogisticsBusinessAdapter(type('Local', (), {'buyer_id': 'demo-buyer'})(), manager)
                result = await adapter.invoke('get_logistics', {'order_id': 'A123'})
                self.assertEqual(result.error_code, 'MCP_INVALID_RESPONSE')
                self.assertFalse(result.retryable)
                self.assertEqual(manager.calls, 1)
            finally:
                await manager.aclose()

    async def test_closed_manager_cannot_spawn_another_child(self):
        await self.manager.start()
        await self.manager.aclose()
        result = await self.adapter().invoke('get_logistics', {'order_id': 'A123'})
        self.assertEqual(result.error_code, 'MCP_UNAVAILABLE')
        self.assertEqual(self.manager.starts, 1)

    async def test_signature_scope_expiry_and_model_injection(self):
        from skillflow.integrations.logistics_mcp.contracts import grant, verify_grant
        import jwt
        token = grant(self.manager.secret, 'demo-buyer', 'A123')
        self.assertEqual(verify_grant(self.manager.secret, token, 'A123'), 'demo-buyer')
        for bad, order in [(token, 'A124'), (token + 'bad', 'A123'),
                           (jwt.encode({'sub': 'demo-buyer', 'order': 'A123', 'tool': 'get_logistics',
                              'iat': 1, 'exp': 2, 'aud': 'skillflow-logistics'}, self.manager.secret, algorithm='HS256'), 'A123')]:
            with self.assertRaisesRegex(ValueError, 'MCP_ACCESS_DENIED'):
                verify_grant(self.manager.secret, bad, order)
        await self.manager.start()
        # 直接调用转换工具而不经可信调用上下文，仍应被拦截器拒绝。
        with self.assertRaises(McpFailure):
            await self.manager._generation.tool.ainvoke({'order_id': 'A123', 'access_grant': token})
        self.assertEqual(self.manager.calls, 0)

    async def test_server_itself_rejects_unsigned_and_wrong_order_grants(self):
        """直接通过真实 SDK 发坏凭据，证明服务端也会校验，而不只依赖客户端。"""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from skillflow.integrations.logistics_mcp.contracts import grant
        import os
        env = {k: os.environ[k] for k in ('SYSTEMROOT', 'PATH', 'TEMP', 'TMP') if k in os.environ}
        env.update(SKILLFLOW_LOGISTICS_SECRET=self.manager.secret,
                   SKILLFLOW_LOGISTICS_DATA=str(ROOT / 'data/logistics/delivery.json'), PYTHONIOENCODING='utf-8')
        params = StdioServerParameters(command=sys.executable,
            args=['-B', '-m', 'skillflow.integrations.logistics_mcp.server'], env=env, cwd=str(ROOT))
        with open(os.devnull, 'w') as errors:
            async with stdio_client(params, errlog=errors) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    for token in ['', 'forged', grant(self.manager.secret, 'demo-buyer', 'A124')]:
                        result = await session.call_tool('get_logistics', {'order_id': 'A123', 'access_grant': token})
                        self.assertTrue(result.isError)
                        self.assertNotIn('演示配送站', str(result))


class McpAgentTests(unittest.TestCase):
    """网页/API 同一链路，脚本模型只提出动作，实际工具通过 MCP 执行。"""
    ask = fixtures.LoopTests.ask
    body = resume_fixtures.ResumeTests.body
    resume = resume_fixtures.ResumeTests.resume

    def setUp(self):
        """沿用隔离测试目录，显式开启 lifespan 保证连接归属同一事件循环。"""
        fixtures.LoopTests.setUp(self)
        self.client.close()
        self.app.state.repository.engine.dispose()
        self.model = LogisticsModel()
        import skillflow.agent.adapters.chat_model as gateway
        gateway.ChatModelGateway.return_value = self.model
        self.settings = replace(self.settings, agent_s4_enabled=True, agent_s5_enabled=True)
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        """先退出 lifespan 释放受管子进程，才清理本项业务和图数据目录。"""
        self.client.__exit__(None, None, None)
        fixtures.LoopTests.tearDown(self)

    def test_agent_native_observation_and_trusted_rendering(self):
        result = self.ask('查询 A123 的配送物流')
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['result']['stage'], 'S5')
        self.assertIn('已签收', result['answer'])
        self.assertIn('来源更新时间', result['answer'])
        self.assertEqual(result['result']['loaded_skills'], {'logistics_tracking': '1.0-s5.1'})
        rows = result['result']['tool_records']
        self.assertEqual([r['name'] for r in rows], ['load_skill', 'get_order', 'get_logistics'])
        state = self.app.state.store.agent_run(result['run_id'])['state']
        checked_context([MessageEnvelope.model_validate(m) for m in state['messages']], [], 100000)
        serialized = json.dumps(state)
        self.assertNotIn('access_grant', serialized)
        self.assertNotIn(self.app.state.agent_service.mcp_manager.secret, serialized)
        self.assertTrue(result['result']['task']['checkpoint_id'])

    def test_no_record_is_completed_and_short_followup_inherits_logistics(self):
        self.ask('查询 A123 的配送物流')
        result = self.ask('A125')
        self.assertEqual(result['status'], 'completed', result)
        self.assertIn('暂无配送物流记录', result['answer'])
        self.assertIn('logistics', result['result']['task_goal']['requirements'])

    def test_foreign_order_is_blocked_before_mcp(self):
        result = self.ask('查询 A900 的配送物流')
        self.assertNotEqual(result['status'], 'completed')
        self.assertEqual(result['result']['stop_reason'], 'ORDER_FORBIDDEN')
        self.assertIn('无权访问', result['answer'])
        self.assertEqual(self.app.state.agent_service.mcp_manager.calls, 0)
        self.assertTrue(any(r['observation']['error_code'] == 'ORDER_FORBIDDEN' for r in result['result']['tool_records']))

    def test_sse_replay_and_health_do_not_reexecute_mcp(self):
        from tests_skillflow.test_streaming import frames
        job = {'session_id': 's5-stream', 'run_id': str(uuid4()), 'message': '查询 A123 配送物流'}
        first = frames(self.client.post('/chat/stream', json=job).text)
        manager = self.app.state.agent_service.mcp_manager
        count = manager.calls
        self.assertEqual(frames(self.client.post('/chat/stream', json=job).text), first)
        self.client.get('/health')
        self.client.get('/runs/' + job['run_id'])
        self.assertEqual(manager.calls, count)

    def test_interruption_replays_old_receipt_as_history_then_refreshes(self):
        """覆盖物流回执领先于检查点；旧调用配对保留，但不能当成恢复后的查询。"""
        service = self.app.state.agent_service
        async def crash(point, state):
            if point == 'after_tool_receipt' and service.mcp_manager.calls == 1:
                raise asyncio.CancelledError()
        service.failpoint = crash
        first = self.ask('查询 A123 配送物流')
        self.assertEqual(first['status'], 'interrupted', first)
        service.failpoint = None
        second = self.resume(first, 'continue')
        self.assertEqual(second['status'], 'completed', second)
        self.assertEqual(service.mcp_manager.calls, 2)
        rows = [r for r in second['result']['tool_records'] if r['name'] == 'get_logistics']
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]['invalidated'])
        self.assertFalse(rows[1].get('invalidated', False))
        self.assertEqual(rows[1]['observation']['data']['execution_run_id'], second['run_id'])
        self.assertGreater(second['result']['budget']['tool_attempts'], first['result']['budget']['tool_attempts'])

    def test_unavailable_mcp_keeps_order_facts_and_allows_only_one_model_retry(self):
        service = self.app.state.agent_service
        async def break_connection():
            await service.mcp_manager.aclose()
        self.client.portal.call(break_connection)
        result = self.ask('查询 A123 配送物流')
        self.assertEqual(result['status'], 'needs_manual_review', result)
        self.assertEqual(result['result']['stop_reason'], 'MCP_UNAVAILABLE')
        self.assertIn('演示蓝牙耳机', result['answer'])
        self.assertIn('物流查询未完成', result['answer'])
        self.assertEqual(len([r for r in result['result']['tool_records'] if r['name'] == 'get_logistics']), 2)
        self.assertEqual(self.client.get('/health').json()['status'], 'degraded')

    def test_logistics_public_schema_and_injected_buyer_are_rejected(self):
        """模型工具面只包含单号，未知参数由执行器拒绝，不能进入 MCP。"""
        service = self.app.state.agent_service
        spec = service.tools.registry.resolve('get_logistics').model_spec()
        self.assertEqual(set(spec.parameters['properties']), {'order_id'})
        original = self.model.generate
        async def inject(messages, specs):
            reply = await original(messages, specs)
            for call in reply.message.tool_calls:
                if call.name == 'get_logistics':
                    call.arguments['buyer_id'] = 'other-buyer'
            return reply
        self.model.generate = inject
        result = self.ask('查询 A123 配送物流')
        self.assertNotEqual(result['status'], 'completed')
        self.assertEqual(service.mcp_manager.calls, 0)
        self.assertTrue(any(r['observation']['error_code'] == 'INVALID_TOOL_ARGUMENTS' for r in result['result']['tool_records']))

    def test_resume_cannot_refresh_with_exhausted_tool_budget(self):
        """旧物流仅保留审计，预算不足不能复用它来宣称新的查询已完成。"""
        from skillflow.agent.config import BudgetLimits
        self.settings = replace(self.settings, agent_limits=BudgetLimits(max_tool_attempts=3))
        self.client.__exit__(None, None, None)
        self.app.state.repository.engine.dispose()
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)
        self.client.__enter__()
        service = self.app.state.agent_service
        async def crash(point, state):
            if point == 'after_tool_receipt' and service.mcp_manager.calls == 1:
                raise asyncio.CancelledError()
        service.failpoint = crash
        first = self.ask('查询 A123 配送物流')
        self.assertEqual(first['status'], 'interrupted', first)
        service.failpoint = None
        second = self.resume(first, 'continue')
        self.assertEqual(second['result']['stop_reason'], 'TOOL_LIMIT', second)
        self.assertEqual(second['status'], 'failed')
        self.assertEqual(service.mcp_manager.calls, 1)
        self.assertTrue(all(r.get('invalidated') for r in second['result']['tool_records'] if r['name'] == 'get_logistics' and r['observation']['status'] == 'success'))
