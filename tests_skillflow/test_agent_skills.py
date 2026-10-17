"""S2 独立验收：实际本地工具、技能规范、范围、证据、版本、故障及分页。"""

import asyncio
from contextlib import closing
import json
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from sqlalchemy import event

from evaluation.agent_runtime.s2_fixtures import seed_quality_demo
from skillflow.agent.bootstrap import create_tool_runtime
from skillflow.agent.config import BudgetLimits
from skillflow.agent.contracts import TaskMetadata, ToolCall
from skillflow.agent.skills import FileSkillRegistry, SKILL_ROOT, SkillError
from skillflow.agent.tool_executor import TargetScope
from skillflow.business.policies import LocalPolicyRepository
from skillflow.business.repository import MockOrderRepository
from skillflow.business.surge import BusinessRepository
from skillflow.config import ROOT
from skillflow.sqlite_store import SqliteStore


class SkillToolTests(unittest.IsolatedAsyncioTestCase):
    """每例使用 B 内独立 SQLite 和政策副本，不请求外部 LLM 或 MCP。"""

    async def asyncSetUp(self):
        self.base = (ROOT / ".skillflow/agent-s2-tests").resolve()
        self.root = self.base / uuid4().hex
        self.fixtures = self.root / "fixtures"
        self.fixtures.mkdir(parents=True)
        for name in ("buyers.json", "orders.json", "inventory.json", "price_adjustments.json"):
            shutil.copyfile(ROOT / "data" / name, self.fixtures / name)
        shutil.copytree(ROOT / "data/policies", self.fixtures / "policies")
        self.store = SqliteStore(self.root / "state")
        self.policies = LocalPolicyRepository(self.fixtures / "policies")
        self.repository = MockOrderRepository(self.fixtures)
        self.runtime = create_tool_runtime(self.store, self.repository, self.policies)
        self.engines = []

    async def asyncTearDown(self):
        for engine in self.engines:
            engine.dispose()
        # 路径核验后仅删除本例创建的隔离目录。
        if self.root.resolve().is_relative_to(self.base) and self.root != self.base:
            shutil.rmtree(self.root)

    async def session(self, order_id="A123", *, runtime=None, scope=None, limits=None):
        """模拟服务端首次登记任务和可信目标，不把这些字段提供给模型。"""
        metadata = TaskMetadata(task_id=uuid4().hex, run_id=str(uuid4()), buyer_id="demo-buyer",
            session_id=uuid4().hex, engine="tool_loop", engine_version="s2_tools_v1", graph_version="s2_diagnostic_v1",
            model_mode="scripted", limits=limits or BudgetLimits())
        self.store.register_run(metadata.run_id, metadata.buyer_id, metadata.session_id, metadata.model_dump(mode="json"))
        return await (runtime or self.runtime).open_task(metadata.run_id, metadata.buyer_id, metadata.session_id,
                                                        scope or TargetScope(explicit_order_ids=(order_id,)))

    async def call(self, session, tool_name, **arguments):
        """每次模拟下一轮模型工具面；旧工具面测试直接调用 dispatch。"""
        return await session.dispatch(ToolCall(call_id=uuid4().hex, name=tool_name, arguments=arguments), session.begin_turn(uuid4().hex))

    async def load(self, session, name="return_exchange"):
        return await self.call(session, "load_skill", name=name, reason="执行固定验收任务")

    async def policy(self, session, order_id="A123"):
        await self.call(session, "get_order", order_id=order_id)
        result = await self.call(session, "retrieve_policy", order_id=order_id, query="退换货资格 价保 优惠 退款依据")
        return [item["ref"] for item in result.data.get("evidence", [])]

    async def test_catalog_lazy_loading_pin_and_repeat(self):
        session = await self.session()
        self.assertEqual([spec.name for spec in session.offered_tools()], ["load_skill"])
        catalog = await self.runtime.skills.list_catalog()
        self.assertEqual(len(catalog), 6)
        self.assertFalse(any("instructions" in item for item in catalog))
        result = await self.load(session)
        self.assertEqual(result.status, "success")
        self.assertIn("退换资格", result.data["instructions"])
        self.assertNotIn("submit_ticket", result.data["opened_tools"])
        self.assertEqual(result.data["unavailable_tools"]["submit_ticket"], "WRITE_NOT_ENABLED")
        duplicate = await self.load(session)
        self.assertTrue(duplicate.data["already_loaded"])
        self.assertNotIn("instructions", duplicate.data)
        other = await self.session()
        self.assertEqual([spec.name for spec in other.offered_tools()], ["load_skill"])
        pinned = session.loaded["return_exchange"]
        self.assertEqual(await self.runtime.skills.get_pinned(pinned.name, pinned.version, pinned.content_hash), pinned)
        with self.assertRaises(SkillError):
            await self.runtime.skills.get_pinned(pinned.name, pinned.version, "wrong")
        session.loaded["return_exchange"].tool_names.append("shell")
        self.assertNotIn("shell", session.loaded["return_exchange"].tool_names)

    async def test_same_response_cannot_gain_new_tools(self):
        session = await self.session()
        surface = session.begin_turn("model-turn-1")
        loaded = await session.dispatch(ToolCall(call_id="load", name="load_skill",
            arguments={"name": "order_tracking", "reason": "查询"}), surface)
        self.assertEqual(loaded.status, "success")
        result = await session.dispatch(ToolCall(call_id="order", name="get_order", arguments={"order_id": "A123"}), surface)
        self.assertEqual(result.error_code, "TOOL_NOT_OFFERED")
        self.assertEqual((await self.call(session, "get_order", order_id="A123")).status, "success")

    async def test_public_schemas_and_injection_rejections(self):
        session = await self.session()
        self.assertEqual((await self.call(session, "get_order", order_id="A123")).error_code, "SKILL_NOT_LOADED")
        await self.load(session)
        schemas = {tool.name: tool.parameters for tool in session.offered_tools()}
        self.assertIn("evidence_refs", schemas["assess_after_sales"]["properties"])
        for schema in schemas.values():
            self.assertFalse(schema["additionalProperties"])
            self.assertTrue({"buyer_id", "confirmed", "evidence", "policy_key", "sku"}.isdisjoint(schema["properties"]))
        for arguments in ({"order_id": "A123", "buyer_id": "other-buyer"},
                          {"order_id": "A123", "evidence": []}, {"order_id": "A123", "confirmed": True}):
            result = await self.call(session, "get_order", **arguments)
            self.assertEqual(result.error_code, "INVALID_TOOL_ARGUMENTS")
        self.assertEqual((await self.call(session, "shell", command="dir")).error_code, "UNKNOWN_TOOL")

    async def test_a123_a124_a125_a126_real_business_results(self):
        for order_id in ("A123", "A124", "A125", "A126"):
            with self.subTest(order=order_id):
                session = await self.session(order_id)
                await self.load(session)
                refs = await self.policy(session, order_id)
                if order_id == "A125":
                    self.assertFalse(refs)
                    self.assertEqual(session.observations[-1].error_code, "POLICY_NOT_REGISTERED")
                    self.assertFalse(any(session.completion_status("return_exchange", order_id).values()))
                    continue
                assessed = await self.call(session, "assess_after_sales", order_id=order_id, evidence_refs=refs)
                self.assertEqual(assessed.status, "success")
                self.assertEqual(assessed.data["return_eligible"], order_id != "A126")
                stock = await self.call(session, "check_inventory", order_id=order_id)
                self.assertEqual(stock.data["available"] > 0, order_id != "A124")
                self.assertTrue(all(session.completion_status("return_exchange", order_id).values()))
                with closing(self.store.connect()) as conn:
                    self.assertEqual(conn.execute("SELECT count(*) FROM tickets").fetchone()[0], 0)

    async def test_preconditions_and_target_ownership(self):
        session = await self.session()
        await self.load(session)
        self.assertEqual((await self.call(session, "check_inventory", order_id="A123")).error_code, "PRECONDITION_MISSING")
        self.assertEqual((await self.call(session, "get_order", order_id="A124")).error_code, "TARGET_NOT_ALLOWED")
        result = await self.call(session, "get_order", order_id="A900")
        self.assertEqual(result.error_code, "ORDER_FORBIDDEN")
        self.assertEqual(result.data, {})
        unknown = await self.session("A999")
        await self.load(unknown, "order_tracking")
        self.assertEqual((await self.call(unknown, "get_order", order_id="A999")).error_code, "ORDER_NOT_FOUND")

    async def test_discount_and_price_skills_use_bound_policy_evidence(self):
        """两种金额工具也经同一引用绑定边界，不只验证退换工具。"""
        for skill_name, tool_name in (("coupon_dispute", "analyze_discount"), ("price_adjustment", "analyze_price_adjustment")):
            session = await self.session()
            await self.load(session, skill_name)
            refs = await self.policy(session)
            result = await self.call(session, tool_name, order_id="A123", evidence_refs=refs)
            self.assertEqual(result.status, "success")
            self.assertTrue(all(session.completion_status(skill_name, "A123").values()))
            if tool_name == "analyze_discount":
                self.assertEqual(result.data["paid_cents"], 18900)
                self.assertTrue(result.data["consistent"])
            else:
                self.assertTrue(result.data["eligible"])
                self.assertGreater(result.data["difference_cents"], 0)

    async def test_independent_tasks_run_local_io_in_parallel(self):
        """仓库读取在线程中运行，任务不共享证据和计数，也不阻塞其他协程。"""
        first, second = await self.session(), await self.session("A124")
        await self.load(first, "order_tracking")
        await self.load(second, "order_tracking")
        active, peak = [0], [0]
        async def observed(*args):
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            try:
                await asyncio.sleep(.03)
                from skillflow.contracts import ToolResult
                return ToolResult(status="success", data={"order": {"order_id": args[1]["order_id"]}})
            finally:
                active[0] -= 1
        with patch.object(first.adapter, "invoke", side_effect=observed), patch.object(second.adapter, "invoke", side_effect=observed):
            results = await asyncio.gather(self.call(first, "get_order", order_id="A123"),
                                           self.call(second, "get_order", order_id="A124"))
        self.assertEqual(peak[0], 2)
        self.assertTrue(all(result.status == "success" for result in results))
        self.assertEqual([first.budget.snapshot().tool_attempts, second.budget.snapshot().tool_attempts], [2, 2])

    async def test_evidence_cannot_cross_task_or_order_and_results_are_copied(self):
        first = await self.session("A123")
        await self.load(first)
        refs = await self.policy(first)
        second = await self.session("A124")
        await self.load(second)
        own_refs = await self.policy(second, "A124")
        rejected = await self.call(second, "assess_after_sales", order_id="A124", evidence_refs=refs)
        self.assertEqual(rejected.error_code, "EVIDENCE_NOT_FOUND")
        good = await self.call(second, "assess_after_sales", order_id="A124", evidence_refs=own_refs)
        self.assertEqual(good.status, "success")
        good.data["return_eligible"] = False
        self.assertTrue(second.observations[-1].data["return_eligible"])
        fake = await self.call(second, "assess_after_sales", order_id="A124", evidence_refs=["fabricated"])
        self.assertEqual(fake.error_code, "EVIDENCE_NOT_FOUND")

    async def test_policy_reload_invalidates_old_references(self):
        session = await self.session()
        await self.load(session)
        old_refs = await self.policy(session)
        path = self.fixtures / "policies/earbuds_v1.md"
        path.write_text(path.read_text(encoding="utf-8") + "\n\n新的政策版本说明。", encoding="utf-8")
        self.policies.reload()
        self.assertEqual((await self.call(session, "assess_after_sales", order_id="A123", evidence_refs=old_refs)).error_code,
                         "SOURCE_VERSION_CHANGED")
        new_refs = await self.policy(session)
        self.assertEqual((await self.call(session, "assess_after_sales", order_id="A123", evidence_refs=old_refs)).error_code,
                         "EVIDENCE_STALE")
        self.assertEqual((await self.call(session, "assess_after_sales", order_id="A123", evidence_refs=new_refs)).status, "success")

    async def test_duplicate_call_receipt_and_budget(self):
        session = await self.session()
        await self.load(session, "order_tracking")
        surface = session.begin_turn("model-call")
        call = ToolCall(call_id="same-id", name="get_order", arguments={"order_id": "A123"})
        with patch.object(session.adapter, "invoke", wraps=session.adapter.invoke) as invoke:
            first = await session.dispatch(call, surface)
            second = await session.dispatch(call, surface)
            self.assertEqual(invoke.call_count, 1)
        self.assertEqual(first, second)
        self.assertEqual(session.budget.snapshot().tool_attempts, 2)
        changed = await session.dispatch(ToolCall(call_id="same-id", name="get_order", arguments={"order_id": "A124"}), surface)
        self.assertEqual(changed.error_code, "TOOL_CALL_CONFLICT")
        self.runtime.close_task(session.metadata.run_id)
        with self.assertRaises(ValueError):
            await self.runtime.open_task(session.metadata.run_id, "demo-buyer", session.metadata.session_id, session.scope)

    async def test_write_and_mcp_remain_unavailable(self):
        session = await self.session()
        await self.load(session)
        self.assertEqual((await self.call(session, "create_ticket_draft", order_id="A123", request_type="return", reason="申请")).error_code,
                         "WRITE_NOT_ENABLED")
        self.assertEqual((await self.call(session, "submit_ticket", ticket_id="fake")).error_code, "WRITE_NOT_ENABLED")
        await self.load(session, "logistics_tracking")
        self.assertEqual((await self.call(session, "get_logistics", order_id="A123")).error_code, "MCP_NOT_READY")

    async def test_unique_discovery_and_multi_candidate_clarification(self):
        unique = await self.session(scope=TargetScope(keyword="手柄"))
        await self.load(unique, "order_tracking")
        page = await self.call(unique, "list_my_orders", keyword="手柄")
        self.assertEqual(page.data["candidate_count"], 1)
        self.assertEqual((await self.call(unique, "get_order", order_id="A125")).status, "success")
        many = await self.session(scope=TargetScope(keyword="耳机"))
        await self.load(many, "order_tracking")
        page = await self.call(many, "list_my_orders", keyword="耳机", page_size=1)
        self.assertTrue(page.data["has_more"])
        self.assertEqual((await self.call(many, "get_order", order_id="A123")).error_code, "CANDIDATE_SCAN_INCOMPLETE")
        self.assertEqual((await self.call(many, "list_my_orders", keyword="耳机", after_order_id="A126")).error_code, "PAGINATION_CURSOR_INVALID")
        page = await self.call(many, "list_my_orders", keyword="耳机", after_order_id=page.data["next_after_order_id"])
        self.assertTrue(page.data["needs_clarification"])
        self.assertEqual((await self.call(many, "get_order", order_id="A123")).error_code, "CLARIFICATION_REQUIRED")
        self.assertEqual((await self.call(many, "list_my_orders", keyword="手柄")).error_code, "TARGET_QUERY_MISMATCH")

    async def test_sql_and_mock_pagination_match_without_n_plus_one(self):
        sql = BusinessRepository(self.store.database_path, self.fixtures)
        self.engines.append(sql.engine)
        statements = []
        def count_statement(*args):
            statements.append(args[2])
        event.listen(sql.engine, "before_cursor_execute", count_statement)
        for keyword in ("", "耳机", "%", "A12"):
            cursor = None
            while True:
                before = len(statements)
                page = sql.query_orders("demo-buyer", keyword, cursor, 2)
                self.assertEqual(len(statements) - before, 1)
                self.assertEqual(page, self.repository.query_orders("demo-buyer", keyword, cursor, 2))
                self.assertTrue(all(item["order_id"] not in {"A900", "A901"} for item in page["items"]))
                if not page["has_more"]:
                    break
                cursor = page["next_after_order_id"]

    async def test_quality_batch_three_states_and_version_change(self):
        sql = BusinessRepository(self.store.database_path, self.fixtures)
        self.engines.append(sql.engine)
        seed_quality_demo(sql)
        # 新闻背景经常使用品牌/型号寻找订单，不能只匹配商品名称。
        self.assertEqual(len(sql.query_orders("demo-buyer", "模拟品牌")["items"]), 4)
        self.assertEqual(len(sql.query_orders("demo-buyer", "S2-A")["items"]), 4)
        runtime = create_tool_runtime(self.store, sql, self.policies)
        scope = TargetScope(keyword="S2 模拟充电宝", quality_batch=True)
        session = await self.session(runtime=runtime, scope=scope)
        await self.load(session, "quality_incident")
        page = await self.call(session, "list_my_orders", keyword=scope.keyword)
        self.assertEqual(len(page.data["items"]), 4)
        decisions = []
        for order_id in ("A700", "A701", "A702", "A704"):
            await self.call(session, "get_order", order_id=order_id)
            result = await self.call(session, "match_quality_incident", order_id=order_id)
            decisions.append(result.data["decision"])
        self.assertEqual(decisions, ["AFFECTED", "NOT_AFFECTED", "UNVERIFIED", "AFFECTED"])
        sql.change_incident("S2-INCIDENT", actor="s2-reviewer", action="withdraw", reason="模拟撤回")
        self.assertEqual((await self.call(session, "list_alternatives", order_id="A700")).error_code, "SOURCE_VERSION_CHANGED")

    async def test_timeout_and_failed_attempts_consume_budget(self):
        session = await self.session(limits=BudgetLimits(max_tool_attempts=2))
        await self.load(session, "order_tracking")
        async def slow(*args):
            await asyncio.sleep(.1)
        session.tool_timeout = .01
        with patch.object(session.adapter, "invoke", side_effect=slow):
            result = await self.call(session, "get_order", order_id="A123")
        self.assertEqual(result.error_code, "TOOL_TIMEOUT")
        self.assertEqual(session.budget.snapshot().tool_attempts, 2)
        self.assertEqual((await self.call(session, "get_order", order_id="A123")).error_code, "TOOL_LIMIT")

    async def test_manifest_path_unknown_condition_and_content_version(self):
        root = self.root / "skills"
        shutil.copytree(SKILL_ROOT / "agent", root / "agent")
        shutil.copytree(SKILL_ROOT / "instructions", root / "instructions")
        original = FileSkillRegistry(self.runtime.registry.names, root)
        pinned = await original.load("return_exchange")
        sop = root / "instructions/return_exchange.md"
        sop.write_text(sop.read_text(encoding="utf-8") + "\n变更说明。", encoding="utf-8")
        changed = FileSkillRegistry(self.runtime.registry.names, root)
        self.assertEqual((await original.load("return_exchange")).content_hash, pinned.content_hash)
        with self.assertRaises(SkillError):
            await changed.get_pinned(pinned.name, pinned.version, pinned.content_hash)
        manifest = root / "agent/return_exchange.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for update in ({"instructions_path": "../../../outside.md"},
                       {"completion_conditions": ["eval_arbitrary_code"]}, {"allowed_tools": ["shell"]}):
            manifest.write_text(json.dumps({**data, **update}), encoding="utf-8")
            with self.assertRaises(SkillError):
                FileSkillRegistry(self.runtime.registry.names, root)

    async def test_trace_records_loaded_sop_version_tools_and_owned_task(self):
        session = await self.session()
        await self.load(session, "order_tracking")
        await self.call(session, "get_order", order_id="A123")
        events = self.store.agent_events(session.metadata.run_id)
        loaded = next(event for event in events if event["node"] == "load_skill")
        self.assertIn("instructions", loaded["details"])
        self.assertIn("content_hash", loaded["details"])
        self.assertEqual(set(loaded["details"]["opened_tools"]), {"load_skill", "list_my_orders", "get_order"})
        self.assertTrue(all(event["metadata"]["task_id"] == session.metadata.task_id for event in events))
        with self.assertRaises(PermissionError):
            await self.runtime.open_task(session.metadata.run_id, "other-buyer", session.metadata.session_id, session.scope)
        with self.assertRaises(ValueError):
            await self.runtime.open_task(session.metadata.run_id, "demo-buyer", session.metadata.session_id, TargetScope(keyword="other"))

    async def test_failed_skill_audit_does_not_open_tools(self):
        """SOP 未可靠登记时不发布新的授权集合，不能通过失败的加载扩权。"""
        session = await self.session()
        append = session.events.append
        async def fail_prepared(runtime_event):
            if runtime_event.node == "load_skill":
                raise RuntimeError("simulated-audit-failure")
            await append(runtime_event)
        with patch.object(session.events, "append", side_effect=fail_prepared):
            with self.assertRaises(RuntimeError):
                await self.load(session)
        self.assertEqual(session.loaded, {})
        self.assertEqual([tool.name for tool in session.offered_tools()], ["load_skill"])


if __name__ == "__main__":
    unittest.main()
