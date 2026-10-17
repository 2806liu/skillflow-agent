"""短期任务上下文验收：连续换单、明确切换、30分钟有效期、故障与隔离。"""

import asyncio
from dataclasses import replace
import json
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import ValidationError

from evaluation.agent_runtime.s2_fixtures import seed_quality_demo
from skillflow.agent.bootstrap import task_metadata
from skillflow.agent.config import BudgetLimits
from skillflow.agent.contracts import TaskMetadata
from skillflow.agent.conversation_policy import decide_conversation
from skillflow.agent.conversation_state import ConversationSnapshot, ConversationView, GoalTemplate
from skillflow.api import create_app
from tests_skillflow import test_agent_loop as loop_fixtures
from tests_skillflow.test_streaming import frames


class ConversationTests(unittest.TestCase):
    """复用隔离 SQL 测试装配，不继承原测试方法，全部模型调用使用替身。"""

    setUp = loop_fixtures.LoopTests.setUp
    tearDown = loop_fixtures.LoopTests.tearDown
    ask = loop_fixtures.LoopTests.ask

    def seed(self):
        """仅在本用例独立 B 数据库内生成固定质量事件样本。"""
        seed_quality_demo(self.app.state.repository)

    def snapshot(self, result):
        """按真实运行元数据读取当前会话记录，避免测试伪造模型记忆。"""
        metadata = TaskMetadata.model_validate(result["result"]["metadata"])
        adapter = self.app.state.agent_service.conversations
        return self.app.state.store.read(adapter.key(metadata))

    def test_chain_preserves_goal_after_unknown_order_and_missing_batch(self):
        self.seed()
        first = self.ask("A700 是否受充电宝质量事件影响，能否退货，有哪些替代品")
        previous = first
        refs = set(first["result"]["result_refs"])
        tasks = {first["result"]["metadata"]["task_id"]}
        for message, oid, decision in (("A701", "A701", "NOT_AFFECTED"), ("那 A702 呢", "A702", "UNVERIFIED"),
                                       ("A705", "A705", "ORDER_NOT_FOUND"), ("按刚才方式查 A704", "A704", "AFFECTED")):
            current = self.ask(message)
            goal = current["result"]["task_goal"]
            self.assertEqual(set(goal["requirements"]), {"order", "incident", "alternatives"}, current)
            self.assertEqual(goal["scope"]["explicit_order_ids"], [oid])
            self.assertFalse(goal["scope"]["quality_batch"])
            self.assertEqual(goal["continuation_of_task_id"], previous["result"]["metadata"]["task_id"])
            self.assertEqual(current["result"]["conversation_context"]["action"], "inherit")
            self.assertNotIn(current["result"]["metadata"]["task_id"], tasks)
            tasks.add(current["result"]["metadata"]["task_id"])
            self.assertFalse(refs.intersection(current["result"]["result_refs"]))
            refs.update(current["result"]["result_refs"])
            records = current["result"]["tool_records"]
            self.assertTrue(all(r["arguments"].get("order_id", oid) == oid for r in records))
            if decision == "ORDER_NOT_FOUND":
                self.assertEqual(current["result"]["stop_reason"], decision)
                self.assertFalse(any(r["name"] == "match_quality_incident" for r in records))
            else:
                matched = next(r for r in records if r["name"] == "match_quality_incident")
                self.assertEqual(matched["observation"]["data"]["decision"], decision)
                self.assertTrue(any(r["name"] == "list_alternatives" for r in records))
            self.assertEqual(self.snapshot(current)["status"], "active")
            previous = current

    def test_explicit_switch_and_followup_use_new_goal(self):
        self.seed()
        self.ask("A700 充电宝质量事件，能否退货及替代品")
        order_only = self.ask("只查 A701 订单状态")
        self.assertEqual(order_only["result"]["task_goal"]["requirements"], ["order"])
        second = self.ask("A702")
        self.assertEqual(second["result"]["task_goal"]["requirements"], ["order"])
        self.assertEqual([r["name"] for r in second["result"]["tool_records"]], ["load_skill", "get_order"])
        discount = self.ask("改查 A123 优惠")
        self.assertEqual(set(discount["result"]["task_goal"]["requirements"]), {"order", "discount"})
        followup = self.ask("A124")
        self.assertEqual(set(followup["result"]["task_goal"]["requirements"]), {"order", "discount"})
        self.assertTrue(any(r["name"] == "analyze_discount" for r in followup["result"]["tool_records"]))

    def test_policy_evidence_is_fresh_for_new_order(self):
        first = self.ask("A123 能否退换")
        second = self.ask("A124")
        def evidence(result):
            """从实际检索观察取引用，不能在替身中预填证据。"""
            return {item["ref"] for r in result["result"]["tool_records"] if r["name"] == "retrieve_policy"
                    for item in r["observation"]["data"]["evidence"]}
        self.assertTrue(evidence(first))
        self.assertTrue(evidence(second))
        self.assertFalse(evidence(first).intersection(evidence(second)))
        self.assertIn("缺货", second["answer"])
        serialized = json.dumps(self.snapshot(second), ensure_ascii=False)
        for forbidden in ("evidence_refs", "paid_cents", "A123", "A124", "tool_calls", "opened_tools"):
            self.assertNotIn(forbidden, serialized)

    def test_restart_keeps_goal_but_recreates_tools(self):
        self.seed()
        first = self.ask("A700 充电宝受影响吗，有哪些替代品")
        self.client.close()
        self.app.state.repository.engine.dispose()
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)
        result = self.ask("A701")
        self.assertEqual(result["result"]["task_goal"]["continuation_of_task_id"], first["result"]["metadata"]["task_id"])
        # 多轮目标和批次结果仍须正确；客服表达不再直接显示内部英文状态。
        self.assertIn("NOT_AFFECTED", str(result['result']['trusted_facts']))
        before = next(e for e in result["events"] if e["node"] == "agent_before_model")
        self.assertEqual(before["details"]["offered_tools"], ["load_skill"])

    def test_expiry_and_sse_replay_do_not_renew(self):
        clock = [1000.]
        self.app.state.agent_service.conversations.clock = lambda: clock[0]
        job = {"session_id": "ttl", "run_id": str(uuid4()), "message": "A123 能否退换"}
        first_frames = frames(self.client.post("/chat/stream", json=job).text)
        first = first_frames[-1][2]
        saved = self.snapshot(first)
        self.assertEqual(saved["expires_at"], 2800.)
        calls = self.model.calls
        clock[0] = 2799.
        self.assertEqual(frames(self.client.post("/chat/stream", json=job).text), first_frames)
        self.assertEqual(self.snapshot(first), saved)
        self.assertEqual(self.model.calls, calls)
        clock[0] = 2800.
        expired = self.ask("A124", "ttl")
        self.assertEqual(expired["status"], "clarify")
        self.assertEqual(expired["result"]["stop_reason"], "CONTEXT_EXPIRED")
        self.assertEqual(expired["result"]["tool_records"], [])
        self.assertEqual(self.model.calls, calls)

    def test_session_buyer_and_version_isolation(self):
        first = self.ask("A123 能否退换")
        missing = self.ask("A124", "new-session")
        self.assertEqual(missing["result"]["stop_reason"], "CONTEXT_MISSING")
        adapter = self.app.state.agent_service.conversations
        metadata = TaskMetadata.model_validate(first["result"]["metadata"])
        foreign = metadata.model_copy(update={"buyer_id": "other-buyer"})
        self.assertEqual(asyncio.run(adapter.load(foreign)).status, "missing")
        # 即使旧记录被误放进另一个键，内部归属核验仍拒绝，不靠键名当权限。
        self.app.state.store.write(adapter.key(foreign), self.snapshot(first))
        self.assertEqual(asyncio.run(adapter.load(foreign)).status, "incompatible")
        changed = metadata.model_copy(update={"engine_version": "future-engine"})
        self.assertEqual(asyncio.run(adapter.load(changed)).status, "incompatible")

    def test_new_order_still_enforces_ownership(self):
        self.ask("A123 能否退换")
        result = self.ask("A900")
        self.assertTrue(any(r["observation"]["error_code"] == "ORDER_FORBIDDEN" for r in result["result"]["tool_records"]))
        self.assertFalse(any(r["observation"]["data"].get("order") for r in result["result"]["tool_records"]))

    def test_confirmed_goal_survives_execution_timeout(self):
        self.seed()
        self.ask("A700 充电宝质量事件及替代品")
        self.model.slow_generate = True
        short_app = create_app(replace(self.settings, agent_limits=BudgetLimits(model_timeout_seconds=.02)))
        try:
            with TestClient(short_app) as client:
                failed = client.post("/chat", json={"session_id": "s3", "message": "A701"}).json()
            self.assertEqual(failed["result"]["stop_reason"], "MODEL_TIMEOUT")
            self.assertEqual(self.snapshot(failed)["status"], "active")
        finally:
            short_app.state.repository.engine.dispose()
            self.model.slow_generate = False
        result = self.ask("A702")
        self.assertIn("UNVERIFIED", str(result['result']['trusted_facts']))
        self.assertEqual(result["result"]["task_goal"]["continuation_of_task_id"], failed["result"]["metadata"]["task_id"])

    def test_unconfirmed_switch_timeout_blocks_old_goal(self):
        self.seed()
        self.ask("A700 充电宝质量事件")
        self.model.slow = True
        short_app = create_app(replace(self.settings, agent_limits=BudgetLimits(model_timeout_seconds=.02)))
        try:
            with TestClient(short_app) as client:
                result = client.post("/chat", json={"session_id": "s3", "message": "改查 A123 优惠"}).json()
            self.assertEqual(result["result"]["stop_reason"], "MODEL_TIMEOUT")
            self.assertEqual(self.snapshot(result)["status"], "paused")
        finally:
            short_app.state.repository.engine.dispose()
            self.model.slow = False
        followup = self.ask("A701")
        self.assertEqual(followup["result"]["stop_reason"], "CONTEXT_PAUSED")
        self.assertEqual(followup["result"]["tool_records"], [])

    def test_topic_and_ambiguity_pause_inheritance(self):
        self.seed()
        for ambiguous, expected in (("这个呢", "CONTEXT_AMBIGUOUS"), ("今天天气如何", "GOAL_UNSUPPORTED"),
                                     ("A701 随便处理一下", "CONTEXT_AMBIGUOUS")):
            self.ask("A700 充电宝质量事件")
            result = self.ask(ambiguous)
            self.assertEqual(result["result"]["stop_reason"], expected)
            self.assertEqual(result["result"]["tool_records"], [])
            self.assertEqual(self.ask("A701")["result"]["stop_reason"], "CONTEXT_PAUSED")

    def test_pending_candidate_selection_does_not_expand_scope(self):
        first = self.ask("查询我的耳机能否退换")
        saved = self.snapshot(first)
        rejected = self.ask("选择 A700")
        self.assertEqual(rejected["result"]["stop_reason"], "CANDIDATE_NOT_LISTED")
        self.assertEqual(self.snapshot(first), saved)
        selected = self.ask("选择 A124")
        self.assertEqual(selected["result"]["conversation_context"]["reason"], "PENDING_SELECTION")
        self.assertIn("缺货", selected["answer"])

    def test_goal_is_saved_before_tools_and_cancel_does_not_erase(self):
        service = self.app.state.agent_service
        run_id = str(uuid4())
        metadata = task_metadata(self.settings, run_id, "demo-buyer", "cancel")
        self.app.state.store.register_run(run_id, "demo-buyer", "cancel", metadata)
        with patch.object(self.model, "generate", new=AsyncMock(side_effect=asyncio.CancelledError)):
            try:
                result = asyncio.run(service.run(run_id, "demo-buyer", "cancel", "A123 能否退换"))
            except asyncio.CancelledError:
                pass
            else:
                # 图内取消与外部任务取消传播方式不同，两者都不得声称已完成调查。
                self.assertEqual((result.status, result.stop_reason), ("failed", "RUN_INTERRUPTED"))
        view = asyncio.run(service.conversations.load(TaskMetadata.model_validate(metadata)))
        self.assertEqual(view.status, "active")
        self.assertEqual(set(view.snapshot.template.requirements), {"order", "returns"})

    def test_business_capability_unsupported_keeps_confirmed_goal(self):
        first = self.ask("A123 查询物流")
        self.assertEqual(first["result"]["stop_reason"], "MCP_NOT_READY")
        self.assertEqual(self.snapshot(first)["status"], "active")
        second = self.ask("A124")
        self.assertEqual(second["result"]["task_goal"]["requirements"], ["logistics", "order"])
        self.assertEqual(second["result"]["stop_reason"], "MCP_NOT_READY")


class ConversationPolicyTests(unittest.TestCase):
    """纯策略验证继承范围；不依赖数据库、模型、定时等待或工具执行。"""

    def test_short_forms_and_batch_permission_reset(self):
        saved = ConversationSnapshot(buyer_id="b", session_id="s", engine_version="e", graph_version="g", schema_version="v",
            status="active", template=GoalTemplate(requirements=("order", "incident", "alternatives")),
            source_task_id="previous-task", source_run_id="previous-run", confirmed_at=100, expires_at=1900)
        for text in ("A701", "那 A701 呢", "按刚才方式查 A701", "再按刚才的方式查 A701"):
            result = decide_conversation(text, ConversationView(status="active", snapshot=saved))
            self.assertEqual(result.action, "inherit")
            self.assertFalse(result.goal.scope.quality_batch)
            self.assertEqual(result.goal.scope.explicit_order_ids, ("A701",))
            self.assertEqual(result.goal.scope.keyword, "")
        self.assertEqual(decide_conversation("只查 A701 订单状态", ConversationView(status="active", snapshot=saved)).action, "new")
        self.assertEqual(decide_conversation("A701 A702", ConversationView(status="active", snapshot=saved)).action, "clarify")
        self.assertEqual(decide_conversation("这个呢", ConversationView(status="active", snapshot=saved)).action, "clarify")

    def test_template_does_not_accept_facts_or_unknown_requirement(self):
        for data in ({"requirements": ["order"], "order_id": "A700"}, {"requirements": ["order", "refund_now"]},
                     {"requirements": ["order", "order"]}, {"requirements": ["incident"]}):
            with self.assertRaises(ValidationError):
                GoalTemplate.model_validate(data)


if __name__ == "__main__":
    unittest.main()
