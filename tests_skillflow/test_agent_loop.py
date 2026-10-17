"""S3 回归：脚本替身验证图边界；真实模型与浏览器验收另存独立报告。"""

import asyncio
import json
import shutil
import unittest
from contextlib import closing
from dataclasses import replace
from unittest.mock import patch
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from fastapi.testclient import TestClient

from skillflow.agent.budget import BudgetExceeded, BudgetTracker
from skillflow.agent.config import BudgetLimits
from skillflow.agent.context import checked_context
from skillflow.agent.contracts import MessageEnvelope, ModelReply, ToolCall
from skillflow.agent.goals import AnswerProposal, GoalProposal, checked_goal
from skillflow.api import create_app
from skillflow.config import ROOT, Settings
from tests_skillflow.test_streaming import frames


class TestModel:
    """只在测试中按观察生成工具消息；不作为真实模型失败时的回退。"""

    def __init__(self):
        self.calls = 0
        self.only_order = False
        self.slow = False
        self.break_final = False
        self.repeat_order = False
        self.slow_generate = False

    async def interpret_goal(self, text):
        """提供可重复目标对象，服务端仍需固定和核验范围。"""
        if self.slow:
            await asyncio.sleep(.08)
        if "天气" in text or "写代码" in text:
            return GoalProposal(supported=False)
        if "随便处理" in text:
            return GoalProposal(needs_clarification=True)
        requirements = ["order"]
        if any(term in text for term in ("充电宝", "质量事件", "批次", "受影响")):
            requirements.append("incident")
        elif "退" in text or "换货" in text:
            requirements.append("returns")
        if "替代" in text:
            requirements.append("alternatives")
        if "优惠" in text:
            requirements.append("discount")
        if "价保" in text or "价差" in text:
            requirements.append("price")
        if "物流" in text:
            requirements.append("logistics")
        return GoalProposal(requirements=requirements,
                            keyword="耳机" if "耳机" in text and "A12" not in text else "")

    async def generate(self, messages, tool_specs):
        """读取真实观察中的 ref，不预填政策或业务结果。"""
        self.calls += 1
        if self.slow_generate:
            await asyncio.sleep(.08)
        goal = json.loads(messages[0].content.split("固定目标：")[1].split("\n可加载技能目录：")[0])
        previous = {}
        for message in messages:
            if message.role == "assistant":
                for call in message.tool_calls:
                    previous[call.name] = {"arguments": call.arguments}
            elif message.role == "tool":
                for name, row in previous.items():
                    if "observation" not in row:
                        row["observation"] = json.loads(message.content)
                        break
        name, args = None, {}
        if "load_skill" not in previous:
            skill = next((skill for requirement, skill in (("incident", "quality_incident"), ("returns", "return_exchange"),
                         ("discount", "coupon_dispute"), ("price", "price_adjustment"), ("logistics", "logistics_tracking")) if requirement in goal["requirements"]), "order_tracking")
            name, args = "load_skill", {"name": skill, "reason": "验证只读调查"}
        elif not goal["scope"]["explicit_order_ids"]:
            if "list_my_orders" not in previous:
                name, args = "list_my_orders", {"keyword": goal["scope"]["keyword"]}
        else:
            oid = goal["scope"]["explicit_order_ids"][0]
            if "get_order" not in previous:
                name, args = "get_order", {"order_id": oid}
            elif self.repeat_order:
                name, args = "get_order", {"order_id": oid}
            elif self.only_order or goal["requirements"] == ["order"] or previous["get_order"]["observation"]["status"] != "success":
                pass
            elif "logistics" in goal["requirements"]:
                # S5 前物流工具未开放，模型替身正确结束，交完成门说明不支持。
                pass
            elif "incident" in goal["requirements"]:
                if "match_quality_incident" not in previous:
                    name, args = "match_quality_incident", {"order_id": oid}
                elif "alternatives" in goal["requirements"] and "list_alternatives" not in previous:
                    name, args = "list_alternatives", {"order_id": oid}
            elif "retrieve_policy" not in previous:
                name, args = "retrieve_policy", {"order_id": oid, "query": "退换条件"}
            elif previous["retrieve_policy"]["observation"]["status"] == "success":
                assess_name = "analyze_discount" if "discount" in goal["requirements"] else "analyze_price_adjustment" if "price" in goal["requirements"] else "assess_after_sales"
                if assess_name not in previous:
                    name, args = assess_name, {"order_id": oid, "evidence_refs": [e["ref"] for e in previous["retrieve_policy"]["observation"]["data"]["evidence"]]}
                elif assess_name == "assess_after_sales" and "check_inventory" not in previous:
                    name, args = "check_inventory", {"order_id": oid}
        return ModelReply(message=MessageEnvelope(role="assistant", content="" if name else "调查结束",
            tool_calls=[ToolCall(call_id=uuid4().hex, name=name, arguments=args)] if name else []))

    async def propose_answer(self, facts, refs):
        """只选择事实索引；故障模式省略索引，用于检验最后一道完成门。"""
        return AnswerProposal(fact_indexes=[] if self.break_final else list(range(len(facts))), result_refs=refs)


class LoopTests(unittest.TestCase):
    """隔离 B 内测试数据库；默认禁止外部模型访问。"""

    def setUp(self):
        self.base = ROOT / ".skillflow" / "agent-s3-tests"
        self.root = self.base / uuid4().hex
        self.settings = Settings(data_dir=self.root, model_mode="mock", agent_engine="tool_loop",
            agent_model_mode="langchain_openai", model="test-model", api_key="test-key", business_backend="sqlalchemy",
            auth_mode="demo", demo_buyer_id="demo-buyer", agent_context_ttl_seconds=1800)
        self.model = TestModel()
        self.model_patch = patch("skillflow.agent.adapters.chat_model.ChatModelGateway", return_value=self.model)
        self.model_patch.start()
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.model_patch.stop()
        self.app.state.repository.engine.dispose()
        # 删除目标在 B 的本测试根内且不是父目录；不删除演示库。
        if self.root.resolve().is_relative_to(self.base.resolve()) and self.root != self.base:
            shutil.rmtree(self.root)

    def ask(self, message, session="s3"):
        response = self.client.post("/chat", json={"session_id": session, "message": message})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_four_orders_and_native_message_loop(self):
        for oid, status, text in (("A123", "completed", "可用库存 5"), ("A124", "completed", "缺货"),
                                  ("A125", "needs_manual_review", "缺少可用政策"), ("A126", "completed", "退货资格不符合")):
            result = self.ask("查询 " + oid + " 能否退换", oid)
            self.assertEqual(result["status"], status, result)
            self.assertIn(text, result["answer"])
            state = self.app.state.store.agent_run(result["run_id"])["state"]
            self.assertEqual(state["goal"], "查询 " + oid + " 能否退换")
            self.assertTrue(state["messages"])
            checked_context([MessageEnvelope.model_validate(m) for m in state["messages"]], [], 100000)
            self.assertEqual(result["result"]["budget"]["model_requests"], result["result"]["budget"]["rounds"] + 2)
            self.assertEqual(self.client.get("/runs/" + result["run_id"]).json(), result)
        with closing(self.app.state.store.connect()) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM tickets").fetchone()[0], 0)

    def test_clarification_survives_restart_and_rechecks(self):
        first = self.ask("查询我的耳机能否退换")
        self.assertEqual(first["status"], "clarify", first)
        self.app.state.repository.engine.dispose()
        self.client.close()
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)
        second = self.ask("选择 A124")
        self.assertEqual(second["status"], "completed", second)
        self.assertIn("缺货", second["answer"])
        self.assertEqual(second["result"]["task_goal"]["continuation_of_task_id"], first["result"]["metadata"]["task_id"])
        self.assertNotEqual(second["result"]["metadata"]["task_id"], first["result"]["metadata"]["task_id"])

    def test_independent_goal_gate_rejects_premature_finish(self):
        self.model.only_order = True
        result = self.ask("A123 能否退换")
        self.assertEqual(result["result"]["stop_reason"], "FINISH_EVIDENCE_INCOMPLETE", result)
        self.assertTrue(self.app.state.store.agent_run(result["run_id"])["state"]["constraints"]["repair_used"])

    def test_final_invalid_proposal_cannot_invent_answer(self):
        self.model.break_final = True
        result = self.ask("A123 能否退换")
        self.assertEqual(result["result"]["stop_reason"], "ANSWER_PROPOSAL_INVALID")
        self.assertEqual(result["status"], "failed")

    def test_duplicate_at_budget_boundary_keeps_message_pair(self):
        """重复拦截碰到预算上限时，也不能留下未配对的原生工具消息。"""
        self.model.repeat_order = True
        app = create_app(replace(self.settings, agent_limits=BudgetLimits(max_tool_attempts=2)))
        try:
            with TestClient(app) as client:
                result = client.post("/chat", json={"session_id": "duplicate", "message": "A123 能否退换"}).json()
            self.assertEqual(result["result"]["stop_reason"], "TOOL_LIMIT", result)
            state = app.state.store.agent_run(result["run_id"])["state"]
            checked_context([MessageEnvelope.model_validate(m) for m in state["messages"]], [], 100000)
            self.assertEqual(state["observations"][-1]["error_code"], "TOOL_LIMIT")
        finally:
            app.state.repository.engine.dispose()

    def test_model_budget_timeout_and_context_limits(self):
        for kwargs, code in (({"max_model_requests": 1}, "MODEL_LIMIT"), ({"max_rounds": 1}, "ROUND_LIMIT"),
                             ({"max_tool_attempts": 1}, "TOOL_LIMIT")):
            budget_app = create_app(replace(self.settings, agent_limits=BudgetLimits(**kwargs)))
            with TestClient(budget_app) as client:
                result = client.post("/chat", json={"session_id": uuid4().hex, "message": "A123 能否退换"}).json()
            budget_app.state.repository.engine.dispose()
            self.assertEqual(result["result"]["stop_reason"], code, result)
        self.model.slow = True
        budget_app = create_app(replace(self.settings, agent_limits=BudgetLimits(model_timeout_seconds=.02)))
        with TestClient(budget_app) as client:
            # 超时用完整新目标触发模型；无上下文的裸单号现在应先澄清目的。
            result = client.post("/chat", json={"session_id": "timeout", "message": "A123 能否退换"}).json()
        self.assertEqual(result["result"]["stop_reason"], "MODEL_TIMEOUT", result)
        budget_app.state.repository.engine.dispose()

    def test_sse_readonly_replay_and_write_endpoints(self):
        job = {"session_id": "stream", "run_id": str(uuid4()), "message": "A123 能否退换"}
        first = frames(self.client.post("/chat/stream", json=job).text)
        calls = self.model.calls
        self.assertEqual(first[-1][1], "final")
        self.assertEqual(first[-1][2]["status"], "completed")
        self.assertEqual(frames(self.client.post("/chat/stream", json=job).text), first)
        self.assertEqual(self.model.calls, calls)
        for path, body in (("/tickets/draft", {"order_id": "A123", "request_type": "return", "reason": "测试"}),
                           ("/tickets/fake/confirm", {"confirmed": True}), ("/tickets/fake/cancel", {})):
            response = self.client.post(path, json={"session_id": "stream", **body})
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()["detail"]["code"], "WRITE_NOT_ENABLED")

    def test_quality_three_states(self):
        from evaluation.agent_runtime.s2_fixtures import seed_quality_demo
        seed_quality_demo(self.app.state.repository)
        for oid, text in (("A700", "AFFECTED"), ("A701", "NOT_AFFECTED"), ("A702", "UNVERIFIED"), ("A704", "需要人工核实")):
            result = self.ask(oid + " 充电宝是否受影响能否退货有哪些替代品", oid)
            # 客服文字与内部原因码分开，业务三态继续从可信证据核对，不要求对话暴露英文枚举。
            self.assertIn(text, str(result['result']['trusted_facts']), result)
            self.assertTrue(any(row["name"] == "match_quality_incident" for row in result["result"]["tool_records"]))


class ContextGoalTests(unittest.TestCase):
    """上下文完整性和目标权限独立于模型正确性。"""

    def test_goal_scope_and_budget_accounting(self):
        goal = checked_goal("A123 能否退换", GoalProposal(requirements=["order"]))
        self.assertIn("returns", goal.requirements)
        self.assertEqual(goal.scope.explicit_order_ids, ("A123",))
        with self.assertRaises(ValueError):
            checked_goal("查询耳机", GoalProposal(keyword="充电宝"))
        budget = BudgetTracker(BudgetLimits(max_rounds=1))
        budget.reserve_model(decision=False)
        budget.reserve_model()
        budget.reserve_model(decision=False)
        self.assertEqual((budget.snapshot().rounds, budget.snapshot().model_requests), (1, 3))

    def test_unpaired_and_oversize_context(self):
        with self.assertRaises(ValueError):
            checked_context([MessageEnvelope(role="assistant", content="", tool_calls=[ToolCall(call_id="x", name="get_order")])], [], 10000)
        with self.assertRaises(BudgetExceeded):
            checked_context([MessageEnvelope(role="user", content="字" * 2000)], [], 1000)


class NativeGatewayTests(unittest.IsolatedAsyncioTestCase):
    """验证适配器确实转换原生工具消息，不依赖网络或测试替身的调度规则。"""

    async def test_bind_tools_and_call_id_round_trip(self):
        from langchain_core.messages import AIMessage, ToolMessage
        from skillflow.agent.adapters.chat_model import ChatModelGateway
        from skillflow.agent.tool_registry import LocalToolRegistry
        gateway = object.__new__(ChatModelGateway)
        gateway.model = Mock()
        reply = AIMessage(content="", tool_calls=[{"id": "native-next", "name": "get_order", "args": {"order_id": "A123"}}])
        runnable = Mock(ainvoke=AsyncMock(return_value=reply))
        gateway.model.bind_tools.return_value = runnable
        messages = [MessageEnvelope(role="assistant", content="", tool_calls=[ToolCall(call_id="native-first", name="load_skill", arguments={"name": "order_tracking"})]),
                    MessageEnvelope(role="tool", content='{"status":"success"}', tool_call_id="native-first")]
        result = await gateway.generate(messages, LocalToolRegistry().model_specs(["get_order"]))
        self.assertEqual(result.message.tool_calls[0].call_id, "native-next")
        sent = runnable.ainvoke.call_args.args[0]
        self.assertIsInstance(sent[0], AIMessage)
        self.assertIsInstance(sent[1], ToolMessage)
        self.assertEqual(sent[1].tool_call_id, "native-first")
        self.assertEqual(gateway.model.bind_tools.call_args.args[0][0]["function"]["name"], "get_order")

    async def test_goal_gate_rejects_changed_inventory(self):
        """完成前重查库存可以拦住模型等待期间的业务事实变化。"""
        from skillflow.agent.tool_executor import ToolSession, BoundaryError
        from skillflow.agent.adapters.local_tools import fingerprint
        from skillflow.contracts import ToolResult
        session = object.__new__(ToolSession)
        order = {"order_id": "A123"}
        session._facts = {("check_inventory", "A123"): {"order_hash": fingerprint(order), "source_version": "policy-v1",
            "result_hash": fingerprint({"available": 5})}}
        session.adapter = Mock(order=AsyncMock(return_value=order), source_version=AsyncMock(return_value="policy-v1"),
            invoke=AsyncMock(return_value=ToolResult(status="success", data={"available": 0})))
        with self.assertRaises(BoundaryError) as caught:
            await session.verify_current()
        self.assertEqual(caught.exception.code, "INVENTORY_CHANGED")


if __name__ == "__main__":
    unittest.main()
