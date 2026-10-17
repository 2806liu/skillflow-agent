"""S1 基础层验收：预算、类型、真实图路由、API、持久化及只读重放。"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
import shutil
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import ValidationError

from skillflow.agent.adapters.scripted_model import ScriptedModel
from skillflow.agent.budget import BudgetExceeded, BudgetTracker
from skillflow.agent.config import BudgetLimits
from skillflow.agent.contracts import AgentState, BudgetUsage, MessageEnvelope, ModelReply, TaskMetadata, ToolCall
from skillflow.agent.graph import READY_ANSWER
from skillflow.api import create_app
from skillflow.config import ROOT, Settings
from tests_skillflow.test_streaming import frames


class BudgetTests(unittest.TestCase):
    """使用受控时钟，不依赖长时间等待验证计时与恢复累计。"""

    def test_snapshot_pause_restore_and_failures_keep_charges(self):
        now = [10.0]
        limits = BudgetLimits(max_rounds=2, max_model_requests=1, max_tool_attempts=1, active_timeout_seconds=5)
        tracker = BudgetTracker(limits, clock=lambda: now[0])
        tracker.start()
        tracker.reserve_model()
        tracker.reserve_tool()
        now[0] += 2
        self.assertEqual(tracker.snapshot().active_seconds, 2)
        self.assertEqual(tracker.pause().active_seconds, 2)
        now[0] += 100
        saved = BudgetUsage.model_validate_json(tracker.snapshot().model_dump_json())
        self.assertEqual(saved.active_seconds, 2)
        restored = BudgetTracker(limits, saved, clock=lambda: now[0])
        restored.start()
        for reserve, code in ((restored.reserve_model, "MODEL_LIMIT"), (restored.reserve_tool, "TOOL_LIMIT")):
            with self.assertRaises(BudgetExceeded) as caught:
                reserve()
            self.assertEqual(caught.exception.code, code)
        now[0] += 3
        with self.assertRaises(BudgetExceeded) as caught:
            restored.remaining_seconds()
        self.assertEqual(caught.exception.code, "TASK_TIMEOUT")
        self.assertEqual(restored.pause().active_seconds, 5)

    def test_round_limit_and_invalid_limits(self):
        tracker = BudgetTracker(BudgetLimits(max_rounds=1))
        tracker.reserve_model()
        with self.assertRaises(BudgetExceeded) as caught:
            tracker.reserve_model()
        self.assertEqual(caught.exception.code, "ROUND_LIMIT")
        self.assertEqual(tracker.snapshot().model_requests, 1)
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValidationError):
                BudgetLimits(active_timeout_seconds=value)

    def test_state_json_and_unknown_message_version_are_rejected(self):
        metadata = TaskMetadata(task_id="t", run_id="r", buyer_id="b", session_id="s", engine="tool_loop",
                                engine_version="tool_loop_s1_v1", graph_version="skeleton_v1", model_mode="scripted")
        state = AgentState(metadata=metadata, goal="查询")
        self.assertEqual(AgentState.model_validate_json(state.model_dump_json()), state)
        with self.assertRaises(ValidationError):
            AgentState.model_validate({**state.model_dump(), "schema_version": "unknown"})
        for data in ({"role": "tool", "content": "未配对"}, {"role": "python", "content": "不可执行"},
                     {"role": "user", "content": "查询", "buyer_id": "forged"}):
            with self.assertRaises(ValidationError):
                MessageEnvelope.model_validate(data)


class FoundationTests(unittest.TestCase):
    """每项使用项目 B 内的独立目录，模型全部脚本化，不访问外部服务。"""

    def setUp(self):
        self.base = (ROOT / ".skillflow" / "agent-s1-tests").resolve()
        self.root = self.base / uuid4().hex
        self.settings = Settings(data_dir=self.root, fixtures_dir=ROOT / "data", model_mode="mock",
                                 agent_engine="tool_loop", agent_model_mode="scripted", auth_mode="demo")
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        # 只删除本测试创建且已核验位于 B 内的目录，不触碰用户演示数据库。
        if self.root.resolve().is_relative_to(self.base) and self.root != self.base and self.root.exists():
            shutil.rmtree(self.root)

    def ask(self, message="查询 A123", session="s1"):
        response = self.client.post("/chat", json={"session_id": session, "message": message})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_api_graph_metadata_and_no_business_claims(self):
        with patch.object(self.app.state.runtime.plan_service, "decide", side_effect=AssertionError("legacy called")):
            result = self.ask()
        self.assertEqual(result["status"], "runtime_ready")
        self.assertEqual(result["answer"], READY_ANSWER)
        self.assertEqual(result["model_mode"], "scripted")
        self.assertEqual(result["result"]["execution_mode"], "simulation")
        self.assertFalse(result["result"]["business_completed"])
        self.assertEqual(result["result"]["stop_reason"], "S1_SKELETON_ONLY")
        metadata = result["result"]["metadata"]
        self.assertEqual(metadata["buyer_id"], self.settings.demo_buyer_id)
        self.assertEqual(metadata["run_id"], result["run_id"])
        self.assertEqual(metadata["engine"], "tool_loop")
        record = self.app.state.store.agent_run(result["run_id"])
        state = AgentState.model_validate(record["state"])
        self.assertEqual(state.budget.model_requests, 1)
        self.assertEqual(state.budget.tool_attempts, 0)
        self.assertGreater(state.budget.active_seconds, 0)
        self.assertEqual([event["node"] for event in result["events"]],
                         ["agent_initialize", "agent_model", "agent_model", "agent_finish"])
        self.assertTrue(all(event["details"]["task_id"] == metadata["task_id"] for event in result["events"]))
        self.assertEqual(self.client.get("/runs/" + result["run_id"]).json(), result)
        self.assertEqual(len(self.app.state.store.agent_events(result["run_id"])), 4)
        context = asyncio.run(self.app.state.session_checkpoint.load(self.settings.demo_buyer_id, "s1"))
        self.assertEqual(context["messages"], [])
        self.assertIsNone(context["order_id"])
        with closing(self.app.state.store.connect()) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM tickets").fetchone()[0], 0)
        self.assertEqual(self.app.state.chat_admission.outstanding, 0)

    def test_model_exception_is_sanitized_and_charged(self):
        model = AsyncMock(side_effect=RuntimeError("secret-sensitive-message"))
        with patch.object(self.app.state.agent_service.model, "generate", model):
            result = self.ask()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["result"]["stop_reason"], "MODEL_FAILED")
        self.assertEqual(result["result"]["budget"]["model_requests"], 1)
        self.assertNotIn("secret-sensitive-message", str(result))
        self.assertEqual(self.app.state.store.agent_run(result["run_id"])["status"], "failed")

    def test_tool_loop_is_selected_before_deterministic_incident_route(self):
        """SQLAlchemy 模式下也先选引擎，不能因咨询质量事件绕回 legacy。"""
        app = create_app(replace(self.settings, data_dir=self.root / "sql", business_backend="sqlalchemy"))
        try:
            with TestClient(app) as client:
                with patch.object(app.state.incident_service, "should_handle", side_effect=AssertionError("legacy incident")) as route:
                    response = client.post("/chat", json={"session_id": "incident", "message": "充电宝质量事件是否影响我的订单"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "runtime_ready")
                self.assertEqual(route.call_count, 0)
        finally:
            app.state.repository.engine.dispose()

    def test_model_and_task_timeouts_are_distinct(self):
        elapsed = [0.0]
        async def slow(*args):
            try:
                await asyncio.sleep(.15)
                return await ScriptedModel().generate([], [])
            finally:
                # 包括wait_for取消在内，模型等待已消耗时间；存储调度不参与本项时钟。
                elapsed[0] = .15
        for limit_name, code in (("model_timeout_seconds", "MODEL_TIMEOUT"), ("active_timeout_seconds", "TASK_TIMEOUT")):
            with self.subTest(limit=limit_name):
                elapsed[0] = 0.0
                config = replace(self.settings, agent_limits=BudgetLimits(**{limit_name: .03}))
                with TestClient(create_app(config)) as client:
                    # 本项只验证模型等待的超时分类；存储/线程调度不应先耗尽30ms。
                    # 累计有效时间已有BudgetTests受控时钟覆盖，生产计时规则不变。
                    with patch('skillflow.agent.service.BudgetTracker',
                               side_effect=lambda limits, usage: BudgetTracker(limits, usage, clock=lambda: elapsed[0])), \
                            patch.object(client.app.state.agent_service.model, "generate", side_effect=slow):
                        result = client.post("/chat", json={"session_id": code, "message": "查询"}).json()
                self.assertEqual(result["result"]["stop_reason"], code)
                self.assertEqual(result["result"]["budget"]["model_requests"], 1)

    def test_unimplemented_tool_call_fails_without_business_execution(self):
        reply = ModelReply(message=MessageEnvelope(role="assistant", content="",
            tool_calls=[ToolCall(call_id="call-1", name="submit_ticket", arguments={})]))
        with patch.object(self.app.state.agent_service.model, "generate", new=AsyncMock(return_value=reply)):
            result = self.ask()
        self.assertEqual(result["result"]["stop_reason"], "TOOL_EXECUTION_NOT_READY")
        self.assertEqual(result["result"]["budget"]["tool_attempts"], 0)

    def test_unsupported_modes_rejected_before_acceptance(self):
        for mode in ("disabled", "langchain_openai"):
            # S3 已支持真实模式；本项继续验证 disabled 或缺失模型配置在受理前拒绝。
            config = replace(self.settings, agent_model_mode=mode, api_key="", model="")
            with TestClient(create_app(config)) as client:
                for path in ("/chat", "/chat/stream"):
                    run_id = str(uuid4())
                    job = {"session_id": "unsupported", "message": "查询"}
                    if path.endswith("stream"):
                        job["run_id"] = run_id
                    response = client.post(path, json=job)
                    self.assertEqual(response.status_code, 503)
                    self.assertEqual(response.json()["detail"]["code"], "MODEL_CAPABILITY_UNSUPPORTED")
                    self.assertIsNone(client.app.state.store.agent_run(run_id))
                    self.assertEqual(client.app.state.chat_admission.outstanding, 0)
                    with closing(client.app.state.store.connect()) as connection:
                        self.assertEqual(connection.execute("SELECT count(*) FROM agent_tasks").fetchone()[0], 0)

    def test_demo_only_and_independent_directory(self):
        with self.assertRaises(ValueError):
            replace(self.settings, data_dir=ROOT / ".skillflow")
        with self.assertRaises(ValueError):
            replace(self.settings, data_dir=ROOT.parent / "not-project-b")
        production = replace(self.settings, auth_mode="production", oidc_issuer="https://identity.invalid",
                             oidc_client_id="s1", oidc_client_secret="test-secret", oidc_redirect_uri="https://b.invalid/cb")
        with TestClient(create_app(production)) as client:
            # 隔离入口认证后验证引擎预检；不发起任何 OIDC 请求。
            from skillflow.auth import Principal
            principal = Principal(self.settings.demo_buyer_id, "demo", "buyer")
            for route in client.app.routes:
                if getattr(route, "path", "") == "/chat":
                    dependency = route.dependant.dependencies[0].call
                    client.app.dependency_overrides[dependency] = lambda: principal
            with patch.object(client.app.state.store, "conversation_owner", return_value=principal.buyer_id):
                response = client.post("/chat", json={"session_id": "s1", "message": "查询"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["detail"]["code"], "ENGINE_DEMO_ONLY")

    def test_stream_replay_survives_configuration_change_and_restart(self):
        run_id = str(uuid4())
        job = {"run_id": run_id, "session_id": "stream", "message": "查询"}
        with patch.object(self.app.state.agent_service.model, "generate", wraps=self.app.state.agent_service.model.generate) as model:
            first = frames(self.client.post("/chat/stream", json=job).text)
            self.assertEqual(model.call_count, 1)
            self.assertEqual(first[-1][1], "final")
            self.assertEqual(first[-1][2]["status"], "runtime_ready")
            self.assertEqual(frames(self.client.post("/chat/stream", json=job).text), first)
            self.assertEqual(model.call_count, 1)
        task_id = self.app.state.store.agent_run(run_id)["metadata"]["task_id"]
        self.client.close()
        self.app = create_app(replace(self.settings, agent_engine="legacy", agent_model_mode="disabled"))
        self.client = TestClient(self.app)
        with patch.object(self.app.state.agent_service.model, "generate", side_effect=AssertionError("replay called")):
            self.assertEqual(frames(self.client.post("/chat/stream", json=job).text), first)
            self.assertEqual(frames(self.client.get(f"/runs/{run_id}/stream", params={"session_id": "stream"}).text), first)
        self.assertEqual(self.app.state.store.agent_run(run_id)["metadata"]["task_id"], task_id)
        self.assertEqual(len(self.app.state.store.agent_events(run_id)), 4)

    def test_owned_metadata_and_completed_run_cannot_restart_with_empty_budget(self):
        result = self.ask()
        tasks = self.app.state.agent_service.tasks
        with self.assertRaises(PermissionError):
            asyncio.run(tasks.get_owned(result["run_id"], "other-buyer", "s1"))
        with self.assertRaises(ValueError):
            asyncio.run(self.app.state.agent_service.run(result["run_id"], self.settings.demo_buyer_id, "s1", "查询"))
        self.assertEqual(self.app.state.store.agent_run(result["run_id"])["state"]["budget"]["model_requests"], 1)

    def test_atomic_registration_and_existing_stream_metadata_unchanged(self):
        from skillflow.agent.bootstrap import task_metadata
        store = self.app.state.store
        run_id = str(uuid4())
        metadata = task_metadata(self.settings, run_id, self.settings.demo_buyer_id, "atomic")
        forged = {**metadata, "buyer_id": "other"}
        with self.assertRaises(ValueError):
            store.create_stream(run_id, self.settings.demo_buyer_id, "atomic", "查询", queued=True, agent_metadata=forged)
        self.assertIsNone(store.stream_record(run_id))
        self.assertIsNone(store.run_owner(run_id))
        self.assertTrue(store.create_stream(run_id, self.settings.demo_buyer_id, "atomic", "查询", queued=True,
                                           agent_metadata=metadata))
        changed = {**metadata, "task_id": str(uuid4()), "engine": "legacy"}
        self.assertFalse(store.create_stream(run_id, self.settings.demo_buyer_id, "atomic", "查询", queued=True,
                                            agent_metadata=changed))
        self.assertEqual(store.agent_run(run_id)["metadata"], metadata)

    def test_queued_stream_uses_frozen_engine_and_limits(self):
        store = self.app.state.store
        run_id = str(uuid4())
        job = {"run_id": run_id, "session_id": "queued", "message": "查询"}
        generate = self.app.state.agent_service.model.generate

        def register(*args, **kwargs):
            # 在受理事务后模拟配置变化，执行必须仍读取首次冻结的元数据。
            original = store_type_create(store, *args, **kwargs)
            object.__setattr__(self.settings, "agent_engine", "legacy")
            object.__setattr__(self.settings, "agent_limits", BudgetLimits(max_model_requests=9))
            return original

        store_type_create = type(store).create_stream
        with patch.object(store, "create_stream", side_effect=register):
            with patch.object(self.app.state.agent_service.model, "generate", wraps=generate) as model:
                response = self.client.post("/chat/stream", json=job)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(model.call_count, 1)
        final = frames(response.text)[-1][2]
        self.assertEqual(final["result"]["metadata"]["engine"], "tool_loop")
        self.assertEqual(final["result"]["metadata"]["limits"]["max_model_requests"], 10)

    def test_cross_session_parallel_same_session_serial_and_budgets_isolated(self):
        """同一图实例并发运行时，Runtime context 和预算仍属于各自任务。"""
        active, peak = [0], [0]

        async def observed(*args):
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            try:
                await asyncio.sleep(.06)
                return await ScriptedModel().generate([], [])
            finally:
                active[0] -= 1

        with self.client:
            with patch.object(self.app.state.agent_service.model, "generate", side_effect=observed):
                for sessions, expected_peak in ((["parallel-a", "parallel-b"], 2), (["serial", "serial"], 1)):
                    peak[0] = 0
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        results = list(pool.map(lambda session: self.client.post(
                            "/chat", json={"session_id": session, "message": "查询"}).json(), sessions))
                    self.assertEqual(peak[0], expected_peak)
                    self.assertEqual(len({item["result"]["metadata"]["task_id"] for item in results}), 2)
                    self.assertTrue(all(item["status"] == "runtime_ready" for item in results))
                    self.assertTrue(all(item["result"]["budget"]["model_requests"] == 1 for item in results))
                    self.assertTrue(all(item["result"]["budget"]["tool_attempts"] == 0 for item in results))
        self.assertEqual(self.app.state.chat_admission.outstanding, 0)


if __name__ == "__main__":
    unittest.main()
