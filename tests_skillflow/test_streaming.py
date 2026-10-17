"""B5 流式验收：进度可见、最终回答校验后发布、断线回放不重复执行业务。"""

from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from skillflow.api import create_app
from skillflow.config import ROOT, Settings


def frames(body: str):
    """解析测试客户端收到的 SSE 文本，返回数据库序号、事件类型和 JSON 数据。"""
    parsed = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        lines = block.splitlines()
        if not lines or lines[0].startswith(":"):
            continue
        fields = dict(line.split(": ", 1) for line in lines if ": " in line)
        if "event" in fields:
            parsed.append((int(fields["id"]), fields["event"], json.loads(fields["data"])))
    return parsed


class StreamingTests(unittest.TestCase):
    def setUp(self):
        # 每例使用项目 B 自有演示数据副本，既能验证 SQLite 回放，也不影响用户的工单。
        self.root = (ROOT / ".skillflow" / "stream-tests" / str(uuid4())).resolve()
        fixtures = self.root / "fixtures"
        (fixtures / "policies").mkdir(parents=True)
        for name in ("buyers.json", "orders.json", "inventory.json", "price_adjustments.json"):
            shutil.copyfile(ROOT / "data" / name, fixtures / name)
        shutil.copyfile(ROOT / "data/policies/earbuds_v1.md", fixtures / "policies/earbuds_v1.md")
        self.settings = Settings(data_dir=self.root / "state", fixtures_dir=fixtures,
                                 timeout=2, model_mode="mock")
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        if self.root.is_relative_to((ROOT / ".skillflow" / "stream-tests").resolve()):
            shutil.rmtree(self.root)

    def test_progress_order_final_and_read_only_replay(self):
        """计划和工具开始先于工具结果；最终回答只在 critic 后出现，重放不增加帧。"""
        run_id = str(uuid4())
        job = {"run_id": run_id, "session_id": "stream-price", "message": "A123当前有价差吗"}
        response = self.client.post("/chat/stream", json=job)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        events = frames(response.text)
        kinds = [kind for _, kind, _ in events]
        self.assertEqual(kinds[0], "started")
        self.assertEqual(kinds[-1], "final")
        self.assertLess(kinds.index("plan"), kinds.index("tool_start"))
        starts = [data["tool"] for _, kind, data in events if kind == "tool_start"]
        self.assertEqual(starts, ["get_order", "retrieve_policy", "analyze_price_adjustment"])
        self.assertTrue(any(kind == "trace" and data["node"] == "critic" for _, kind, data in events))
        self.assertFalse(any("answer" in data for _, kind, data in events[:-1]))
        self.assertEqual(events[-1][2], self.client.get("/runs/" + run_id).json())
        replay = self.client.get(f"/runs/{run_id}/stream", params={"session_id": "stream-price"})
        self.assertEqual(frames(replay.text), events)
        after = events[3][0]
        suffix = frames(self.client.get(f"/runs/{run_id}/stream", params={
            "session_id": "stream-price", "after": after}).text)
        self.assertEqual(suffix, [frame for frame in events if frame[0] > after])
        self.assertEqual(frames(self.client.post("/chat/stream", json=job).text), events)
        # 已存在的执行记录若以错误会话读取，应返回可识别的越权错误。
        denied_replay = self.client.get(f"/runs/{run_id}/stream", params={"session_id": "other"})
        self.assertEqual(denied_replay.status_code, 403)
        self.assertEqual(denied_replay.json()["detail"]["code"], "RUN_FORBIDDEN")
        self.assertEqual(self.client.post("/chat/stream", json={**job, "message": "查询 A124"}).status_code, 409)

    def test_tool_start_is_visible_while_tool_is_running(self):
        """慢工具尚未结束时 SQLite 已保存开始帧，证明这不是完成后伪装的流。"""
        run_id = str(uuid4())
        tool = self.app.state.runtime.tools["get_order"]
        original = tool.call

        async def slow_call(arguments):
            await asyncio.sleep(0.6)
            return await original(arguments)

        with patch.object(tool, "call", side_effect=slow_call):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.client.post, "/chat/stream", json={
                    "run_id": run_id, "session_id": "stream-slow", "message": "查询 A123"})
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    current = self.app.state.store.stream_frames(run_id)
                    if any(item["kind"] == "tool_start" for item in current):
                        break
                    time.sleep(0.02)
                self.assertTrue(any(item["kind"] == "tool_start" for item in current))
                self.assertFalse(any(item["kind"] == "final" for item in current))
                self.assertEqual(future.result(timeout=5).status_code, 200)

    def test_replay_of_draft_request_does_not_create_second_ticket(self):
        """流式创建草稿之后重复 POST 和重启回放都只读取原 run_id 与原工单。"""
        session = "stream-draft"
        for message in ("A123我要退货", "对的"):
            self.assertEqual(self.client.post("/chat", json={"session_id": session,
                "message": message}).status_code, 200)
        job = {"run_id": str(uuid4()), "session_id": session, "message": "申请吧"}
        first = frames(self.client.post("/chat/stream", json=job).text)
        self.assertEqual(first[-1][1], "final")
        ticket = first[-1][2]["result"]["ticket"]
        self.assertEqual(ticket["status"], "draft")
        self.assertEqual(frames(self.client.post("/chat/stream", json=job).text), first)
        self.assertEqual([item.action for item in self.app.state.tickets.events(
            self.app.state.tickets.get(ticket["ticket_id"]))].count("draft_created"), 1)
        self.client.close()
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)
        restored = frames(self.client.get(f"/runs/{job['run_id']}/stream", params={
            "session_id": session}).text)
        self.assertEqual(restored, first)
        self.assertEqual(self.app.state.tickets.get(ticket["ticket_id"])["status"], "draft")

    def test_restart_marks_only_stale_run_interrupted(self):
        """服务重启后仅过期运行被标记中断；读取它不会执行原消息。"""
        run_id = str(uuid4())
        self.assertTrue(self.app.state.store.create_stream(run_id, self.settings.demo_buyer_id, "stale", "查询 A123"))
        with self.app.state.store.transaction() as conn:
            conn.execute("UPDATE run_streams SET updated_at=0 WHERE run_id=?", (run_id,))
        self.client.close()
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)
        replay = self.client.get(f"/runs/{run_id}/stream", params={"session_id": "stale"})
        self.assertEqual([kind for _, kind, _ in frames(replay.text)], ["started", "interrupted"])
        self.assertIsNone(self.app.state.store.read("run_" + run_id))

    def test_saved_answer_is_replayed_after_final_frame_crash_window(self):
        """完整回答已入库而 SSE 最终帧未写时，恢复应补发原回答而不重做业务。"""
        run_id = str(uuid4())
        store = self.app.state.store
        store.create_stream(run_id, self.settings.demo_buyer_id, "saved-final", "查询 A123")
        stored_answer = {"run_id": run_id, "status": "completed", "answer": "已核验回答"}
        store.write("run_" + run_id, stored_answer)
        with store.transaction() as conn:
            conn.execute("UPDATE run_streams SET updated_at=0 WHERE run_id=?", (run_id,))
        replay = frames(self.client.get(f"/runs/{run_id}/stream", params={
            "session_id": "saved-final"}).text)
        self.assertEqual([kind for _, kind, _ in replay], ["started", "final"])
        self.assertEqual(replay[-1][2], stored_answer)


if __name__ == "__main__":
    unittest.main()
