"""短期目标的 SQLite 适配；归属、版本和有效期检查独立于 Agent 图及模型。"""

import asyncio
import json
from time import time

from ..conversation_state import ConversationSnapshot, ConversationView, GoalTemplate


class SqliteConversationContext:
    """同进程会话锁由 API 提供；这里只做 B 自有 KV 原子保存，不声称多 worker 串行。"""

    def __init__(self, store, ttl_seconds=1800, clock=time):
        self.store, self.ttl_seconds, self.clock = store, ttl_seconds, clock

    def key(self, metadata):
        """元组用 JSON 编码后哈希，避免含冒号的身份/会话字符串拼接产生歧义。"""
        identity = json.dumps([metadata.buyer_id, metadata.session_id], ensure_ascii=False)
        return "agent_s3_context_" + self.store.session_key(identity)

    async def load(self, metadata):
        """过期、不兼容和归属不符都不能继承；旧 S3 无时间记录也不猜有效期。"""
        raw = await asyncio.to_thread(self.store.read, self.key(metadata))
        if not raw:
            legacy_key = "agent_s3_pending_" + self.store.session_key(metadata.buyer_id + ":" + metadata.session_id)
            old = await asyncio.to_thread(self.store.read, legacy_key)
            return ConversationView(status="incompatible" if old else "missing")
        try:
            saved = ConversationSnapshot.model_validate(raw)
        except (ValueError, TypeError):
            return ConversationView(status="incompatible")
        expected = (metadata.buyer_id, metadata.session_id, metadata.engine_version, metadata.graph_version, metadata.schema_version)
        actual = (saved.buyer_id, saved.session_id, saved.engine_version, saved.graph_version, saved.schema_version)
        if actual != expected:
            return ConversationView(status="incompatible")
        if self.clock() >= saved.expires_at:
            return ConversationView(status="expired", snapshot=saved)
        return ConversationView(status=saved.status, snapshot=saved)

    async def _write(self, metadata, *, template=None, reason=None):
        """在工具执行前提交目标或暂停屏障；后续业务失败不清除已确认目标。"""
        now = self.clock()
        saved = ConversationSnapshot(buyer_id=metadata.buyer_id, session_id=metadata.session_id,
            engine_version=metadata.engine_version, graph_version=metadata.graph_version, schema_version=metadata.schema_version,
            status="active" if template else "paused", template=template, source_task_id=metadata.task_id,
            source_run_id=metadata.run_id, confirmed_at=now, expires_at=now + self.ttl_seconds, pause_reason=reason)
        await asyncio.to_thread(self.store.write, self.key(metadata), saved.model_dump(mode="json"))
        return saved

    async def remember_goal(self, metadata, goal):
        """仅保存白名单需求枚举，不继承旧订单、证据、工具面或批量权限。"""
        return await self._write(metadata, template=GoalTemplate(requirements=tuple(sorted(set(goal.requirements)))))

    async def pause(self, metadata, reason):
        """新需求未确认前阻止回跳旧目标；包括目标识别超时和不支持的话题。"""
        return await self._write(metadata, reason=reason)

    async def save_outcome(self, state):
        """只补充实际多候选结果；不凭 completed/failed 改写目标或刷新有效期。"""
        candidates = state.constraints.get("candidates", [])
        if state.status != "clarify" or not candidates:
            return
        view = await self.load(state.metadata)
        saved = view.snapshot
        if view.status == "active" and saved.source_task_id == state.metadata.task_id:
            updated = saved.model_copy(update={"candidates": list(candidates)})
            await asyncio.to_thread(self.store.write, self.key(state.metadata), updated.model_dump(mode="json"))
