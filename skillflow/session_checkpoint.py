"""项目 B 会话的 LangGraph SQLite Checkpointer。

图只保存经过服务端生成的 JSON 会话上下文；输入为买家和会话 ID，输出
为当前商品、待确认动作等状态。首次读取可从旧 JSON 导入后的 SQLite KV
恢复，之后以 Checkpointer 为准。实际工单状态仍由工单表重新读取。
"""

import asyncio
from typing import TypedDict

import aiosqlite

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from skillflow.sqlite_store import SqliteStore


class SessionState(TypedDict):
    """只保存可序列化会话提示，不保存订单资格或工单事实副本。"""
    context: dict


class SessionCheckpoint:
    """以买家和会话组成 thread_id，在服务重启后恢复确认门。"""

    def __init__(self, store: SqliteStore):
        self.store = store
        builder = StateGraph(SessionState)
        builder.add_node("remember", self._remember)
        builder.add_edge(START, "remember")
        builder.add_edge("remember", END)
        self.builder = builder

    @staticmethod
    def _remember(state: SessionState):
        """将最新上下文写回图状态；失败由 Checkpointer 向 API 显式报告。"""
        return {"context": state["context"]}

    def _config(self, buyer_id: str, session_id: str):
        """买家隔离的线程 ID 防止相同聊天 ID 读取另一买家的状态。"""
        return {"configurable": {"thread_id": self.store.session_key(f"{buyer_id}:{session_id}")}}

    async def load(self, buyer_id: str, session_id: str):
        """先读 Checkpointer；首次迁移或空线程才使用 SQLite KV 会话。"""
        base = await asyncio.to_thread(self.store.session, session_id, buyer_id)
        # 每次调用独立连接；默认 5 秒 busy timeout 在峰值写入时会导致已受理任务无 final。
        async with aiosqlite.connect(self.store.database_path, timeout=30) as conn:
            await conn.execute("PRAGMA busy_timeout=30000")
            saver = AsyncSqliteSaver(conn)
            # 会话只含普通 JSON 类型，禁止从数据库反序列化任意 Python 类。
            saver.serde = JsonPlusSerializer(allowed_msgpack_modules=[])
            graph = self.builder.compile(checkpointer=saver)
            snapshot = await graph.aget_state(self._config(buyer_id, session_id))
            context = snapshot.values.get("context") if snapshot.values else None
        return {**base, **context} if isinstance(context, dict) else base

    async def save(self, buyer_id: str, session_id: str, context: dict):
        """先提交 LangGraph 检查点，再镜像到 SQLite KV 供旧会话兼容读取。"""
        # 与 load 使用相同的写等待上限，避免 SQLite 短暂独占写锁被当成任务失败。
        async with aiosqlite.connect(self.store.database_path, timeout=30) as conn:
            await conn.execute("PRAGMA busy_timeout=30000")
            saver = AsyncSqliteSaver(conn)
            saver.serde = JsonPlusSerializer(allowed_msgpack_modules=[])
            graph = self.builder.compile(checkpointer=saver)
            await graph.ainvoke({"context": context}, config=self._config(buyer_id, session_id))
        # SQLite 写锁可能等待 Checkpointer 的异步提交，不能在事件循环线程同步等待。
        await asyncio.to_thread(self.store.write, self.store.session_key(f"{buyer_id}:{session_id}"), context)
