"""实际执行图 Saver 的生命周期；仅项目 B 独立 SQLite，不依赖诊断快照恢复。"""

from contextlib import asynccontextmanager
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


class SqliteGraphCheckpoints:
    """每段执行使用独立连接，关闭时不遗留后台线程；图数据只有受控 JSON。"""

    def __init__(self, root):
        self.path=root/'agent_checkpoints.sqlite3'

    @asynccontextmanager
    async def open(self):
        """禁 pickle 和任意模块反序列化，内部 Interrupt 由 SDK 安全类型白名单处理。"""
        async with aiosqlite.connect(self.path) as conn:
            await conn.execute('PRAGMA busy_timeout=30000')
            serde=JsonPlusSerializer(pickle_fallback=False,allowed_msgpack_modules=None)
            saver=AsyncSqliteSaver(conn,serde=serde)
            await saver.setup()
            yield saver
