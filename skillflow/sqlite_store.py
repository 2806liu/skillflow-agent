"""项目 B 的 SQLite 持久化入口。

输入为项目 B 的数据目录，输出为隔离的连接、KV 读写和事务。旧 JSON 只在
首次启动时导入且保留原文件；数据库不可写或迁移失败时直接报错，不再将
会话和工单静默放进进程内存，否则重启后确认对象会消失。
"""

import hashlib
import json
import sqlite3
import time
from collections import deque
from contextlib import closing, contextmanager
from pathlib import Path
from uuid import uuid4

from skillflow.storage import JsonStore


class SqliteStore(JsonStore):
    """复用会话键规则，把会话、Trace 和工单保存于项目 B 自己的数据库。"""

    def __init__(self, root: Path):
        self.root = root
        # 仅保留最近样本，不记录 SQL 参数或订单内容；用于定位 SQLite 单写入者瓶颈。
        self.write_wait_ms = deque(maxlen=4096)
        self.busy_errors = 0
        self.root.mkdir(parents=True, exist_ok=True)
        self.database_path = root / "skillflow.sqlite3"
        with closing(self.connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS kv (
                    name TEXT PRIMARY KEY, value_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS legacy_sources (
                    name TEXT PRIMARY KEY, content_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY, buyer_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, order_id TEXT NOT NULL, sku TEXT NOT NULL,
                    request_type TEXT NOT NULL, reason TEXT NOT NULL, status TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL, source_run_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ticket_events (
                    event_id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id),
                    buyer_id TEXT NOT NULL, session_id TEXT NOT NULL, action TEXT NOT NULL,
                    result TEXT NOT NULL, created_at TEXT NOT NULL, error_code TEXT
                );
                -- SSE 进度与最终结果按 run_id 保存；断线只读回放，不重新执行图或工单。
                CREATE TABLE IF NOT EXISTS run_streams (
                    run_id TEXT PRIMARY KEY, buyer_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    message_hash TEXT NOT NULL, status TEXT NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS stream_frames (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL
                        REFERENCES run_streams(run_id), kind TEXT NOT NULL, payload_json TEXT NOT NULL
                );
                -- B6 认证会话与一次性登录事务只在项目 B 的数据库保存摘要和期限。
                CREATE TABLE IF NOT EXISTS auth_flows (
                    state_hash TEXT PRIMARY KEY, nonce TEXT NOT NULL,
                    verifier TEXT NOT NULL, expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    session_hash TEXT PRIMARY KEY, buyer_id TEXT NOT NULL, subject TEXT NOT NULL,
                    role TEXT NOT NULL, csrf_hash TEXT NOT NULL, expires_at REAL NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS identity_bindings (
                    issuer TEXT NOT NULL, subject TEXT NOT NULL, buyer_id TEXT NOT NULL,
                    role TEXT NOT NULL, PRIMARY KEY(issuer, subject)
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT PRIMARY KEY, buyer_id TEXT NOT NULL, created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS run_owners (
                    run_id TEXT PRIMARY KEY, buyer_id TEXT NOT NULL, session_id TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_stream_run_seq ON stream_frames(run_id, seq);
                CREATE INDEX IF NOT EXISTS idx_run_owner_buyer ON run_owners(buyer_id, run_id);
                -- S1 最小任务元数据与诊断快照；不作为 LangGraph 检查点或恢复授权。
                CREATE TABLE IF NOT EXISTS agent_tasks (
                    task_id TEXT PRIMARY KEY, buyer_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    engine TEXT NOT NULL, engine_version TEXT NOT NULL, graph_version TEXT NOT NULL,
                    schema_version TEXT NOT NULL, model_mode TEXT NOT NULL, limits_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY REFERENCES run_owners(run_id),
                    task_id TEXT NOT NULL REFERENCES agent_tasks(task_id),
                    status TEXT NOT NULL, state_json TEXT, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_run_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES agent_runs(run_id), payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_agent_events_run ON agent_run_events(run_id, seq);
                CREATE INDEX IF NOT EXISTS idx_ticket_order ON tickets(buyer_id, order_id);
                CREATE INDEX IF NOT EXISTS idx_event_ticket ON ticket_events(ticket_id, created_at);
                CREATE UNIQUE INDEX IF NOT EXISTS uniq_active_type
                    ON tickets(buyer_id, order_id, request_type)
                    WHERE status IN ('draft', 'submitted');
                CREATE UNIQUE INDEX IF NOT EXISTS uniq_active_return_exchange
                    ON tickets(buyer_id, order_id)
                    WHERE request_type IN ('return', 'exchange') AND status IN ('draft', 'submitted');
                CREATE UNIQUE INDEX IF NOT EXISTS uniq_active_after_sales_price
                    ON tickets(buyer_id, order_id)
                    WHERE request_type IN ('return', 'exchange', 'price_adjustment')
                    AND status IN ('draft', 'submitted');
            """)
            # 早期 SSE 开发库若缺心跳列，原地补列；旧帧仍可按 run_id 回放。
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(run_streams)")}
            if "updated_at" not in columns:
                conn.execute("ALTER TABLE run_streams ADD COLUMN updated_at REAL NOT NULL DEFAULT 0")
            # 旧库原地加任务租约列；已有 SSE 与工单数据不迁走也不清空。
            if "lease_owner" not in columns:
                conn.execute("ALTER TABLE run_streams ADD COLUMN lease_owner TEXT")
            if "lease_until" not in columns:
                conn.execute("ALTER TABLE run_streams ADD COLUMN lease_until REAL")
            if "queued_at" not in columns:
                conn.execute("ALTER TABLE run_streams ADD COLUMN queued_at REAL")
            if "started_at" not in columns:
                conn.execute("ALTER TABLE run_streams ADD COLUMN started_at REAL")
            frame_columns = {row["name"] for row in conn.execute("PRAGMA table_info(stream_frames)")}
            if "created_at" not in frame_columns:
                conn.execute("ALTER TABLE stream_frames ADD COLUMN created_at REAL NOT NULL DEFAULT 0")
        self._import_legacy_json()

    def connect(self):
        """每次操作新建连接；SQLite 写锁和忙等待负责跨进程串行化。"""
        conn = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextmanager
    def transaction(self):
        """工单状态和历史写入共享一个 IMMEDIATE 事务，异常时全部回滚。"""
        with closing(self.connect()) as conn:
            started = time.perf_counter()
            try:
                conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError:
                self.busy_errors += 1
                raise
            self.write_wait_ms.append((time.perf_counter() - started) * 1000)
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def read(self, name: str, default=None):
        """从 SQLite KV 返回独立对象；缺失时保持原接口的默认值。"""
        with closing(self.connect()) as conn:
            row = conn.execute("SELECT value_json FROM kv WHERE name=?", (name,)).fetchone()
        return json.loads(row["value_json"]) if row else default

    def write(self, name: str, value):
        """会话和 Trace 原子覆盖；写入失败向 API 报错，防止虚假成功。"""
        payload = json.dumps(value, ensure_ascii=False)
        with self.transaction() as conn:
            conn.execute("INSERT INTO kv(name, value_json) VALUES(?, ?) "
                         "ON CONFLICT(name) DO UPDATE SET value_json=excluded.value_json", (name, payload))

    def create_stream(self, run_id: str, buyer_id: str, session_id: str, message: str,
                      queued: bool = False, agent_metadata: dict | None = None) -> bool:
        """原子登记运行；重复 run_id 仅在买家、会话和消息一致时允许回放。"""
        digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM run_streams WHERE run_id=?", (run_id,)).fetchone()
            if row:
                if (row["buyer_id"], row["session_id"], row["message_hash"]) != (buyer_id, session_id, digest):
                    raise ValueError("run_id 已用于其他请求")
                return False
            accepted_at = time.time()
            conn.execute("INSERT INTO run_streams(run_id,buyer_id,session_id,message_hash,status,updated_at,queued_at,started_at) "
                         "VALUES(?,?,?,?,?,?,?,?)",
                         (run_id, buyer_id, session_id, digest, "queued" if queued else "running",
                          accepted_at, accepted_at, None if queued else accepted_at))
            # 流式运行与非流式运行共用归属表；登记和首帧必须同一事务提交。
            conn.execute("INSERT INTO run_owners VALUES(?, ?, ?)", (run_id, buyer_id, session_id))
            # 同一事务冻结元数据；重复 POST 在上方返回，不能重建任务或替换引擎。
            if agent_metadata is not None:
                self._insert_agent_metadata(conn, agent_metadata, run_id, buyer_id, session_id)
            conn.execute("INSERT INTO stream_frames(run_id, kind, payload_json, created_at) VALUES(?, 'started', ?, ?)",
                         (run_id, json.dumps({"run_id": run_id}, ensure_ascii=False), accepted_at))
            return True

    def claim_stream(self, run_id: str, worker_id: str) -> bool:
        """队列任务原子认领；只有持有本进程执行位的工作者可从 queued 转 running。"""
        with self.transaction() as conn:
            claimed_at = time.time()
            changed = conn.execute("UPDATE run_streams SET status='running', lease_owner=?, lease_until=?, "
                                   "updated_at=?, started_at=? WHERE run_id=? AND status='queued'",
                                   (worker_id, claimed_at + 20, claimed_at, claimed_at, run_id)).rowcount
            return changed == 1

    def create_auth_flow(self, state_hash: str, nonce: str, verifier: str, expires_at: float):
        """保存授权码流程的随机 state 摘要；原始 state 不写数据库或 Trace。"""
        with self.transaction() as conn:
            conn.execute("INSERT INTO auth_flows VALUES(?, ?, ?, ?)",
                         (state_hash, nonce, verifier, expires_at))

    def provision_identity(self, issuer: str, subject: str, buyer_id: str, role: str = "buyer"):
        """仅供可信管理流程离线建立身份映射；登录声明本身不能领取演示买家数据。"""
        if not issuer or not subject or not buyer_id or role not in {"buyer", "admin"}:
            raise ValueError("身份映射字段无效")
        with self.transaction() as conn:
            conn.execute("INSERT INTO identity_bindings VALUES(?, ?, ?, ?) "
                         "ON CONFLICT(issuer, subject) DO UPDATE SET buyer_id=excluded.buyer_id, role=excluded.role",
                         (issuer, subject, buyer_id, role))

    def identity_binding(self, issuer: str, subject: str):
        """只根据已验证令牌的发行者和主体读取预先建立的买家归属。"""
        with closing(self.connect()) as conn:
            row = conn.execute("SELECT * FROM identity_bindings WHERE issuer=? AND subject=?",
                               (issuer, subject)).fetchone()
        return dict(row) if row else None

    def consume_auth_flow(self, state_hash: str, now: float):
        """回调只能消耗一次登录事务；过期或重放均返回空。"""
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM auth_flows WHERE state_hash=?", (state_hash,)).fetchone()
            conn.execute("DELETE FROM auth_flows WHERE state_hash=?", (state_hash,))
            return dict(row) if row and row["expires_at"] > now else None

    def create_auth_session(self, session_hash: str, buyer_id: str, subject: str, role: str,
                            csrf_hash: str, expires_at: float):
        """签发认证会话时只保存 Cookie 和 CSRF 摘要，不保存浏览器中的原值。"""
        with self.transaction() as conn:
            conn.execute("INSERT INTO auth_sessions VALUES(?, ?, ?, ?, ?, ?, 0)",
                         (session_hash, buyer_id, subject, role, csrf_hash, expires_at))

    def auth_session(self, session_hash: str, now: float):
        """每次访问按期限及撤销位验证；不存在即视为未登录。"""
        with closing(self.connect()) as conn:
            row = conn.execute("SELECT * FROM auth_sessions WHERE session_hash=? AND revoked=0 "
                               "AND expires_at>?", (session_hash, now)).fetchone()
        return dict(row) if row else None

    def revoke_auth_session(self, session_hash: str):
        """退出立即撤销当前 Cookie 对应的认证会话。"""
        with self.transaction() as conn:
            conn.execute("UPDATE auth_sessions SET revoked=1 WHERE session_hash=?", (session_hash,))

    def create_conversation(self, buyer_id: str) -> str:
        """由服务端分配对话 ID；客户端不能用猜测的 ID 领取他人会话。"""
        session_id = str(uuid4())
        with self.transaction() as conn:
            conn.execute("INSERT INTO conversations VALUES(?, ?, ?)", (session_id, buyer_id, time.time()))
        return session_id

    def conversation_owner(self, session_id: str) -> str | None:
        """只读取会话所属买家，用于区分不存在的会话和已有会话的越权访问。"""
        with closing(self.connect()) as conn:
            row = conn.execute("SELECT buyer_id FROM conversations WHERE session_id=?",
                               (session_id,)).fetchone()
        return row["buyer_id"] if row else None

    def owns_conversation(self, buyer_id: str, session_id: str) -> bool:
        """使用 Checkpointer 前先验证真实买家的对话归属。"""
        with closing(self.connect()) as conn:
            row = conn.execute("SELECT 1 FROM conversations WHERE session_id=? AND buyer_id=?",
                               (session_id, buyer_id)).fetchone()
        return bool(row)

    def register_run(self, run_id: str, buyer_id: str, session_id: str, agent_metadata: dict | None = None):
        """非流式运行在调用模型前登记买家；旧 KV 缺归属时不能补领。"""
        with self.transaction() as conn:
            conn.execute("INSERT INTO run_owners VALUES(?, ?, ?)", (run_id, buyer_id, session_id))
            # 非流式归属和引擎版本也在同一事务登记，排队执行时只读固定值。
            if agent_metadata is not None:
                self._insert_agent_metadata(conn, agent_metadata, run_id, buyer_id, session_id)

    @staticmethod
    def _insert_agent_metadata(conn, metadata: dict, run_id: str, buyer_id: str, session_id: str):
        """只允许登记与受理身份一致的元数据；调用方必须持有登记运行的事务。"""
        if (metadata["run_id"], metadata["buyer_id"], metadata["session_id"]) != (run_id, buyer_id, session_id):
            raise ValueError("任务元数据与受理归属不一致")
        conn.execute("INSERT INTO agent_tasks VALUES(?,?,?,?,?,?,?,?,?,?)", (
            metadata["task_id"], buyer_id, session_id, metadata["engine"], metadata["engine_version"],
            metadata["graph_version"], metadata["schema_version"], metadata["model_mode"],
            json.dumps(metadata["limits"], ensure_ascii=False), time.time()))
        conn.execute("INSERT INTO agent_runs VALUES(?,?, 'created', NULL, ?)",
                     (run_id, metadata["task_id"], time.time()))

    def agent_run(self, run_id: str):
        """加载服务端冻结的任务元数据；历史运行无记录时返回 None，保持 legacy 兼容。"""
        with closing(self.connect()) as conn:
            row = conn.execute("SELECT t.*, r.run_id, r.status, r.state_json FROM agent_runs r "
                               "JOIN agent_tasks t ON t.task_id=r.task_id WHERE r.run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        metadata = {key: row[key] for key in (
            "task_id", "run_id", "buyer_id", "session_id", "engine", "engine_version",
            "graph_version", "schema_version", "model_mode")}
        metadata["limits"] = json.loads(row["limits_json"])
        return {"metadata": metadata, "status": row["status"],
                "state": json.loads(row["state_json"]) if row["state_json"] else None}

    def save_agent_state(self, run_id: str, metadata: dict, state: dict):
        """保存受控诊断快照，不允许替换任务、归属或版本。"""
        record = self.agent_run(run_id)
        if record is None or record["metadata"] != metadata:
            raise ValueError("任务元数据不匹配")
        with self.transaction() as conn:
            conn.execute("UPDATE agent_runs SET status=?, state_json=?, updated_at=? WHERE run_id=?",
                         (state["status"], json.dumps(state, ensure_ascii=False), time.time(), run_id))

    def append_agent_event(self, run_id: str, event: dict):
        """按持久序号保存新图事件；异常不静默丢弃。"""
        with self.transaction() as conn:
            conn.execute("INSERT INTO agent_run_events(run_id,payload_json) VALUES(?,?)",
                         (run_id, json.dumps(event, ensure_ascii=False)))

    def agent_events(self, run_id: str):
        """供受控验收和适配器读取事件；外部 API 仍需先验证运行归属。"""
        with closing(self.connect()) as conn:
            rows = conn.execute("SELECT payload_json FROM agent_run_events WHERE run_id=? ORDER BY seq", (run_id,)).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def run_owner(self, run_id: str):
        """执行记录读取必须通过独立归属表，不能仅凭可猜测的 UUID。"""
        with closing(self.connect()) as conn:
            row = conn.execute("SELECT * FROM run_owners WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def stream_record(self, run_id: str):
        """读取运行归属和状态；调用方须用服务端买家与会话做匹配。"""
        with closing(self.connect()) as conn:
            row = conn.execute("SELECT * FROM run_streams WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def append_stream_frame(self, run_id: str, kind: str, payload: dict):
        """逐帧持久化进度；最终帧同时关闭运行，保证回放不会停在半完成状态。"""
        with self.transaction() as conn:
            conn.execute("INSERT INTO stream_frames(run_id, kind, payload_json, created_at) VALUES(?, ?, ?, ?)",
                         (run_id, kind, json.dumps(payload, ensure_ascii=False), time.time()))
            conn.execute("UPDATE run_streams SET updated_at=? WHERE run_id=?", (time.time(), run_id))
            if kind in {"final", "error", "interrupted"}:
                conn.execute("UPDATE run_streams SET status=? WHERE run_id=?", (kind, run_id))

    def touch_stream(self, run_id: str):
        """后台任务定时刷新租约；其他进程据此判断运行仍存活。"""
        with self.transaction() as conn:
            conn.execute("UPDATE run_streams SET updated_at=?, lease_until=? "
                         "WHERE run_id=? AND status='running'",
                         (time.time(), time.time() + 20, run_id))

    def stream_frames(self, run_id: str, after_seq: int = 0):
        """按数据库序号读取未展示帧；GET 回放绝不调用 Planner 或业务工具。"""
        with closing(self.connect()) as conn:
            rows = conn.execute("SELECT seq, kind, payload_json, created_at FROM stream_frames "
                                "WHERE run_id=? AND seq>? ORDER BY seq", (run_id, after_seq)).fetchall()
        return [{"seq": row["seq"], "kind": row["kind"], "payload": json.loads(row["payload_json"]),
                 "created_at": row["created_at"]}
                for row in rows]

    def interrupt_stream(self, run_id: str, stale_seconds: float = 15):
        """心跳过期后先查完整回答；已入库则补发 final，否则才标记中断。"""
        with self.transaction() as conn:
            row = conn.execute("SELECT status, updated_at, lease_until FROM run_streams WHERE run_id=?",
                               (run_id,)).fetchone()
            # 排队状态不能沿用运行中的 15 秒阈值；仍有活租约的跨进程工作者不可被重连误中断。
            stale = (row and ((row["status"] == "running" and
                               time.time() - row["updated_at"] > stale_seconds and
                               (row["lease_until"] is None or row["lease_until"] < time.time())) or
                              (row["status"] == "queued" and time.time() - row["updated_at"] > 300)))
            if stale:
                saved = conn.execute("SELECT value_json FROM kv WHERE name=?", ("run_" + run_id,)).fetchone()
                kind = "final" if saved else "interrupted"
                payload = (saved["value_json"] if saved else
                           json.dumps({"message": "服务重启中断本次执行；请查询工单实际状态后再发起新请求。"}, ensure_ascii=False))
                conn.execute("INSERT INTO stream_frames(run_id, kind, payload_json, created_at) VALUES(?, ?, ?, ?)",
                             (run_id, kind, payload, time.time()))
                conn.execute("UPDATE run_streams SET status=?, updated_at=? WHERE run_id=?",
                             (kind, time.time(), run_id))

    def _import_legacy_json(self):
        """按文件摘要增量接收旧 JSON，保留源文件；较新的 SQLite 工单状态优先。"""
        with self.transaction() as conn:
            # 文件摘要记录让重启时可以接收旧版服务停机前的末次写入，同时避免重复导入。
            for pattern in ("session_*.json", "run_*.json"):
                for path in self.root.glob(pattern):
                    raw = path.read_bytes()
                    digest = hashlib.sha256(raw).hexdigest()
                    previous = conn.execute("SELECT content_hash FROM legacy_sources WHERE name=?", (path.name,)).fetchone()
                    if previous and previous["content_hash"] == digest:
                        continue
                    value = json.loads(raw.decode("utf-8"))
                    # 检查点读取优先于 KV；更新旧会话文件只用于承接旧进程的最后状态。
                    conn.execute("INSERT INTO kv(name, value_json) VALUES(?, ?) "
                                 "ON CONFLICT(name) DO UPDATE SET value_json=excluded.value_json",
                                 (path.stem, json.dumps(value, ensure_ascii=False)))
                    conn.execute("INSERT INTO legacy_sources(name, content_hash) VALUES(?, ?) "
                                 "ON CONFLICT(name) DO UPDATE SET content_hash=excluded.content_hash",
                                 (path.name, digest))
            old_tickets = self.root / "tickets.json"
            if old_tickets.exists():
                raw = old_tickets.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                previous = conn.execute("SELECT content_hash FROM legacy_sources WHERE name=?", (old_tickets.name,)).fetchone()
                if previous and previous["content_hash"] == digest:
                    return
                records = json.loads(raw.decode("utf-8"))
                for record in records.values():
                    existing = conn.execute("SELECT updated_at FROM tickets WHERE ticket_id=?",
                                            (record["ticket_id"],)).fetchone()
                    if not existing:
                        conn.execute("""INSERT INTO tickets VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                     (record["ticket_id"], record["buyer_id"], record["session_id"],
                                      record["order_id"], record["sku"], record["request_type"],
                                      record["reason"], record["status"], record["idempotency_key"],
                                      record["source_run_id"], json.dumps(record.get("metadata", {}), ensure_ascii=False),
                                      record["created_at"], record["updated_at"]))
                    elif record["updated_at"] > existing["updated_at"]:
                        # 只接收更新时间更晚的旧工单，避免旧 JSON 覆盖 B4 中已取消的状态。
                        conn.execute("UPDATE tickets SET status=?, metadata_json=?, updated_at=? WHERE ticket_id=?",
                                     (record["status"], json.dumps(record.get("metadata", {}), ensure_ascii=False),
                                      record["updated_at"], record["ticket_id"]))
                    for event in record.get("ticket_events", []):
                        conn.execute("INSERT OR IGNORE INTO ticket_events VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                                     (event["event_id"], event["ticket_id"], event["buyer_id"],
                                      event["session_id"], event["action"], event["result"],
                                      event["created_at"], event.get("error_code")))
                conn.execute("INSERT INTO legacy_sources(name, content_hash) VALUES(?, ?) "
                             "ON CONFLICT(name) DO UPDATE SET content_hash=excluded.content_hash",
                             (old_tickets.name, digest))
