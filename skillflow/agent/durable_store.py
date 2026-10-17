"""S4 业务库账本：任务租约、累计预算、调用回执和显式恢复请求；不承担图调度。"""

from contextlib import closing
import json
import time
from uuid import uuid4

from .contracts import BudgetUsage, TaskMetadata
from .budget import BudgetTracker


class TaskConflict(ValueError):
    """稳定业务拒绝码，供 API 区分权限、版本冲突及正在执行。"""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


class DurableStore:
    """仅使用项目 B 的 SqliteStore；所有竞争判断在 IMMEDIATE 短事务内完成。"""

    def __init__(self, store, clock=time.time, lease_seconds=30):
        self.store, self.clock, self.lease_seconds = store, clock, lease_seconds
        # 增量迁移只新增 S4 表，不改旧工单与 S1—S3 记录；版本不兼容时拒绝猜测升级。
        with closing(store.connect()) as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS agent_s4_migrations(version INTEGER PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS agent_lifecycle(
                    task_id TEXT PRIMARY KEY REFERENCES agent_tasks(task_id), status TEXT NOT NULL,
                    version INTEGER NOT NULL, last_run_id TEXT NOT NULL, checkpoint_id TEXT,
                    lease_owner TEXT, lease_until REAL, segment_started REAL,
                    budget_json TEXT NOT NULL, pending_json TEXT, updated_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS agent_attempts(
                    attempt_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, run_id TEXT NOT NULL,
                    kind TEXT NOT NULL, usage_json TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS agent_invocations(
                    task_id TEXT NOT NULL, call_id TEXT NOT NULL, digest TEXT NOT NULL,
                    observation_json TEXT NOT NULL, session_json TEXT NOT NULL,
                    PRIMARY KEY(task_id,call_id));
                CREATE TABLE IF NOT EXISTS agent_confirmations(
                    confirmation_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, buyer_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, ticket_id TEXT NOT NULL, action_hash TEXT NOT NULL,
                    facts_hash TEXT NOT NULL, generation INTEGER NOT NULL, expires_at REAL NOT NULL,
                    status TEXT NOT NULL, payload_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS agent_operation_receipts(
                    operation_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, buyer_id TEXT NOT NULL,
                    action_hash TEXT NOT NULL, result_json TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS agent_resume_requests(
                    task_id TEXT NOT NULL, request_id TEXT NOT NULL, request_hash TEXT NOT NULL,
                    run_id TEXT NOT NULL, PRIMARY KEY(task_id,request_id));
            ''')
            versions = [r[0] for r in db.execute('SELECT version FROM agent_s4_migrations')]
            if versions and versions != [1]:
                raise TaskConflict('CHECKPOINT_INCOMPATIBLE')
            db.execute('INSERT OR IGNORE INTO agent_s4_migrations VALUES(1)')

    def owned(self, task_id, buyer_id, session_id=None):
        """任务归属以业务表为准；客户端 task/thread 标识不提供授权。"""
        with closing(self.store.connect()) as db:
            row = db.execute('SELECT * FROM agent_tasks WHERE task_id=?', (task_id,)).fetchone()
            life = db.execute('SELECT * FROM agent_lifecycle WHERE task_id=?', (task_id,)).fetchone()
        if not row or row['buyer_id'] != buyer_id or (session_id is not None and row['session_id'] != session_id):
            raise TaskConflict('TASK_FORBIDDEN')
        if not life:
            raise TaskConflict('CHECKPOINT_INCOMPATIBLE')
        result = dict(life)
        result['metadata'] = {k: row[k] for k in ('task_id','buyer_id','session_id','engine','engine_version',
                            'graph_version','schema_version','model_mode')}
        result['metadata'].update(run_id=life['last_run_id'], limits=json.loads(row['limits_json']))
        result['budget'] = json.loads(result.pop('budget_json'))
        result['pending_action'] = json.loads(result.pop('pending_json') or 'null')
        # 租约过期表示原执行者不再拥有执行权；读取不自动驱动执行图。
        if result['status'] == 'running' and (result['lease_until'] or 0) <= self.clock():
            result['status'] = 'interrupted'
        return result

    def begin(self, metadata):
        """首次执行只有一次认领机会，重入必须使用 resume。"""
        now = self.clock()
        with self.store.transaction() as db:
            if db.execute('SELECT 1 FROM agent_lifecycle WHERE task_id=?', (metadata.task_id,)).fetchone():
                raise TaskConflict('TASK_ALREADY_STARTED')
            db.execute('INSERT INTO agent_lifecycle VALUES(?,?,1,?,NULL,?,?,?, ?,NULL,?)',
                (metadata.task_id, 'running', metadata.run_id, metadata.run_id, now+self.lease_seconds,
                 now, BudgetUsage().model_dump_json(), now))

    def _lease(self, db, metadata):
        """每次持久写检查 fencing 标识与租约，失去执行权的旧协程不得继续提交。"""
        row = db.execute('SELECT * FROM agent_lifecycle WHERE task_id=?', (metadata.task_id,)).fetchone()
        if not row or row['lease_owner'] != metadata.run_id or (row['lease_until'] or 0) <= self.clock():
            raise TaskConflict('TASK_LEASE_LOST')
        return row

    def reserve(self, metadata, kind, usage):
        """模型和工具在执行前预扣账；崩溃后即使结果未知也不会退还次数。"""
        with self.store.transaction() as db:
            self._lease(db, metadata)
            db.execute('INSERT INTO agent_attempts VALUES(?,?,?,?,?,?)',
                (uuid4().hex, metadata.task_id, metadata.run_id, kind, usage.model_dump_json(), self.clock()))
            db.execute('UPDATE agent_lifecycle SET budget_json=?,segment_started=?,updated_at=? WHERE task_id=?',
                (usage.model_dump_json(), self.clock(), self.clock(), metadata.task_id))

    def heartbeat(self, metadata, usage):
        """执行期间续租并保存有效耗时；业务库计数独立于可能落后的图检查点。"""
        now = self.clock()
        with self.store.transaction() as db:
            self._lease(db, metadata)
            db.execute('UPDATE agent_lifecycle SET lease_until=?,budget_json=?,segment_started=?,updated_at=? WHERE task_id=?',
                (now+self.lease_seconds, usage.model_dump_json(), now, now, metadata.task_id))

    def finish(self, metadata, status, usage, checkpoint_id, pending=None):
        """等待/终止时释放租约和有效计时，不占用户思考时间；旧 run 的诊断保持独立。"""
        with self.store.transaction() as db:
            self._lease(db, metadata)
            db.execute('UPDATE agent_lifecycle SET status=?,version=version+1,checkpoint_id=?,lease_owner=NULL,'
                'lease_until=NULL,segment_started=NULL,budget_json=?,pending_json=?,updated_at=? WHERE task_id=?',
                (status, checkpoint_id, usage.model_dump_json(), json.dumps(pending,ensure_ascii=False), self.clock(), metadata.task_id))

    def claim_resume(self, task_id, buyer_id, session_id, expected_version, checkpoint_id, request_id, request_hash):
        """同请求幂等，同 task CAS；原任务只新建 run，不重新登记 task 或重置预算。"""
        self.owned(task_id, buyer_id, session_id)
        with self.store.transaction() as db:
            prior = db.execute('SELECT * FROM agent_resume_requests WHERE task_id=? AND request_id=?', (task_id,request_id)).fetchone()
            if prior:
                if prior['request_hash'] != request_hash:
                    raise TaskConflict('RESUME_REQUEST_CONFLICT')
                return prior['run_id'], False
            row = db.execute('SELECT * FROM agent_lifecycle WHERE task_id=?', (task_id,)).fetchone()
            now = self.clock()
            if row['status'] == 'running' and (row['lease_until'] or 0) > now:
                raise TaskConflict('TASK_BUSY')
            if row['version'] != expected_version or (row['status']=='waiting_confirmation' and row['checkpoint_id'] != checkpoint_id):
                raise TaskConflict('TASK_VERSION_CONFLICT')
            if row['status'] not in {'waiting_confirmation','interrupted','running'}:
                raise TaskConflict('TASK_NOT_RESUMABLE')
            usage = BudgetUsage.model_validate_json(row['budget_json'])
            if row['status'] == 'running':
                # 意外退出最后一段按租约截止保守计费，不包含截止后的长时间停机。
                usage.active_seconds += max(0, (row['lease_until'] or now)-(row['segment_started'] or now))
            run_id = str(uuid4())
            db.execute('INSERT INTO run_owners VALUES(?,?,?)', (run_id,buyer_id,session_id))
            db.execute("INSERT INTO agent_runs VALUES(?,?,'created',NULL,?)", (run_id,task_id,now))
            # 恢复 run 从登记起有独立 SSE 帧；重复请求始终回放同一 run。
            db.execute("INSERT INTO run_streams(run_id,buyer_id,session_id,message_hash,status,updated_at,lease_owner,lease_until) "
                "VALUES(?,?,?,?,'running',?,?,?)",(run_id,buyer_id,session_id,request_hash,now,run_id,now+self.lease_seconds))
            db.execute("INSERT INTO stream_frames(run_id,kind,payload_json,created_at) VALUES(?,'started',?,?)",
                (run_id,json.dumps({'run_id':run_id,'task_id':task_id}),now))
            db.execute('INSERT INTO agent_resume_requests VALUES(?,?,?,?)', (task_id,request_id,request_hash,run_id))
            db.execute("UPDATE agent_lifecycle SET status='running',version=version+1,last_run_id=?,lease_owner=?,"
                'lease_until=?,segment_started=?,budget_json=?,checkpoint_id=?,updated_at=? WHERE task_id=?',
                (run_id,run_id,now+self.lease_seconds,now,usage.model_dump_json(),checkpoint_id,now,task_id))
            return run_id, True

    def invocation(self, metadata, call_id):
        """调用结果与可信工具状态一并读取，用于图检查点落后于工具完成的窗口。"""
        with closing(self.store.connect()) as db:
            row=db.execute('SELECT * FROM agent_invocations WHERE task_id=? AND call_id=?', (metadata.task_id,call_id)).fetchone()
        return dict(row) if row else None

    def save_invocation(self, metadata, call_id, digest, observation, session):
        """结果写后才让图推进；冲突调用不能覆盖原收据。"""
        with self.store.transaction() as db:
            self._lease(db,metadata)
            old=db.execute('SELECT digest FROM agent_invocations WHERE task_id=? AND call_id=?',(metadata.task_id,call_id)).fetchone()
            if old and old[0]!=digest:
                raise TaskConflict('TOOL_CALL_CONFLICT')
            db.execute('INSERT OR IGNORE INTO agent_invocations VALUES(?,?,?,?,?)',
                (metadata.task_id,call_id,digest,observation.model_dump_json(),json.dumps(session,ensure_ascii=False)))


class DurableBudget(BudgetTracker):
    """预算算法复用基础组件，追加同步短事务预扣账；不在网络请求后才计数。"""

    def __init__(self, metadata, ledger, usage):
        super().__init__(metadata.limits,usage)
        self.metadata,self.ledger=metadata,ledger

    def reserve_model(self, *, decision=True):
        """新网络尝试计数，即使重放失败节点也不获得免费调用。"""
        super().reserve_model(decision=decision)
        self.ledger.reserve(self.metadata,'model_decision' if decision else 'model',self.snapshot())

    def reserve_tool(self):
        """拒绝、失败和重试也计入同一任务账本。"""
        super().reserve_tool()
        self.ledger.reserve(self.metadata,'tool',self.snapshot())
