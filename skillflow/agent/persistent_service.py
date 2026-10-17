"""S4 应用服务：实际图 Saver、显式恢复、运行租约和独立响应；不在 API 内执行业务规则。"""

import asyncio
import json
import re
from contextlib import suppress
from uuid import uuid4

from langgraph.types import Command
from langgraph.errors import NodeCancelledError

from .adapters.checkpoints import SqliteGraphCheckpoints
from .adapters.conversation import SqliteConversationContext
from .adapters.events import SqliteEventSink
from .adapters.task_store import SqliteTaskStore
from .contracts import AgentState, BudgetUsage, MessageEnvelope, RuntimeEvent, TaskMetadata
from .durable_store import DurableStore, DurableBudget, TaskConflict
from .loop_graph import LoopContext
from .loop_service import CollectedEvents, response_projection
from .persistent_graph import build_persistent_graph
from .tool_runtime import ToolRuntime
from .tool_registry import LocalToolRegistry
from .skills import FileSkillRegistry
from .adapters.local_tools import fingerprint
from .budget import BudgetExceeded
from skillflow.business.ticket_commands import TicketCommandService

ENGINE_VERSION='tool_loop_s4_v1'
GRAPH_VERSION='persistent_tool_loop_v1'


class PersistentAgentService:
    """每次请求独立上下文；同 task 的竞争由数据库 CAS/租约保护，图统一调度。"""

    def __init__(self,store,settings,repository,policies,model):
        self.store,self.settings,self.model=store,settings,model
        self.tasks,self.events=SqliteTaskStore(store),SqliteEventSink(store)
        self.ledger=DurableStore(store)
        self.checkpoints=SqliteGraphCheckpoints(store.root)
        self.conversations=SqliteConversationContext(store,settings.agent_context_ttl_seconds)
        self.tools=ToolRuntime(store,repository,policies,settings.timeout)
        self.tools.registry=LocalToolRegistry(incident_available=True,writes_enabled=True)
        self.tools.skills=FileSkillRegistry(self.tools.registry.names,s4=True)
        self.commands=TicketCommandService(self.ledger,repository,policies,settings.agent_confirmation_seconds)
        self.failpoint=None
        # S5 版本与资源由装配层显式注入，S4 默认行为及工具内容锁不变。
        self.engine_version, self.graph_version, self.stage = ENGINE_VERSION, GRAPH_VERSION, 'S4'
        self.mcp_manager = None

    async def start(self):
        """生命周期启动资源；连接失败只降级物流，不影响本地售后。"""
        if self.mcp_manager:
            await self.mcp_manager.start()

    async def aclose(self):
        """先退出子进程，再释放模型 HTTP 客户端。"""
        try:
            if self.mcp_manager:
                await self.mcp_manager.aclose()
        finally:
            if hasattr(self.model, 'aclose'):
                await self.model.aclose()

    @staticmethod
    def graph_config(metadata):
        """thread_id 只由可信归属和图版本计算，不能由网页任意指定。"""
        return {'configurable':{'thread_id':fingerprint({'buyer':metadata.buyer_id,'task':metadata.task_id,'graph':metadata.graph_version})},
                'recursion_limit':128}

    def check_version(self,metadata):
        """旧图/技能不能静默改成新图执行；旧 run 仍可只读查询。"""
        if (metadata.engine_version,metadata.graph_version,metadata.model_mode)!=(self.engine_version,self.graph_version,'langchain_openai'):
            raise TaskConflict('CHECKPOINT_INCOMPATIBLE')

    async def detail(self,task_id,buyer_id,session_id=None):
        """任务查询不驱动图，不续确认期限，不请求模型或工具。"""
        item=await asyncio.to_thread(self.ledger.owned,task_id,buyer_id,session_id)
        metadata=TaskMetadata.model_validate(item['metadata'])
        self.check_version(metadata)
        async with self.checkpoints.open() as saver:
            snapshot=await build_persistent_graph(saver).aget_state(self.graph_config(metadata))
        checkpoint_id=(snapshot.config or {}).get('configurable',{}).get('checkpoint_id')
        pending=item['pending_action']
        if item['status']=='interrupted' and snapshot.values:
            # 意外退出可能发生在 Saver 已暂停、业务生命周期尚未写 waiting 的窗口。
            pending=snapshot.values['payload'].get('constraints',{}).get('pending_action',pending)
        return {'task_id':task_id,'status':item['status'],'version':item['version'],
            'checkpoint_id':checkpoint_id,'last_run_id':item['last_run_id'],'budget':item['budget'],
            'pending_action':pending,
            'resumable':item['status'] in {'waiting_confirmation','interrupted'} and bool(snapshot.next),
            'engine_version':metadata.engine_version,'graph_version':metadata.graph_version}

    async def list_tasks(self,buyer_id,session_id):
        """同会话可以有多个任务；只公开当前买家的受控任务摘要。"""
        from contextlib import closing
        with closing(self.store.connect()) as db:
            ids=[r[0] for r in db.execute('SELECT t.task_id FROM agent_tasks t JOIN agent_lifecycle l ON l.task_id=t.task_id '
                'WHERE t.buyer_id=? AND t.session_id=? ORDER BY t.created_at DESC LIMIT 30',(buyer_id,session_id))]
        return [await self.detail(task,buyer_id,session_id) for task in ids]

    async def run(self,run_id,buyer_id,session_id,message):
        """新任务与 S3 一样先登记归属，再认领；检查点从实际图初始输入开始。"""
        metadata=await self.tasks.get_owned(run_id,buyer_id,session_id)
        self.check_version(metadata)
        await asyncio.to_thread(self.ledger.begin,metadata)
        if re.search(r'改成|改为',message):
            ids=re.findall(r'(?<![A-Za-z0-9])A[0-9]{3,8}(?![A-Za-z0-9])',message.upper())
            if len(set(ids))==1:
                # 明确变更当前这笔申请时撤销旧卡片；历史 run 仍保留原响应用于回放。
                with self.store.transaction() as db:
                    rows=db.execute("SELECT * FROM agent_confirmations WHERE buyer_id=? AND session_id=? AND status='pending'",
                        (buyer_id,session_id)).fetchall()
                    for row in rows:
                        if json.loads(row['payload_json'])['order_id']==ids[0]:
                            db.execute("UPDATE agent_confirmations SET status='invalidated' WHERE confirmation_id=?",(row['confirmation_id'],))
                            db.execute("UPDATE agent_lifecycle SET status='needs_manual_review',version=version+1,pending_json=NULL "
                                "WHERE task_id=? AND status='waiting_confirmation'",(row['task_id'],))
        state=AgentState(metadata=metadata,goal=message,messages=[MessageEnvelope(role='user',content=message)])
        return await self._execute(metadata,state,None)

    async def resume(self,task_id,buyer_id,request):
        """显式 POST，先校验身份、版本与当前卡片，再 CAS 新建同 task 的 run。"""
        item=await self.detail(task_id,buyer_id,request.session_id)
        # 请求幂等先于当前卡片版本判断：上一请求可能已经续卡或完成，重发仍返回它的原 run。
        from contextlib import closing
        with closing(self.store.connect()) as db:
            prior=db.execute('SELECT * FROM agent_resume_requests WHERE task_id=? AND request_id=?',(task_id,request.request_id)).fetchone()
        if prior:
            if prior['request_hash']!=fingerprint(request.model_dump(mode='json')):
                raise TaskConflict('RESUME_REQUEST_CONFLICT')
            saved=self.store.read('run_'+prior['run_id'])
            if saved:
                return saved
            raise TaskConflict('TASK_BUSY')
        if request.checkpoint_id!=item['checkpoint_id'] or request.expected_version!=item['version']:
            # 相同 request_id 的成功重试允许返回原结果，由账本进一步比对请求哈希。
            from contextlib import closing
            with closing(self.store.connect()) as db:
                prior=db.execute('SELECT 1 FROM agent_resume_requests WHERE task_id=? AND request_id=?',(task_id,request.request_id)).fetchone()
            if not prior:
                raise TaskConflict('TASK_VERSION_CONFLICT')
        elif not item['resumable']:
            raise TaskConflict('TASK_BUSY' if item['status']=='running' else 'TASK_NOT_RESUMABLE')
        pending=item['pending_action']
        if item['status']=='waiting_confirmation' and (not pending or request.confirmation_id!=pending['confirmation_id'] or
                request.action_hash!=pending['action_hash'] or request.decision not in {'confirm','refresh'}):
            raise TaskConflict('CONFIRMATION_REQUIRED')
        run_id,created=await asyncio.to_thread(self.ledger.claim_resume,task_id,buyer_id,request.session_id,
            request.expected_version,request.checkpoint_id,request.request_id,fingerprint(request.model_dump(mode='json')))
        if not created:
            saved=self.store.read('run_'+run_id)
            if saved:
                return saved
            raise TaskConflict('TASK_BUSY')
        record=self.store.agent_run(run_id)
        metadata=TaskMetadata.model_validate(record['metadata'])
        return await self._execute(metadata,None,request)

    async def _execute(self,metadata,state,request):
        """等待确认是正常暂停，进程取消是可恢复中断；明确失败保持终态。"""
        item=self.ledger.owned(metadata.task_id,metadata.buyer_id,metadata.session_id)
        budget=DurableBudget(metadata,self.ledger,BudgetUsage.model_validate(item['budget']))
        events=CollectedEvents(self.events)
        conversation=await self.conversations.load(metadata)
        c=LoopContext(self.model,budget,events,self.tasks,self.tools,conversation,self.settings.agent_max_context_chars,
                      context_store=self.conversations)
        c.metadata,c.ledger,c.commands,c.failpoint=metadata,self.ledger,self.commands,self.failpoint
        c.stage=self.stage
        config=self.graph_config(metadata)
        budget.start()
        async def heartbeat():
            """网络请求期间也续租；失败向执行段传播，失去租约不允许继续业务提交。"""
            while True:
                await asyncio.sleep(3)
                await asyncio.to_thread(self.ledger.heartbeat,metadata,budget.snapshot())
                if self.store.stream_record(metadata.run_id):
                    await asyncio.to_thread(self.store.touch_stream,metadata.run_id)
        pulse=asyncio.create_task(heartbeat())
        async with self.checkpoints.open() as saver:
            graph=build_persistent_graph(saver)
            checkpoint_id=None
            try:
                previous=await graph.aget_state(config)
                if request:
                    if not previous.values or not previous.next:
                        raise TaskConflict('CHECKPOINT_INCOMPATIBLE')
                    state=AgentState.model_validate(previous.values['payload'])
                    state.metadata=metadata
                    c.current_state=state
                    await events.append(RuntimeEvent(metadata=metadata,node='task_resumed',status='running',budget=budget.snapshot(),
                        details={'stage':self.stage,'previous_checkpoint_id':request.checkpoint_id}))
                    is_interrupt=any(t.interrupts for t in previous.tasks)
                    decision=request.decision
                    confirmation_id=request.confirmation_id
                    if is_interrupt and decision=='continue':
                        # “继续中断任务”不充当确认；先复核并展示卡片，等待明确提交。
                        decision='refresh'
                        confirmation_id=state.constraints['pending_action']['confirmation_id']
                    graph_input=Command(resume={'decision':decision,'confirmation_id':confirmation_id}) if is_interrupt else None
                else:
                    graph_input={'payload':state.model_dump(mode='json')}
                raw=await graph.ainvoke(graph_input,config=config,context=c,durability='sync')
                state=AgentState.model_validate(raw['payload'])
                state.metadata=metadata
                snapshot=await graph.aget_state(config)
                checkpoint_id=snapshot.config['configurable']['checkpoint_id']
                if any(t.interrupts for t in snapshot.tasks):
                    state.status,state.stop_reason='waiting_confirmation','USER_CONFIRMATION_REQUIRED'
                    # 客服只说明草稿与待确认状态；正式提交结果仍以后续事务回执为准。
                    state.answer=('已为您准备好售后申请草稿，暂时还没有提交。\n\n'
                        +state.constraints['pending_action']['summary']
                        +f'\n\n请核对下方申请卡片，确认无误后点击“确认提交”。卡片有效 {self.settings.agent_confirmation_seconds/60:g} 分钟，过期后需要重新核验。')
                elif state.status in {'created','running'}:
                    state.status,state.stop_reason='failed','RUNTIME_FAILED'
            except (asyncio.CancelledError,NodeCancelledError):
                state=getattr(c,'current_state',state)
                state.status,state.stop_reason='interrupted','RUN_INTERRUPTED'
                state.answer='程序执行中断，已保存进度。请查询任务并明确继续；未执行真实退款。'
                snapshot=await asyncio.shield(graph.aget_state(config))
                checkpoint_id=(snapshot.config or {}).get('configurable',{}).get('checkpoint_id')
            except Exception as exc:
                state=getattr(c,'current_state',state)
                code=getattr(exc,'code',None)
                if not code and isinstance(exc,ValueError) and str(exc).isupper():
                    code=str(exc)
                state.status='failed' if code in {None,'MODEL_TIMEOUT','MODEL_FAILED','MODEL_LIMIT','ROUND_LIMIT','TOOL_LIMIT','TASK_TIMEOUT','CHECKPOINT_INCOMPATIBLE','SKILL_VERSION_INCOMPATIBLE'} else 'needs_manual_review'
                state.stop_reason=code or 'RUNTIME_FAILED'
                state.answer='本次申请或调查未完成，原因：'+state.stop_reason+'。已有草稿保留，未执行真实退款。'
                snapshot=await graph.aget_state(config)
                checkpoint_id=(snapshot.config or {}).get('configurable',{}).get('checkpoint_id')
            finally:
                pulse.cancel()
                with suppress(asyncio.CancelledError):
                    await pulse
                budget.pause()
            state.metadata=metadata
            state.budget=budget.snapshot()
            state.events=list(events.items)
            await events.append(RuntimeEvent(metadata=metadata,node='finish',status=state.status,budget=state.budget,
                details={'stage':self.stage,'stop_reason':state.stop_reason}))
            state.events=list(events.items)
            await self.tasks.save_state(state)
            await asyncio.to_thread(self.ledger.finish,metadata,state.status,state.budget,checkpoint_id,state.constraints.get('pending_action'))
            await self.conversations.save_outcome(state)
        return state


def persistent_projection(state):
    """只公开业务卡片和结果；可信证据仓、工具面、租约及内部恢复状态不对外导出。"""
    citations,result=response_projection(state)
    result.update(read_only=not bool((result.get('task_goal') or {}).get('application')),
        stage='S5' if state.metadata.engine_version.startswith('tool_loop_s5_') else 'S4',pending_action=state.constraints.get('pending_action'),ticket=state.constraints.get('ticket'),
        task_status=state.status)
    return citations,result
