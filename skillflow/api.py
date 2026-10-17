import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import time
from uuid import UUID, uuid4
from urllib.parse import parse_qs
from weakref import WeakSet, WeakValueDictionary

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse

from skillflow.business.repository import MockOrderRepository
from skillflow.business.surge import BusinessRepository
from skillflow.business.incident_service import IncidentService
from skillflow.admission import ChatAdmission
from skillflow.agent.bootstrap import create_service, task_metadata
from skillflow.agent.adapters.events import to_trace
from skillflow.auth import AUTH_COOKIE, CSRF_COOKIE, AuthError, AuthService, Principal
from skillflow.business.policies import LocalPolicyRepository
from skillflow.config import Settings
from skillflow.business.sqlite_tickets import SqliteTicketStore
from skillflow.contracts import ChatRequest, ChatResponse, StreamChatRequest, TicketCancelRequest, TicketConfirmRequest, TicketDraftRequest, TicketResponse, TraceEvent
from skillflow.planner import Planner, extract_order_ids
from skillflow.runtime import AgentRuntime
from skillflow.dialogue_runtime import DialogueRuntime, bind_ticket, clear_pending
from skillflow.skills.registry import load_skills
from skillflow.sqlite_store import SqliteStore
from skillflow.session_checkpoint import SessionCheckpoint
from skillflow.observability import RequestMetrics
from skillflow.ticket_runtime import TicketRuntime
from skillflow.streaming import progress_sink, sse_frame
from skillflow.tools import create_tools
from skillflow.contracts import TicketEvent


stream_logger = logging.getLogger("skillflow.stream")


def create_app(settings=None, policies=None, planner=None, identity_provider=None):
    """组装项目 B 独立应用；测试可替换政策库接口，无需外部服务。"""
    settings = settings or Settings()
    # 所有会话、Trace、工单和历史使用项目 B 自有 SQLite；旧 JSON 首次启动只导入不删除。
    store = SqliteStore(settings.data_dir)
    ticket_store = SqliteTicketStore(store)
    repository = (BusinessRepository(store.database_path, settings.fixtures_dir)
                  if settings.business_backend == "sqlalchemy" else MockOrderRepository(settings.fixtures_dir))
    incident_service = (IncidentService(repository, ticket_store, settings.model_mode)
                        if settings.business_backend == "sqlalchemy" else None)
    if settings.auth_mode == "demo" and not repository.has_buyer(settings.demo_buyer_id):
        raise ValueError(f"Unknown demo buyer: {settings.demo_buyer_id}")
    skills = load_skills()
    # 启动时只索引本项目 data/policies，既不请求外部服务，也不读取其他项目。
    policies = policies or LocalPolicyRepository(settings.fixtures_dir / "policies")
    planner_service = planner or Planner(settings, skills)
    # 每个买家持有独立的工具闭包与对话图，请求间不会覆盖全局的当前买家。
    # 买家运行时最多缓存 64 份，避免一千位演示买家的缓存无限增长。
    bundles = OrderedDict()
    def buyer_bundle(buyer_id: str):
        if buyer_id not in bundles:
            runtime = AgentRuntime(planner_service, skills, create_tools(repository, buyer_id, policies), settings.timeout)
            ticket_runtime = TicketRuntime(runtime, ticket_store, buyer_id)
            dialogue_runtime = DialogueRuntime(runtime, ticket_runtime, ticket_store, settings, buyer_id)
            bundles[buyer_id] = (runtime, ticket_runtime, dialogue_runtime)
            if len(bundles) > 64:
                bundles.popitem(last=False)
        bundles.move_to_end(buyer_id)
        return bundles[buyer_id]

    auth = AuthService(store, settings, identity_provider)
    runtime, ticket_runtime, dialogue_runtime = buyer_bundle(settings.demo_buyer_id) if settings.auth_mode == "demo" else (None, None, None)
    session_checkpoint = SessionCheckpoint(store)
    @asynccontextmanager
    async def lifespan(application):
        """通过标准 lifespan 释放新图客户端，兼容当前 FastAPI 的生命周期接口。"""
        try:
            if hasattr(application.state.agent_service, 'start'):
                # SDK 上下文由受管任务拥有，应用资源在接收请求前初始化。
                await application.state.agent_service.start()
            yield
        finally:
            service = application.state.agent_service
            if hasattr(service, 'aclose'):
                await service.aclose()
            else:
                model = getattr(service, "model", None)
                if hasattr(model, "aclose"):
                    await model.aclose()

    app = FastAPI(title="SkillFlow：数码小店调查 Agent", version="0.2.0", lifespan=lifespan)
    app.state.runtime, app.state.store, app.state.policies, app.state.tickets = runtime, store, policies, ticket_store
    app.state.ticket_runtime = ticket_runtime
    app.state.dialogue_runtime = dialogue_runtime
    app.state.auth = auth
    app.state.session_checkpoint = session_checkpoint
    app.state.request_metrics = RequestMetrics()
    app.state.repository = repository
    app.state.incident_service = incident_service
    # 装配不发网络请求；新图只能在受理预检通过后执行，旧路径继续使用原运行时。
    app.state.agent_service = create_service(store, settings, repository, policies)

    @app.middleware("http")
    async def observe_request(request: Request, call_next):
        """统计响应状态和建连耗时；SSE 的数值只覆盖响应头，不代表任务完成耗时。"""
        started = time.perf_counter()
        request_id = uuid4().hex
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            route = request.scope.get("route")
            # 找不到路由时只记固定标签，避免把未知路径或令牌写入日志。
            route_name = route.path if route is not None else "<unmatched>"
            app.state.request_metrics.record(
                request.method, route_name, status,
                (time.perf_counter() - started) * 1000, request_id)
    policy_lock = asyncio.Lock()
    # 同进程会话先串行；跨进程工单幂等由 SQLite 事务与唯一索引保证。
    # 会话上下文按买家和会话串行；弱引用在空闲后回收锁，避免用户量扩大后内存常驻。
    session_locks = WeakValueDictionary()
    global_chat_lock = asyncio.Lock()
    def session_lock(buyer_id: str, session_id: str):
        if settings.chat_lock_mode == "global":
            return global_chat_lock
        key = (buyer_id, session_id)
        lock = session_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            session_locks[key] = lock
        return lock

    admission = ChatAdmission(settings.max_active_chats, settings.max_waiting_chats)
    app.state.chat_admission = admission
    active_streams = {}
    # 弱引用使完成或取消的 SSE 订阅不常驻内存。
    class StreamNotifier:
        def __init__(self):
            self.watchers = WeakSet()

    stream_watchers = WeakValueDictionary()

    def notifier_for(run_id: str):
        notifier = stream_watchers.get(run_id)
        if notifier is None:
            notifier = StreamNotifier()
            stream_watchers[run_id] = notifier
        return notifier

    def wake_stream(run_id: str):
        """唤醒本进程 SSE 订阅者；跨进程由低频数据库重放兜底。"""
        notifier = stream_watchers.get(run_id)
        if notifier:
            for watcher in tuple(notifier.watchers):
                watcher.set()

    def current_principal(request: Request) -> Principal:
        """在 Planner、模型和工具之前验证服务端会话；失败统一返回 401。"""
        try:
            return auth.principal(request)
        except AuthError as exc:
            raise HTTPException(401, "需要登录。") from exc

    def writing_principal(request: Request, principal: Principal = Depends(current_principal)) -> Principal:
        """所有改变会话或工单的 Cookie 请求都必须通过 CSRF 校验。"""
        try:
            auth.verify_csrf(request, principal)
        except AuthError as exc:
            raise HTTPException(403, "CSRF 校验失败。") from exc
        return principal

    def deny_access(code: str):
        """统一返回资源越权的 HTTP 403 和稳定错误码，不附带其他买家的资源内容。"""
        raise HTTPException(status_code=403, detail={"code": code, "message": "无权访问。"})

    def require_conversation(principal: Principal, session_id: str):
        """生产模式区分不存在的会话与其他买家的会话，先校验再读写上下文。"""
        if settings.auth_mode != "production":
            return
        owner = store.conversation_owner(session_id)
        if owner is None:
            raise HTTPException(404, "会话不存在。")
        if owner != principal.buyer_id:
            deny_access("SESSION_FORBIDDEN")

    def require_order(principal: Principal, order_id: str):
        """已存在的他人订单返回 403；未知订单交由具体查询流程返回 404 或澄清。"""
        if settings.auth_mode == "production":
            owner = repository.owner_of_order(order_id) if settings.business_backend == "sqlalchemy" else (
                repository.orders.get(order_id) or {}).get("buyer_id")
            if owner and owner != principal.buyer_id:
                deny_access("ORDER_FORBIDDEN")

    def require_agent_ready(engine: str, model_mode: str):
        """演示或显式 S6 登录接入；配置缺失在占容量和登记前拒绝。"""
        if engine != "tool_loop":
            return
        if settings.auth_mode != "demo" and not settings.agent_s6_enabled:
            raise HTTPException(503, detail={"code": "ENGINE_DEMO_ONLY", "message": "S1 新引擎仅开放独立演示。"})
        if model_mode not in {"scripted", "langchain_openai"} or (model_mode == "langchain_openai" and not (settings.model and settings.api_key)):
            raise HTTPException(503, detail={"code": "MODEL_CAPABILITY_UNSUPPORTED",
                                            "message": "新引擎需要显式 scripted，或配置完整的 langchain_openai 模式。"})

    def require_ticket_writable():
        """S3 网页及直接 HTTP 都禁止工单写入；S4 才接确认与事务恢复。"""
        if settings.agent_engine == "tool_loop" and settings.agent_model_mode == "langchain_openai":
            raise HTTPException(409, detail={"code": "WRITE_NOT_ENABLED", "message": "S3 仅开放只读调查。"})

    def new_task_metadata(run_id: str, principal: Principal, session_id: str):
        """首次受理冻结引擎和预算，后续排队或 SSE 重放不能重新选引擎。"""
        require_agent_ready(settings.agent_engine, settings.agent_model_mode)
        return task_metadata(settings, run_id, principal.buyer_id, session_id)

    def require_ticket(principal: Principal, ticket_id: str):
        """工单读取和状态变更共用同一权限检查，避免遗漏某个入口。"""
        record = ticket_store.get(ticket_id)
        if not record:
            raise HTTPException(404, "工单不存在。")
        if record["buyer_id"] != principal.buyer_id:
            deny_access("TICKET_FORBIDDEN")
        return record

    async def save_context(buyer_id, session_id, context):
        """会话键包含服务端买家身份，浏览器传入的会话 ID 不充当身份凭证。"""
        # 先写入 LangGraph 检查点，再镜像 SQLite 会话数据，失败时不返回成功。
        await session_checkpoint.save(buyer_id, session_id, context)

    async def sync_ticket_context(buyer_id, session_id, state):
        """网页按钮和聊天共用当前工单引用；确认信息只来自实际草稿。"""
        if state.get("ticket"):
            context = await session_checkpoint.load(buyer_id, session_id)
            bind_ticket(context, state["ticket"])
            await save_context(buyer_id, session_id, context)

    @app.get("/")
    async def home():
        """聊天页面与 API 同源部署，引用也回到本服务读取。"""
        return FileResponse(Path(__file__).parent / "web" / "index.html")

    @app.get('/assets/chat-progress.js', include_in_schema=False)
    async def chat_progress_script():
        """固定静态资源路径，不开放任意文件读取；脚本只负责前端等待反馈。"""
        return FileResponse(Path(__file__).parent / 'web' / 'chat_progress.js', media_type='text/javascript')

    @app.get("/auth/login")
    async def auth_login():
        """生产模式跳转可信身份提供方；演示模式不开放模拟登录接口。"""
        if settings.auth_mode != "production":
            raise HTTPException(404, "接口不存在。")
        try:
            return RedirectResponse(await auth.begin_login())
        except AuthError as exc:
            raise HTTPException(401, "登录初始化失败。") from exc

    @app.post("/auth/callback")
    async def auth_callback(request: Request):
        """从 form_post 请求体读取授权码，避免访问日志的 URL 记录令牌。"""
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/x-www-form-urlencoded":
            raise HTTPException(400, "登录回调格式无效。")
        body = await request.body()
        if len(body) > 8192:
            raise HTTPException(400, "登录回调过大。")
        try:
            fields = parse_qs(body.decode("utf-8"), strict_parsing=True)
            if len(fields.get("state", [])) != 1 or len(fields.get("code", [])) != 1:
                raise ValueError("重复登录字段")
            state, code = fields["state"][0], fields["code"][0]
        except (UnicodeError, ValueError, KeyError, IndexError) as exc:
            raise HTTPException(400, "登录回调格式无效。") from exc
        try:
            token, csrf, _ = await auth.finish_login(state, code, repository)
        except AuthError as exc:
            raise HTTPException(401, "登录验证失败。") from exc
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(AUTH_COOKIE, token, max_age=settings.auth_session_seconds,
                            httponly=True, secure=True, samesite="lax", path="/")
        response.set_cookie(CSRF_COOKIE, csrf, max_age=settings.auth_session_seconds,
                            httponly=False, secure=True, samesite="lax", path="/")
        return response

    @app.get("/auth/me")
    async def auth_me(principal: Principal = Depends(current_principal)):
        """仅返回当前身份的公开字段，供刷新页面恢复登录状态。"""
        from skillflow.business.s6_samples import examples_for
        # 能力配置仅登录后公开，不向模型或前端泄露身份凭据及服务密钥。
        return {"buyer_id": principal.buyer_id, "role": principal.role, "auth_mode": settings.auth_mode,
                "runtime": {"agent_engine": settings.agent_engine, "agent_model_mode": settings.agent_model_mode,
                    "agent_s4_enabled": settings.agent_s4_enabled, "agent_s5_enabled": settings.agent_s5_enabled,
                    "agent_s6_enabled": settings.agent_s6_enabled, "business_backend": settings.business_backend,
                    "agent_confirmation_seconds": settings.agent_confirmation_seconds,
                    "agent_context_ttl_seconds": settings.agent_context_ttl_seconds,
                    "examples": examples_for(settings.fixtures_dir, principal.buyer_id, repository)
                        if settings.agent_s6_enabled else []}}

    @app.post("/auth/logout")
    async def auth_logout(principal: Principal = Depends(writing_principal)):
        """先撤销本地会话，再给浏览器可信 IdP 登出地址，防止原 SSO 静默登录。"""
        auth.logout(principal)
        try:
            logout_url = await auth.end_session_url()
        except AuthError:
            # 身份提供方暂不可用时仍完成本地退出，不能让旧 Cookie 恢复访问。
            logout_url = None
        response = JSONResponse({"logout_url": logout_url})
        response.delete_cookie(AUTH_COOKIE, path="/")
        response.delete_cookie(CSRF_COOKIE, path="/")
        return response

    @app.post("/sessions")
    async def create_session(principal: Principal = Depends(writing_principal)):
        """由服务端创建买家专属对话 ID，浏览器不能指定它的归属。"""
        return {"session_id": store.create_conversation(principal.buyer_id)}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str, principal: Principal = Depends(current_principal)):
        """刷新页面时核对会话归属，未知或外买家 ID 统一返回 404。"""
        require_conversation(principal, session_id)
        return {"session_id": session_id}

    @app.get("/health")
    async def health():
        """仅报告本应用状态，不探测其他项目的健康接口。"""
        mcp = app.state.agent_service.mcp_manager.health() if settings.agent_s5_enabled else None
        status = 'degraded' if mcp and mcp['status'] != 'ready' else 'ok'
        if settings.auth_mode == "production":
            # 原登录部署保持匿名健康返回兼容；仅启用物流时增加其就绪状态。
            return {"status": status, "auth_mode": "production", **({"logistics_mcp": mcp} if mcp else {})}
        # 新图展示实际选择的模型模式，避免沿用旧 Planner 配置而误标为真实模型。
        return {"status": status, "logistics_mcp": mcp, "auth_mode": "demo", "model_mode": (
                    settings.agent_model_mode if settings.agent_engine == "tool_loop" else settings.model_mode),
                "model_trust_env": settings.model_trust_env, "order_mode": "mock_json",
                "dialogue_version": "structured_dialogue_v1", "product_confirmation": settings.confirm_product,
                "model_configured": bool(settings.model and settings.api_key) and (
                    settings.agent_model_mode == "langchain_openai" if settings.agent_engine == "tool_loop" else settings.model_mode != "mock"),
                "buyer_id": settings.demo_buyer_id,
                "business_backend": settings.business_backend,
                "agent_engine": settings.agent_engine, "agent_model_mode": settings.agent_model_mode,
                "agent_runtime_stage": "S5_mcp" if settings.agent_s5_enabled else "S4_persistent" if settings.agent_s4_enabled else "S3_readonly" if settings.agent_engine == "tool_loop" and settings.agent_model_mode == "langchain_openai" else "S1_skeleton",
                "agent_context_ttl_seconds": settings.agent_context_ttl_seconds,
                "read_only": settings.agent_engine == "tool_loop" and settings.agent_model_mode == "langchain_openai" and not settings.agent_s4_enabled,
                "agent_s4_enabled": settings.agent_s4_enabled, "agent_confirmation_seconds": settings.agent_confirmation_seconds,
                "agent_s5_enabled": settings.agent_s5_enabled,
                "chat_admission": admission.snapshot(),
                "available_demo_buyers": [buyer["buyer_id"] for buyer in repository.list_buyers()],
                "policy_mode": "local_bm25_v1", "policy_source": "project_b_local",
                "policy_registered": bool(policies.registered_policies().get("earbuds_v1")),
                "ticket_mode": "sqlite_with_transactional_history_and_idempotency",
                "ticket_conflict_mode": "pre_draft_gate_v2",
                "session_mode": "sqlite_langgraph_checkpointer",
                "stream_mode": "sqlite_sse_replay_v1",
                "skills": list(skills.values())}

    @app.get("/demo/buyers")
    async def demo_buyers():
        """返回本地演示买家目录；该接口只服务于开发验收，不代表生产登录接口。"""
        if settings.auth_mode != "demo":
            raise HTTPException(404, "接口不存在。")
        return {"current_buyer_id": settings.demo_buyer_id, "buyers": repository.list_buyers()}

    @app.post("/demo/setup-policy")
    async def setup_policy():
        """重新索引本项目政策；保留页面按钮接口，但不再执行上传或跨服务导入。"""
        if settings.auth_mode != "demo":
            raise HTTPException(404, "接口不存在。")
        async with policy_lock:
            try:
                if settings.agent_s4_enabled:
                    # 与 S4 提交复核串行重建，避免政策字典在事务复核中途被替换。
                    with app.state.agent_service.commands.policy_lock:
                        policies.reload()
                else:
                    policies.reload()
                return {"status": "ready", "documents": policies.registered_policies()}
            except (OSError, ValueError):
                raise HTTPException(500, "本地政策索引失败，请检查项目 B 的 data/policies。")

    @app.get("/policies/{document_id}/source")
    async def policy_source(document_id: str, principal: Principal = Depends(current_principal)):
        """根据已登记文档 ID 返回本项目原文；未知 ID 不映射为任意路径。"""
        path = policies.source(document_id)
        if not path or not path.is_file():
            raise HTTPException(404, "本地政策文档不存在。")
        return FileResponse(path, media_type="text/plain; charset=utf-8")

    @app.get("/demo/orders")
    async def demo_orders():
        """只展示服务端演示用户的订单，不暴露其他用户数据。"""
        if settings.auth_mode != "demo":
            raise HTTPException(404, "接口不存在。")
        return (repository.list_orders(settings.demo_buyer_id)
                if settings.business_backend == "sqlalchemy" else
                [order for order_id in repository.orders
                 if (order := repository.get_order(order_id, settings.demo_buyer_id))])

    @app.get("/orders/{order_id}")
    async def order_detail(order_id: str, principal: Principal = Depends(current_principal)):
        """先识别已存在的他人订单并拒绝，再读取当前买家的订单详情。"""
        require_order(principal, order_id)
        order = repository.get_order(order_id, principal.buyer_id)
        if not order:
            raise HTTPException(404, "订单不存在。")
        return order

    async def execute_chat(request: ChatRequest, run_id: str, principal: Principal, queue_wait_ms: float = 0) -> ChatResponse:
        """在同一个业务锁内执行对话图并持久化买家隔离的上下文和 Trace。"""
        if not request.message.strip():
            raise HTTPException(422, "消息不能为空。")
        require_conversation(principal, request.session_id)
        for order_id in extract_order_ids(request.message):
            require_order(principal, order_id)
        session_queued_at = time.perf_counter()
        async with session_lock(principal.buyer_id, request.session_id):
            session_wait_ms = (time.perf_counter() - session_queued_at) * 1000
            # 路由必须先于质量事件分支；只读冻结元数据，不按当前配置重选引擎。
            registration = await asyncio.to_thread(store.agent_run, run_id)
            if registration and registration["metadata"]["engine"] == "tool_loop":
                metadata = registration["metadata"]
                require_agent_ready(metadata["engine"], metadata["model_mode"])
                state = await app.state.agent_service.run(run_id, principal.buyer_id, request.session_id, request.message)
                if metadata["model_mode"] == "langchain_openai":
                    from skillflow.agent.loop_service import response_projection
                    if settings.agent_s4_enabled:
                        from skillflow.agent.persistent_service import persistent_projection as response_projection
                    citations, projection = response_projection(state)
                    # 完整响应包含排队；独立字段让验收定位准入与同会话锁等待。
                    projection['timing'] = {'admission_wait_ms': queue_wait_ms, 'session_wait_ms': session_wait_ms}
                    if settings.agent_s4_enabled:
                        projection['task'] = await app.state.agent_service.detail(state.metadata.task_id,principal.buyer_id,request.session_id)
                    result = ChatResponse(run_id=run_id, session_id=request.session_id, status=state.status,
                        answer=state.answer, model_mode="langchain_openai", citations=citations,
                        events=[to_trace(event) for event in state.events], result=projection)
                    await asyncio.to_thread(store.write, "run_" + run_id, result.model_dump())
                    return result
                result = ChatResponse(
                    run_id=run_id, session_id=request.session_id, status=state.status,
                    answer=state.answer, model_mode="scripted", events=[to_trace(event) for event in state.events],
                    result={"execution_mode": "simulation", "business_completed": False,
                            "metadata": metadata, "stop_reason": state.stop_reason,
                            "budget": state.budget.model_dump(), "token_usage": "unavailable"})
                # 脚本答复只存本 run，不写入 legacy 的订单、事件或工单对话上下文。
                await asyncio.to_thread(store.write, "run_" + run_id, result.model_dump())
                return result
            # 确定性事件咨询只需持久化当前订单和消息；独立 KV 避免每次建立 LangGraph
            # Checkpointer 并争抢 SQLite 写锁。原 Agent 对话仍保持原检查点语义。
            incident_key = "incident_" + store.session_key(f"{principal.buyer_id}:{request.session_id}")
            incident_context = (await asyncio.to_thread(store.read, incident_key) or {}) if incident_service else {}
            # 质量事件场景使用可信批次与已审核版本的确定性流程；旧商品仍用原 Agent 图。
            incident_route = (incident_service is not None and
                              await asyncio.to_thread(incident_service.should_handle,
                                                      request.message, incident_context, principal.buyer_id))
            if incident_route:
                context = incident_context
                result = await incident_service.chat(message=request.message, session_id=request.session_id,
                                                     run_id=run_id, buyer_id=principal.buyer_id, context=context)
            else:
                context = await session_checkpoint.load(principal.buyer_id, request.session_id)
                dialogue_runtime = buyer_bundle(principal.buyer_id)[2]
                state = await dialogue_runtime.graph.ainvoke({
                    "message": request.message, "session_id": request.session_id, "context": context,
                    "events": [], "results": {}, "citations": []})
                results = dict(state.get("results", {}))
                ticket = state.get("ticket")
                if ticket:
                    results["ticket"] = ticket.model_dump()
                    results["ticket_events"] = [item.model_dump() for item in state.get("ticket_events", [])]
                result = ChatResponse(
                    run_id=run_id, session_id=request.session_id, status=state["status"],
                    answer=state["answer"], model_mode=settings.model_mode, dialogue=state.get("dialogue"),
                    plan=state.get("plan"), citations=state.get("citations", []),
                    events=state.get("events", []), result=results)
            context["messages"] = [*context.get("messages", []),
                                   {"role": "user", "content": request.message},
                                   {"role": "assistant", "content": result.answer}][-12:]
            if incident_route:
                await asyncio.to_thread(store.write, incident_key, context)
            else:
                await save_context(principal.buyer_id, request.session_id, context)
            await asyncio.to_thread(store.write, "run_" + result.run_id, result.model_dump())
            return result

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest, principal: Principal = Depends(writing_principal)):
        """保留原有非流式 API，返回经过引用校验的完整回答。"""
        if not request.message.strip():
            raise HTTPException(422, "消息不能为空。")
        require_conversation(principal, request.session_id)
        for order_id in extract_order_ids(request.message):
            require_order(principal, order_id)
        run_id = str(uuid4())
        metadata = new_task_metadata(run_id, principal, request.session_id)
        if not await admission.reserve(principal.buyer_id):
            raise HTTPException(status_code=429, detail={"code": "CHAT_BUSY", "message": "当前请求较多，请稍后重试。"},
                                headers={"Retry-After": "2"})
        try:
            # 在进入执行等待队列前原子登记归属和引擎，数据库失败要退还容量。
            await asyncio.to_thread(store.register_run, run_id, principal.buyer_id, request.session_id, metadata)
        except BaseException:
            await admission.release(principal.buyer_id)
            raise
        async with admission.run(principal.buyer_id) as queue_wait_ms:
            return await execute_chat(request, run_id, principal, queue_wait_ms)

    async def stream_frames(run_id: str, after: int = 0):
        """从 SQLite 顺序读帧；浏览器断开连接不会取消后台执行。"""
        last_seq, silent_ticks = after, 0
        notifier = notifier_for(run_id)
        watcher = asyncio.Event()
        notifier.watchers.add(watcher)
        while True:
            frames = store.stream_frames(run_id, last_seq)
            for frame in frames:
                last_seq = frame["seq"]
                yield sse_frame(frame)
            record = store.stream_record(run_id)
            if (record["status"] == "running" and time.time() - record["updated_at"] > 15 or
                    record["status"] == "queued" and time.time() - record["updated_at"] > 300):
                await asyncio.to_thread(store.interrupt_stream, run_id)
                # 只有真正补写终态才立即重放；活租约对应其他进程时继续正常等待。
                if store.stream_record(run_id)["status"] not in {"running", "queued"}:
                    continue
            if record["status"] not in {"running", "queued"}:
                # 状态和帧分两次读取；终态可能恰好在两次读取之间提交，退出前补读以免漏掉 final。
                for frame in store.stream_frames(run_id, last_seq):
                    last_seq = frame["seq"]
                    yield sse_frame(frame)
                break
            # 本进程写入即时唤醒；两秒一次兜底读取其他进程的数据库事件。
            try:
                await asyncio.wait_for(watcher.wait(), timeout=2)
            except asyncio.TimeoutError:
                pass
            watcher.clear()
            silent_ticks += 1
            if silent_ticks >= 5:
                # 注释帧供代理和浏览器保持连接，不占用可回放的事件序号。
                yield ": keep-alive\n\n"
                silent_ticks = 0

    async def _run_streamed_chat_body(request: StreamChatRequest, principal: Principal, queue_wait_ms: float = 0):
        """后台任务独立于 SSE 连接；异常只发送失败帧，不发布未经校验的回答。"""
        run_id = str(request.run_id)
        if not await asyncio.to_thread(store.claim_stream, run_id, uuid4().hex):
            return
        async def heartbeat():
            """慢模型调用期间续租运行状态，使跨进程的只读重连不会误判中断。"""
            while True:
                await asyncio.sleep(3)
                try:
                    await asyncio.to_thread(store.touch_stream, run_id)
                except Exception:
                    # 数据库短暂繁忙时保留当前任务，由后续心跳与终态写入恢复。
                    pass

        heartbeat_task = asyncio.create_task(heartbeat())
        frame_queue = asyncio.Queue(maxsize=128)
        writer_failure = []

        async def frame_writer():
            """进度顺序落盘；写锁等待放在线程中，不阻塞 Checkpointer 的提交。"""
            try:
                while True:
                    kind, payload = await frame_queue.get()
                    try:
                        await asyncio.to_thread(store.append_stream_frame, run_id, kind, payload)
                        wake_stream(run_id)
                    finally:
                        frame_queue.task_done()
            except Exception as exc:
                writer_failure.append(exc)
                while not frame_queue.empty():
                    frame_queue.get_nowait()
                    frame_queue.task_done()

        writer_task = asyncio.create_task(frame_writer())
        try:
            def persist_progress(kind, payload):
                # 进度数量上限保护内存；溢出视为明确失败而非无限排队。
                frame_queue.put_nowait((kind, payload))

            with progress_sink(persist_progress):
                result = await execute_chat(request, run_id, principal, queue_wait_ms)
            await frame_queue.join()
            if writer_failure:
                raise writer_failure[0]
            await asyncio.to_thread(store.append_stream_frame, run_id, "final", result.model_dump())
            wake_stream(run_id)
        except asyncio.CancelledError:
            await frame_queue.join()
            await asyncio.to_thread(store.append_stream_frame, run_id, "interrupted",
                                    {"message": "服务中断；请先查询工单状态，再重新发送请求。"})
            wake_stream(run_id)
            raise
        except Exception as exc:
            # 只记异常类型与 run_id，不记录订单详情、Cookie 或令牌；重连不能重做写入。
            stream_logger.error("stream_run_failed run_id=%s error_type=%s", run_id, type(exc).__name__)
            await frame_queue.join()
            await asyncio.to_thread(store.append_stream_frame, run_id, "error",
                                    {"code": "CHAT_EXECUTION_FAILED", "message": "执行失败；请查询工单状态后再重试。"})
            wake_stream(run_id)
        finally:
            heartbeat_task.cancel()
            writer_task.cancel()
            await asyncio.gather(heartbeat_task, writer_task, return_exceptions=True)
            active_streams.pop(run_id, None)

    async def run_streamed_chat(request: StreamChatRequest, principal: Principal):
        """有界工作位内执行持久化任务；排队时不占模型并发额度。"""
        try:
            async with admission.run(principal.buyer_id) as queue_wait_ms:
                await _run_streamed_chat_body(request, principal, queue_wait_ms)
        finally:
            active_streams.pop(str(request.run_id), None)

    @app.post("/chat/stream")
    async def chat_stream(request: StreamChatRequest, principal: Principal = Depends(writing_principal)):
        """首次 POST 启动任务；同 run_id 重复 POST 只回放，不重复调用图或工具。"""
        if not request.message.strip():
            raise HTTPException(422, "消息不能为空。")
        require_conversation(principal, request.session_id)
        for order_id in extract_order_ids(request.message):
            require_order(principal, order_id)
        run_id = str(request.run_id)
        # 已存在 run_id 是只读重放；新任务仅在有界容量中登记为 queued。
        reserved = False
        metadata = None
        if store.stream_record(run_id) is None:
            metadata = new_task_metadata(run_id, principal, request.session_id)
            if not await admission.reserve(principal.buyer_id):
                raise HTTPException(status_code=429, detail={"code": "CHAT_BUSY", "message": "当前请求较多，请稍后重试。"},
                                    headers={"Retry-After": "2"})
            reserved = True
        try:
            created = await asyncio.to_thread(store.create_stream, run_id, principal.buyer_id,
                                              request.session_id, request.message, queued=True, agent_metadata=metadata)
        except ValueError:
            if reserved:
                await admission.release(principal.buyer_id)
            # 相同 run_id 指向其他买家是越权；同买家参数冲突仍按 409 处理。
            existing = store.stream_record(run_id)
            if existing and existing["buyer_id"] != principal.buyer_id:
                deny_access("RUN_FORBIDDEN")
            raise HTTPException(409, "run_id 已用于其他请求。")
        except Exception:
            if reserved:
                await admission.release(principal.buyer_id)
            raise
        if created:
            active_streams[run_id] = asyncio.create_task(run_streamed_chat(request, principal))
        elif reserved:
            await admission.release(principal.buyer_id)
        if not created and store.stream_record(run_id)["status"] == "running" and run_id not in active_streams:
            await asyncio.to_thread(store.interrupt_stream, run_id)
        return StreamingResponse(stream_frames(run_id), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})

    @app.get("/runs/{run_id}/stream")
    async def replay_stream(run_id: UUID, session_id: str = Query(min_length=1, max_length=128),
                            after: int = Query(default=0, ge=0),
                            principal: Principal = Depends(current_principal)):
        """按 run_id 和会话只读恢复进度；不触发 Planner、工具或工单节点。"""
        key = str(run_id)
        record = store.stream_record(key)
        if not record:
            raise HTTPException(404, "执行记录不存在。")
        if record["buyer_id"] != principal.buyer_id or record["session_id"] != session_id:
            deny_access("RUN_FORBIDDEN")
        require_conversation(principal, session_id)
        if record["status"] == "running" and key not in active_streams:
            await asyncio.to_thread(store.interrupt_stream, key)
        return StreamingResponse(stream_frames(key, after), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})

    # S4 恢复入口独立复用认证、CSRF、准入与会话锁，不借用旧 tickets 提交入口。
    from skillflow.agent.http import install_agent_routes
    install_agent_routes(app,settings,writing_principal,current_principal,require_conversation,session_lock,admission,wake_stream)

    @app.post("/tickets/draft", response_model=TicketResponse)
    async def create_ticket_draft(request: TicketDraftRequest, principal: Principal = Depends(writing_principal)):
        """运行项目 B 工单 LangGraph 草稿图；此接口不会提交正式工单。"""
        require_ticket_writable()
        require_conversation(principal, request.session_id)
        require_order(principal, request.order_id)
        if incident_service:
            order = await asyncio.to_thread(repository.get_order, request.order_id, principal.buyer_id)
            if order and order.get("category") == "power_bank":
                if request.request_type != "return":
                    return TicketResponse(status="rejected", error_code="INCIDENT_RETURN_ONLY",
                                          message="质量事件演示仅支持退货草稿；其他售后请人工核实。")
                return await incident_service.create_draft(
                    buyer_id=principal.buyer_id, session_id=request.session_id,
                    order_id=request.order_id, reason=request.reason, source_run_id=str(uuid4()))
        async with session_lock(principal.buyer_id, request.session_id):
            state = await buyer_bundle(principal.buyer_id)[1].draft_graph.ainvoke({"request": request.model_dump(), "events": [], "citations": []})
            await sync_ticket_context(principal.buyer_id, request.session_id, state)
            return TicketResponse(status=state["status"], message=state["message"], error_code=state.get("error_code"),
                                  ticket=state.get("ticket"), ticket_events=state.get("ticket_events", []),
                                  citations=state.get("citations", []), events=state["events"])

    @app.post("/tickets/{ticket_id}/confirm", response_model=TicketResponse)
    async def confirm_ticket(ticket_id: str, request: TicketConfirmRequest, principal: Principal = Depends(writing_principal)):
        """运行工单 LangGraph 确认图；重复确认不会生成第二张工单。"""
        require_ticket_writable()
        require_conversation(principal, request.session_id)
        record = require_ticket(principal, ticket_id)
        if incident_service and record["metadata"].get("incident_id"):
            return await incident_service.confirm(buyer_id=principal.buyer_id, session_id=request.session_id,
                                                  ticket_id=ticket_id, confirmed=request.confirmed)
        async with session_lock(principal.buyer_id, request.session_id):
            # 不区分“不存在”和“其他会话的工单”，避免通过接口枚举他人记录。
            record = require_ticket(principal, ticket_id)
            state = await buyer_bundle(principal.buyer_id)[1].confirm_graph.ainvoke({
                "request": {"ticket_id": ticket_id, **request.model_dump()}, "events": []})
            await sync_ticket_context(principal.buyer_id, request.session_id, state)
            return TicketResponse(status=state["status"], message=state["message"], error_code=state.get("error_code"),
                                  ticket=state.get("ticket"), ticket_events=state.get("ticket_events", []),
                                  events=state["events"])

    @app.get("/tickets/{ticket_id}", response_model=TicketResponse)
    async def get_ticket(ticket_id: str, session_id: str = Query(min_length=1, max_length=128),
                         principal: Principal = Depends(current_principal)):
        """按服务端 buyer_id 校验工单归属；session_id 只用于记录调用来源。"""
        require_conversation(principal, session_id)
        record = require_ticket(principal, ticket_id)
        status = "submitted" if record["status"] == "submitted" else "cancelled" if record["status"] == "cancelled" else "draft_created"
        return TicketResponse(status=status, message="工单状态查询成功。", ticket=ticket_store.public(record),
                              events=[TraceEvent(node="ticket_query", status="success", summary="已读取工单状态。")])

    @app.post("/tickets/{ticket_id}/cancel", response_model=TicketResponse)
    async def cancel_ticket(ticket_id: str, request: TicketCancelRequest, principal: Principal = Depends(writing_principal)):
        """运行取消状态图；取消后允许用户重新选择退货或换货路径。"""
        require_ticket_writable()
        require_conversation(principal, request.session_id)
        async with session_lock(principal.buyer_id, request.session_id):
            record = require_ticket(principal, ticket_id)
            state = await buyer_bundle(principal.buyer_id)[1].cancel_graph.ainvoke({
                "request": {"ticket_id": ticket_id, **request.model_dump()}, "events": []})
            await sync_ticket_context(principal.buyer_id, request.session_id, state)
            return TicketResponse(status=state["status"], message=state["message"], error_code=state.get("error_code"),
                                  ticket=state.get("ticket"), ticket_events=state.get("ticket_events", []),
                                  events=state["events"])

    @app.get("/tickets/{ticket_id}/events", response_model=list[TicketEvent])
    async def ticket_events(ticket_id: str, session_id: str = Query(min_length=1, max_length=128),
                            principal: Principal = Depends(current_principal)):
        """按服务端 buyer_id 返回工单审计历史；session_id 仅记录查询来源。"""
        require_conversation(principal, session_id)
        record = require_ticket(principal, ticket_id)
        return ticket_store.events(record)

    @app.get("/runs/{run_id}", response_model=ChatResponse)
    async def run(run_id: UUID, principal: Principal = Depends(current_principal)):
        """UUID 类型校验后读取执行记录，避免把 URL 参数当作文件路径。"""
        owner = store.run_owner(str(run_id))
        if not owner:
            raise HTTPException(404, "执行记录不存在。")
        if owner["buyer_id"] != principal.buyer_id:
            deny_access("RUN_FORBIDDEN")
        require_conversation(principal, owner["session_id"])
        result = store.read("run_" + str(run_id))
        if not result:
            raise HTTPException(404, "执行记录不存在。")
        return result

    return app


class LazySkillFlowApp:
    """延迟创建默认 ASGI 应用，避免测试导入 create_app 时触碰正在使用的数据库。"""

    def __init__(self):
        self.instance = None

    async def __call__(self, scope, receive, send):
        if self.instance is None:
            self.instance = create_app()
        await self.instance(scope, receive, send)


app = LazySkillFlowApp()
