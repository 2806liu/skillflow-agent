"""S4 HTTP 适配：认证、CSRF、准入、版本与幂等请求；领域规则仍在命令服务。"""

import asyncio
import re
from typing import Literal
from uuid import UUID

from fastapi import Depends, HTTPException, Query
from pydantic import Field

from .contracts import Contract
from .durable_store import TaskConflict
from .adapters.events import to_trace
from .persistent_service import persistent_projection
from skillflow.contracts import ChatResponse


class ResumeRequest(Contract):
    """客户端引用已展示的动作和版本；不能传任意图状态、buyer 或 thread_id。"""
    session_id: str=Field(min_length=1,max_length=128)
    request_id: str=Field(min_length=1,max_length=128)
    expected_version: int=Field(ge=1)
    checkpoint_id: str=Field(min_length=1,max_length=128)
    decision: Literal['confirm','refresh','continue']
    confirmation_id: str|None=None
    action_hash: str|None=None
    # 明确短句通过同一个确认入口，不让模型解释“确认”或给出授权布尔值。
    source_text: str|None=Field(default=None,max_length=40)


def install_agent_routes(app,settings,writing_principal,current_principal,require_conversation,session_lock,admission,wake_stream):
    """与旧 tickets 接口分开装配；不解除 S3/新图对旧工单写入口的屏障。"""
    def service():
        if not settings.agent_s4_enabled:
            raise HTTPException(404,'S4 未启用。')
        # S6 保留所有身份、CSRF、任务归属和图版本检查，仅解除演示环境屏障。
        if settings.auth_mode!='demo' and not settings.agent_s6_enabled:
            raise HTTPException(503,detail={'code':'ENGINE_DEMO_ONLY','message':'S4 当前在独立演示配置验收。'})
        return app.state.agent_service

    def reject(exc):
        status=403 if exc.code.endswith('FORBIDDEN') else 409
        raise HTTPException(status,detail={'code':exc.code,'message':'任务不能继续：'+exc.code})

    @app.get('/agent/tasks')
    async def tasks(session_id:str=Query(min_length=1,max_length=128),principal=Depends(current_principal)):
        """刷新读取同会话任务，不自动执行或续卡。"""
        require_conversation(principal,session_id)
        try:
            return {'tasks':await service().list_tasks(principal.buyer_id,session_id)}
        except TaskConflict as exc:
            reject(exc)

    @app.get('/agent/tasks/{task_id}')
    async def task(task_id:UUID,session_id:str=Query(min_length=1,max_length=128),principal=Depends(current_principal)):
        """即使知道 UUID 也必须验证买家和会话。"""
        require_conversation(principal,session_id)
        try:
            return await service().detail(str(task_id),principal.buyer_id,session_id)
        except TaskConflict as exc:
            reject(exc)

    @app.post('/agent/tasks/{task_id}/resume')
    async def resume(task_id:UUID,request:ResumeRequest,principal=Depends(writing_principal)):
        """按钮或明确确认短句统一进入此用例；恢复同 task、新 run、累计预算。"""
        svc=service()
        require_conversation(principal,request.session_id)
        if request.source_text and (request.decision!='confirm' or not re.fullmatch(r'\s*(?:确认|确认提交|我确认提交|提交申请)\s*[。！!]?\s*',request.source_text)):
            raise HTTPException(422,detail={'code':'CONFIRMATION_REQUIRED','message':'请核对具体申请并明确确认。'})
        if not await admission.reserve(principal.buyer_id):
            raise HTTPException(429,detail={'code':'CHAT_BUSY','message':'当前请求较多，请稍后重试。'},headers={'Retry-After':'2'})
        async with admission.run(principal.buyer_id):
            async with session_lock(principal.buyer_id,request.session_id):
                try:
                    state=await svc.resume(str(task_id),principal.buyer_id,request)
                except TaskConflict as exc:
                    reject(exc)
                if isinstance(state,dict):
                    return state
                citations,projection=persistent_projection(state)
                projection['task']=await svc.detail(state.metadata.task_id,principal.buyer_id,request.session_id)
                result=ChatResponse(run_id=state.metadata.run_id,session_id=request.session_id,status=state.status,
                    answer=state.answer,model_mode='langchain_openai',citations=citations,
                    events=[to_trace(e) for e in state.events],result=projection)
                store=app.state.store
                await asyncio.to_thread(store.write,'run_'+result.run_id,result.model_dump())
                # 新 run 的进度与 final 单独落帧，旧 run 的帧不覆盖。GET 重连仍只有读取行为。
                for event in result.events:
                    await asyncio.to_thread(store.append_stream_frame,result.run_id,'trace',event.model_dump())
                await asyncio.to_thread(store.append_stream_frame,result.run_id,'final',result.model_dump())
                wake_stream(result.run_id)
                return result
