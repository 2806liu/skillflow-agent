"""项目 B 的 SSE 进度出口：只发布执行元数据，最终回答由 API 校验后单独发布。"""

from contextlib import contextmanager
from contextvars import ContextVar
import json


_progress_sink = ContextVar("skillflow_progress_sink", default=None)


@contextmanager
def progress_sink(callback):
    """把当前异步任务的进度写入函数绑定到上下文；退出时隔离下一次请求。"""
    token = _progress_sink.set(callback)
    try:
        yield
    finally:
        _progress_sink.reset(token)


def publish(kind: str, payload: dict):
    """由图节点发布计划、工具开始或 Trace；无流订阅的原有接口直接跳过。"""
    callback = _progress_sink.get()
    if callback is not None:
        callback(kind, payload)


def publish_trace(event):
    """TraceEvent 在生成处发布，避免整张图结束后才批量回放。"""
    publish("trace", event.model_dump())


def sse_frame(frame: dict) -> str:
    """将 SQLite 帧编码为标准 SSE；id 是持久序号，浏览器可据此去重。"""
    return (f"id: {frame['seq']}\n"
            f"event: {frame['kind']}\n"
            f"data: {json.dumps(frame['payload'], ensure_ascii=False)}\n\n")
