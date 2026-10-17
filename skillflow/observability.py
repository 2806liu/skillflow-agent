"""项目 B 的请求级监控：记录有限的状态与耗时，不采集身份或业务正文。"""

from collections import Counter, deque
import json
import logging
from threading import Lock


class RequestMetrics:
    """保存当前进程的请求统计；输入仅含方法、路由模板、状态码和耗时。"""

    def __init__(self, sample_limit: int = 2048):
        # 有界样本避免长时间运行时内存无限增长；重启后统计从零开始。
        self._durations = deque(maxlen=sample_limit)
        self._statuses = Counter()
        self._lock = Lock()
        self._logger = logging.getLogger("uvicorn.error")

    def record(self, method: str, route: str, status: int, duration_ms: float, request_id: str):
        """完成一次响应后更新进程内统计，并写一行无敏感字段的 JSON 日志。"""
        status_group = f"{status // 100}xx"
        with self._lock:
            self._durations.append(duration_ms)
            self._statuses[status_group] += 1
        # 路由由 FastAPI 的模板给出；不写入真实 URL、查询参数、Cookie、正文或身份。
        self._logger.info(json.dumps({
            "event": "http_request", "request_id": request_id, "method": method,
            "route": route, "status": status, "duration_ms": round(duration_ms, 2),
        }, ensure_ascii=False, separators=(",", ":")))

    def snapshot(self) -> dict:
        """返回当前进程的总量、状态分组和最近样本 P95，不含请求级业务信息。"""
        with self._lock:
            durations = sorted(self._durations)
            statuses = dict(self._statuses)
        p95 = durations[max(0, (95 * len(durations) + 99) // 100 - 1)] if durations else None
        return {"requests": sum(statuses.values()), "status_counts": statuses,
                "sample_count": len(durations), "p95_ms": round(p95, 2) if p95 is not None else None}
