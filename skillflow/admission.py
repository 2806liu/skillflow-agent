"""单进程有界聊天准入。

受理前计算执行位和等待位；队列满即拒绝。已受理任务无论成功或失败都释放额度。
跨进程扩容仍需共享调度器，本模块只负责项目 B 本机单进程演示。
"""

import asyncio
import time
from contextlib import asynccontextmanager


class ChatAdmission:
    def __init__(self, active: int, waiting: int, per_buyer: int = 4):
        self.active_limit = active
        self.waiting_limit = waiting
        self.per_buyer = per_buyer
        self.semaphore = asyncio.Semaphore(active)
        self.lock = asyncio.Lock()
        self.outstanding = 0
        self.by_buyer: dict[str, int] = {}
        self.running = 0
        self.rejected = 0
        self.peak_waiting = 0
        self.peak_outstanding = 0

    async def reserve(self, buyer_id: str) -> bool:
        """返回 True 才表示可登记 run_id；拒绝时不留待处理业务任务。"""
        async with self.lock:
            if (self.outstanding >= self.active_limit + self.waiting_limit or
                    self.by_buyer.get(buyer_id, 0) >= self.per_buyer):
                self.rejected += 1
                return False
            self.outstanding += 1
            self.by_buyer[buyer_id] = self.by_buyer.get(buyer_id, 0) + 1
            self.peak_outstanding = max(self.peak_outstanding, self.outstanding)
            self.peak_waiting = max(self.peak_waiting, max(0, self.outstanding - self.active_limit))
            return True

    @asynccontextmanager
    async def run(self, buyer_id: str):
        """并发执行位仅在实际执行时占用；异常和取消都归还容量。"""
        queued_at = time.perf_counter()
        try:
            async with self.semaphore:
                self.running += 1
                try:
                    # S6 区分准入排队与模型耗时；原调用方不接收该值也兼容。
                    yield (time.perf_counter() - queued_at) * 1000
                finally:
                    self.running -= 1
        finally:
            await self.release(buyer_id)

    async def release(self, buyer_id: str) -> None:
        async with self.lock:
            self.outstanding -= 1
            remaining = self.by_buyer[buyer_id] - 1
            if remaining:
                self.by_buyer[buyer_id] = remaining
            else:
                self.by_buyer.pop(buyer_id, None)

    def snapshot(self) -> dict:
        """只暴露容量计数，不输出买家 ID、订单或请求内容。"""
        return {"running": self.running, "waiting": max(0, self.outstanding - self.running),
                "rejected": self.rejected, "active_limit": self.active_limit,
                "waiting_limit": self.waiting_limit, "peak_waiting": self.peak_waiting,
                "peak_outstanding": self.peak_outstanding}
