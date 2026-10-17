"""累计预算跟踪；单调时钟仅在当前执行段使用，跨段只携带累计秒数。"""

from time import monotonic

from .config import BudgetLimits
from .contracts import BudgetUsage


class BudgetExceeded(Exception):
    """稳定终止码，不能通过创建新执行段清零。"""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class BudgetTracker:
    """每次运行独享；调用前计数，失败不退还，排队和用户等待不启动时钟。"""

    def __init__(self, limits: BudgetLimits, usage: BudgetUsage | None = None, clock=monotonic):
        self.limits = limits
        self.usage = (usage or BudgetUsage()).model_copy(deep=True)
        self.clock = clock
        self._started = None

    def start(self):
        """开始或继续有效执行段，保持此前累计消耗。"""
        if self._started is None:
            self._started = self.clock()

    @property
    def is_running(self) -> bool:
        """便于工具组件共享图级预算，避免结束工具调用时误停整个图的计时。"""
        return self._started is not None

    def snapshot(self) -> BudgetUsage:
        """多次读取不会重复累计，快照可以直接序列化。"""
        elapsed = max(0, self.clock() - self._started) if self._started is not None else 0
        return self.usage.model_copy(update={"active_seconds": self.usage.active_seconds + elapsed})

    def pause(self) -> BudgetUsage:
        """结束执行段，用户等待期间暂停计时。"""
        self.usage = self.snapshot()
        self._started = None
        return self.snapshot()

    def remaining_seconds(self) -> float:
        """有效执行时间用尽时禁止下一次调用。"""
        remaining = self.limits.active_timeout_seconds - self.snapshot().active_seconds
        if remaining <= 0:
            raise BudgetExceeded("TASK_TIMEOUT")
        return remaining

    def reserve_model(self, *, decision: bool = True):
        """工具决策占一轮及一个请求；其他模型阶段仅占请求，任一不足均不发请求。"""
        self.remaining_seconds()
        # 目标识别与最终提案只占请求；旧 S1 调用保留每请求一轮的默认语义。
        if decision and self.usage.rounds >= self.limits.max_rounds:
            raise BudgetExceeded("ROUND_LIMIT")
        if self.usage.model_requests >= self.limits.max_model_requests:
            raise BudgetExceeded("MODEL_LIMIT")
        self.usage.rounds += int(decision)
        self.usage.model_requests += 1

    def reserve_tool(self):
        """预留 S2/S3 的工具尝试计数，load_skill 和重试也应走该入口。"""
        self.remaining_seconds()
        if self.usage.tool_attempts >= self.limits.max_tool_attempts:
            raise BudgetExceeded("TOOL_LIMIT")
        self.usage.tool_attempts += 1
