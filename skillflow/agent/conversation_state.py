"""短期任务上下文契约：只记调查目的与来源，不保存订单事实、证据或工具权限。"""

from typing import Literal
import re

from pydantic import ConfigDict, Field, model_validator

from .contracts import Contract

Requirement = Literal["order", "returns", "discount", "price", "incident", "alternatives", "logistics"]
CONVERSATION_VERSION = "short_goal_v1"
GOAL_LABELS = {"order": "查询本人订单", "returns": "结合店铺政策核验退换资格及库存",
               "discount": "核对优惠金额", "price": "核验价保资格",
               "incident": "依据已审核质量事件核对批次影响和退货资格",
               "alternatives": "查询在售有库存替代候选", "logistics": "查询配送物流"}


class GoalTemplate(Contract):
    """可继承的最小目标；没有旧单号、关键词、批量标志、金额或事实结论。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    requirements: tuple[Requirement, ...] = Field(min_length=1, max_length=7)

    @model_validator(mode="after")
    def valid_requirements(self):
        """模板必须包含订单核验，且不能通过重复字段扩大内容。"""
        if "order" not in self.requirements or len(set(self.requirements)) != len(self.requirements):
            raise ValueError("CONTEXT_GOAL_INVALID")
        return self

    @property
    def summary(self):
        """摘要由受控目标枚举渲染，不沿用旧用户原文中的订单号或指令。"""
        # 展示顺序方便阅读，不是预编译的工具执行顺序。
        return "；".join(label for name, label in GOAL_LABELS.items() if name in self.requirements)


class ConversationSnapshot(Contract):
    """B SQLite 中的版本化会话记录；暂停状态不提供可继承模板。"""

    version: Literal["short_goal_v1"] = CONVERSATION_VERSION
    buyer_id: str
    session_id: str
    engine_version: str
    graph_version: str
    schema_version: str
    status: Literal["active", "paused"]
    template: GoalTemplate | None = None
    source_task_id: str
    source_run_id: str
    confirmed_at: float = Field(ge=0)
    expires_at: float = Field(ge=0)
    candidates: list[str] = Field(default_factory=list, max_length=50)
    pause_reason: str | None = None

    @model_validator(mode="after")
    def coherent(self):
        """格式有效不等于归属和版本有效；适配器加载时还须验证当前元数据。"""
        if self.expires_at <= self.confirmed_at or ((self.status == "active") != (self.template is not None)):
            raise ValueError("CONTEXT_STATE_INVALID")
        if self.status == "paused" and self.candidates:
            raise ValueError("CONTEXT_STATE_INVALID")
        if len(set(self.candidates)) != len(self.candidates) or any(not re.fullmatch(r"A[0-9]{3,8}", c) for c in self.candidates):
            raise ValueError("CONTEXT_CANDIDATES_INVALID")
        return self


class ConversationView(Contract):
    """加载结果明确区分缺失、过期、暂停和不兼容，策略不能默默降级查询目的。"""

    status: Literal["missing", "active", "expired", "paused", "incompatible"] = "missing"
    snapshot: ConversationSnapshot | None = None
