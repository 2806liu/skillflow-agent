"""任务目标契约：解释用户需求，固定调查范围，不生成执行步骤或业务结论。"""

import re
from typing import Literal

from pydantic import Field

from .contracts import Contract
from .tool_executor import TargetScope


class GoalProposal(Contract):
    """模型只解释需求；单号、批量权限及最低完成条件由服务端复核。"""

    requirements: list[Literal["order", "returns", "discount", "price", "incident", "alternatives", "logistics"]] = Field(default_factory=list)
    keyword: str = Field(default="", max_length=80)
    supported: bool = True
    # 含糊的新需求不能被自动解释成最窄的 order 目标。
    needs_clarification: bool = False
    # S4 写意图需要模型解释与本轮原文双重约束；默认仅调查。
    requested_action: Literal["investigate", "apply"] = "investigate"
    request_type: Literal["return", "exchange", "discount_dispute", "price_adjustment"] | None = None


class TaskGoal(Contract):
    """执行前冻结的业务目标；后续 Skill 选择不能降低完成标准。"""

    requirements: list[str]
    scope: TargetScope
    original_request: str
    continuation_of_task_id: str | None = None
    # 只有本轮明确申请才绑定写意图；短期 GoalTemplate 不保存该字段。
    application: dict[str, str] | None = None


def application_intent(message: str):
    """从本轮明确表达约束申请类型；询问资格、否定和带条件的表达不授权建单。"""
    ids=tuple(dict.fromkeys(re.findall(r"(?<![A-Za-z0-9])A[0-9]{3,8}(?![A-Za-z0-9])",message.upper())))
    if not re.search(r"申请|办理|帮我退货|帮我换货",message) or re.search(
            r"能否|能不能|是否|可以.*[吗么]|怎么|如何|不(?:要|想|用)|暂不|先别|如果|假如|能申请吗|解释|示例|例子|假设|告诉我",message):
        return None
    # “把退货改成换货”取变更后的类型；对应旧卡片由应用层先失效，不自动取消草稿。
    kind_text=re.split(r'改成|改为',message)[-1]
    kinds=[kind for terms,kind in ((('退货',),'return'),(('换货',),'exchange'),
        (('优惠争议','优惠纠纷'),'discount_dispute'),(('价保','补差'),'price_adjustment'))
        if any(term in kind_text for term in terms)]
    if len(ids)!=1 or len(kinds)!=1:
        raise ValueError('APPLICATION_TARGET_AMBIGUOUS')
    return {"order_id":ids[0],"request_type":kinds[0]}


class AnswerProposal(Contract):
    """模型选择可信事实的展示顺序与语气，不能自由编造事实句。"""

    fact_indexes: list[int]
    result_refs: list[str]
    tone: Literal["neutral", "empathic"] = "neutral"


def checked_goal(message: str, proposal: GoalProposal) -> TaskGoal:
    """仅构造新目标；继承统一经 conversation_policy 校验，避免绕过有效期与版本检查。"""
    ids = tuple(dict.fromkeys(re.findall(r"(?<![A-Za-z0-9])A[0-9]{3,8}(?![A-Za-z0-9])", message.upper())))
    continuation = None
    source = message
    requested = set(proposal.requirements)
    incident = any(term in source for term in ("质量事件", "批次", "受影响", "召回", "新闻", "充电宝"))
    # 此处只设最低验收条件，不决定工具顺序；工具步骤仍由模型实际选择。
    if incident:
        requested.add("incident")
        # 事件内的退货资格已由确定性事件工具覆盖；仅明确另问店铺政策时追加普通退换调查。
        if "店铺政策" not in source:
            requested.discard("returns")
    if any(term in source for term in ("退", "换货", "售后资格")) and not incident:
        requested.add("returns")
    if any(term in source for term in ("替代", "替换商品")):
        requested.update(("incident", "alternatives"))
    if any(term in source for term in ("价保", "价差", "降价")):
        requested.add("price")
    if any(term in source for term in ("优惠", "优惠券")):
        requested.add("discount")
    if any(term in source for term in ("物流", "快递")):
        requested.add("logistics")
    requested.add("order")
    # 明确只查状态时强制切换，防止模型把旧质量事件或退换需求混进来。
    if re.search(r"订单(?:状态|信息)|(?:仅|只)(?:查询|查|看).*订单", source) and not any(
            term in source for term in ("退", "换货", "优惠", "价保", "质量事件", "批次", "替代", "物流", "受影响")):
        requested = {"order"}
    keyword = proposal.keyword.strip() if not ids else ""
    if keyword and keyword not in source:
        raise ValueError("TARGET_QUERY_UNGROUNDED")
    batch = incident and any(term in source for term in ("所有", "全部", "每笔", "批量"))
    application=None
    if proposal.requested_action=='apply':
        application=application_intent(message)
        if application is None or proposal.request_type!=application['request_type']:
            raise ValueError('APPLICATION_INTENT_UNGROUNDED')
    return TaskGoal(requirements=sorted(requested), scope=TargetScope(
        explicit_order_ids=ids, keyword=keyword, quality_batch=batch), original_request=source,
        continuation_of_task_id=continuation, application=application)
