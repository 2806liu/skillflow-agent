"""工具注册表：公开参数与内部参数分离，只返回当前可用的授权工具 Schema。"""

from dataclasses import dataclass

from pydantic import BaseModel, Field

from .contracts import Contract, ToolSpec


class OrderArgs(Contract):
    """订单身份只允许单号；买家、SKU、金额和确认均由服务端提供。"""

    order_id: str = Field(pattern=r"^A[0-9]{3,8}$")


class PolicyArgs(OrderArgs):
    """模型提供检索意图，实际 policy_key 取自可信订单。"""

    query: str = Field(min_length=1, max_length=1000)


class EvidenceArgs(OrderArgs):
    """只接受已登记引用，禁止传原始 Evidence 或任意规则文本。"""

    evidence_refs: list[str] = Field(min_length=1, max_length=16)


class ListOrdersArgs(Contract):
    """有界游标分页；执行器还会限制关键词及游标连续性。"""

    keyword: str = Field(default="", max_length=80)
    after_order_id: str | None = Field(default=None, pattern=r"^A[0-9]{3,8}$")
    page_size: int = Field(default=20, ge=1, le=50)


class LoadSkillArgs(Contract):
    """技能按名称查配置映射，不能接收模型指定的文件路径。"""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    version: str | None = Field(default=None, min_length=1, max_length=40)
    reason: str = Field(min_length=1, max_length=400)


class DraftArgs(OrderArgs):
    """写工具仅声明公开参数，S4 前无可执行处理器。"""

    request_type: str = Field(pattern=r"^(return|exchange|discount_dispute|price_adjustment)$")
    reason: str = Field(min_length=1, max_length=500)


class SubmitArgs(Contract):
    """模型不能传 confirmed；真正授权只能来自后续服务端确认流程。"""

    ticket_id: str = Field(min_length=1, max_length=100)


@dataclass(frozen=True)
class ToolDefinition:
    """内部定义包含前置条件和可用性，不包含任意可执行配置代码。"""

    name: str
    description: str
    arguments: type[BaseModel]
    requires: tuple[str, ...] = ()
    effect: str = "read_only"
    unavailable_code: str | None = None

    def model_spec(self) -> ToolSpec:
        """Pydantic extra=forbid 形成 additionalProperties=false 的公开 Schema。"""
        return ToolSpec(name=self.name, description=self.description, parameters=self.arguments.model_json_schema(),
                        version="s2_tools_v1", effect=self.effect)


class LocalToolRegistry:
    """声明可用能力；S2 始终禁用写入，物流只保留声明。"""

    def __init__(self, incident_available: bool = False, writes_enabled: bool = False, logistics_enabled: bool = False):
        incident_error = None if incident_available else "INCIDENT_BACKEND_UNAVAILABLE"
        definitions = [
            ToolDefinition("load_skill", "按任务需要加载技能规范及允许的工具", LoadSkillArgs),
            ToolDefinition("list_my_orders", "分页发现本人订单候选", ListOrdersArgs),
            ToolDefinition("get_order", "查询本人且在当前任务范围内的目标订单", OrderArgs),
            ToolDefinition("retrieve_policy", "检索该目标订单对应的本地政策", PolicyArgs, ("get_order",)),
            ToolDefinition("assess_after_sales", "依据可信政策引用核验退换资格", EvidenceArgs, ("get_order", "retrieve_policy")),
            ToolDefinition("check_inventory", "读取已核验退换目标的库存", OrderArgs, ("get_order", "assess_after_sales")),
            ToolDefinition("analyze_discount", "核对可信订单金额和优惠政策", EvidenceArgs, ("get_order", "retrieve_policy")),
            ToolDefinition("analyze_price_adjustment", "按可信价格凭证及政策判断价保", EvidenceArgs, ("get_order", "retrieve_policy")),
            ToolDefinition("match_quality_incident", "按可信批次和已审核事件版本确定性判断", OrderArgs, ("get_order",), unavailable_code=incident_error),
            ToolDefinition("list_alternatives", "列出同类在售有库存候选，不作安全保证", OrderArgs, ("get_order", "match_quality_incident"), unavailable_code=incident_error),
            # 能力开关固定于图版本，连接健康变化不改变 Schema，否则旧检查点无法恢复。
            ToolDefinition("get_logistics", "通过 MCP 查询本人订单配送物流，返回来源更新时间" if logistics_enabled else "查询本人订单配送物流", OrderArgs, ("get_order",), unavailable_code=None if logistics_enabled else "MCP_NOT_READY"),
            # S4 显式装配才开放写工具；S2/S3 的默认构造继续保持只读。
            ToolDefinition("create_ticket_draft", "在本轮明确申请且完成调查后创建草稿", DraftArgs, effect="draft_write", unavailable_code=None if writes_enabled else "WRITE_NOT_ENABLED"),
            ToolDefinition("submit_ticket", "提出提交草稿动作；必须暂停等待服务端确认卡片", SubmitArgs, effect="confirmed_write", unavailable_code=None if writes_enabled else "WRITE_NOT_ENABLED"),
        ]
        self._definitions = {definition.name: definition for definition in definitions}

    @property
    def names(self) -> set[str]:
        """供 Manifest 配置校验，调用方获取独立集合。"""
        return set(self._definitions)

    def resolve(self, name: str) -> ToolDefinition | None:
        """不通过 getattr 或模块路径执行模型提供的名称。"""
        return self._definitions.get(name)

    def model_specs(self, names: list[str]) -> list[ToolSpec]:
        """白名单求交，禁用工具不会因加载 Manifest 就出现在模型可用集合。"""
        return [definition.model_spec() for name in sorted(set(names))
                if (definition := self.resolve(name)) is not None and definition.unavailable_code is None]
