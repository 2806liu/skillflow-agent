from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatRequest(Contract):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)


class StreamChatRequest(ChatRequest):
    """浏览器预生成 run_id；重试同一请求只回放进度，不能重做业务写入。"""
    run_id: UUID


class DialogueDecision(Contract):
    """LLM/规则层输出的对话动作。

    该对象只描述用户想做的事，不直接执行订单或工单写操作；API 层会再次
    校验订单归属、工单状态和确认门，避免模型把“退款”误当成已执行退款。
    """
    intent: Literal[
        "order_query", "return_exchange", "coupon_dispute", "price_adjustment",
        "ticket_action", "unsupported", "unknown"
    ]
    order_id: str | None = Field(default=None, pattern=r"^A[0-9]{3,8}$")
    ticket_id: str | None = None
    reference: Literal["explicit", "current_order", "current_ticket", "none"] = "none"
    needs_clarification: bool = False
    clarification_question: str = Field(default="", max_length=500)
    requested_action: Literal[
        "investigate", "confirm_product", "create_draft", "confirm_submit", "confirm_cancel", "decline", "cancel", "status", "none"
    ] = "none"
    request_type: Literal["return", "exchange", "discount_dispute", "price_adjustment"] | None = None


class AssistantResponse(Contract):
    """模型选择事实段落顺序与服务语气，程序按原文渲染金额、资格和状态。

    禁止自由生成事实句；段落索引必须完整且无重复，引用编号必须匹配真实证据。
    这样回答节点可调整表达顺序，但不能把预计退款改成已退款。
    """
    answer: str = Field(min_length=1, max_length=4000)
    fact_indexes: list[int] = Field(default_factory=list)
    tone: Literal["neutral", "empathic", "acknowledge"] = "neutral"
    citation_numbers: list[int] = Field(default_factory=list)


class PlanStep(Contract):
    step_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,30}$")
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class Plan(Contract):
    decision: Literal["execute", "clarify", "unsupported"]
    # 技能名必须来自项目 B 的本地注册表；price_adjustment 计划仍只执行只读价差分析。
    # 价保草稿由独立工单图在重新核验后创建，调查计划本身不能直接退款或提交。
    skill: Literal["order_tracking", "return_exchange", "coupon_dispute", "price_adjustment"]
    goal: str = Field(min_length=1, max_length=500)
    order_id: str | None = Field(default=None, pattern=r"^A[0-9]{3,8}$")
    message: str = Field(default="", max_length=1000)
    steps: list[PlanStep] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def coherent(self):
        if self.decision == "execute" and (not self.order_id or not self.steps):
            raise ValueError("executable plans need an order_id and steps")
        if self.decision != "execute" and self.steps:
            raise ValueError("clarify/unsupported plans cannot execute tools")
        if self.decision != "execute" and not self.message:
            raise ValueError("clarify/unsupported plans need a user message")
        return self


class ToolResult(Contract):
    status: Literal["success", "empty", "failed"]
    data: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    message: str = ""
    retryable: bool = False


class Evidence(Contract):
    citation_number: int
    document_id: str
    chunk_id: str
    filename: str | None = None
    page: int | None = None
    snippet: str
    score: float


class TraceEvent(Contract):
    node: str
    status: str
    summary: str
    step_id: str | None = None
    tool: str | None = None
    duration_ms: float = 0
    details: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(Contract):
    run_id: str
    session_id: str
    status: str
    answer: str
    model_mode: str
    dialogue: DialogueDecision | None = None
    plan: Plan | None = None
    citations: list[Evidence] = Field(default_factory=list)
    events: list[TraceEvent] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)


class TicketDraftRequest(Contract):
    """创建售后或价保工单草稿的请求；正式提交必须经过单独确认接口。"""
    session_id: str = Field(min_length=1, max_length=128)
    order_id: str = Field(pattern=r"^A[0-9]{3,8}$")
    request_type: Literal["return", "exchange", "discount_dispute", "price_adjustment"]
    reason: str = Field(min_length=2, max_length=200)


class TicketConfirmRequest(Contract):
    """工单确认请求；confirmed=False 只保留草稿，不执行写入。"""
    session_id: str = Field(min_length=1, max_length=128)
    confirmed: bool


class TicketCancelRequest(Contract):
    """取消售后或价保工单的请求；服务端按买家隔离，会话 ID 只记录操作来源。"""
    session_id: str = Field(min_length=1, max_length=128)
    reason: str = Field(default="用户取消售后申请", max_length=200)


class TicketRecord(Contract):
    """对外返回的工单记录；不暴露服务端 buyer_id 和会话内部字段。"""
    ticket_id: str
    order_id: str
    sku: str
    request_type: Literal["return", "exchange", "discount_dispute", "price_adjustment"]
    reason: str
    status: Literal["draft", "submitted", "cancelled"]
    idempotency_key: str
    source_run_id: str
    # 金额争议与价保工单保存可复算数值和引用编号；退换货工单使用空字典。
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class TicketEvent(Contract):
    """工单的持久化操作审计；与单次 LangGraph 节点 Trace 分开保存。"""
    event_id: str
    ticket_id: str
    buyer_id: str
    session_id: str
    action: str
    result: str
    created_at: str
    error_code: str | None = None


class TicketResponse(Contract):
    """工单接口统一响应；包含状态、提示和可回溯的调查事件。"""
    status: Literal["draft_created", "submitted", "already_submitted", "cancelled", "rejected"]
    message: str
    error_code: str | None = None
    ticket: TicketRecord | None = None
    ticket_events: list[TicketEvent] = Field(default_factory=list)
    citations: list[Evidence] = Field(default_factory=list)
    events: list[TraceEvent] = Field(default_factory=list)
