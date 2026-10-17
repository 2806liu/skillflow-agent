"""图状态与端口数据契约；只保存可校验 JSON，不保存客户端、Cookie 或密钥。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .config import BudgetLimits, SCHEMA_VERSION


class Contract(BaseModel):
    """拒绝未声明字段，避免把任意对象或隐式业务参数混入状态。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class BudgetUsage(Contract):
    """可序列化的累计消耗；不包含进程相关的 monotonic 起点。"""

    rounds: int = Field(default=0, ge=0)
    tool_attempts: int = Field(default=0, ge=0)
    model_requests: int = Field(default=0, ge=0)
    active_seconds: float = Field(default=0, ge=0)


class TaskMetadata(BaseModel):
    """服务端登记的不可变任务归属和引擎版本；不能作为客户端授权凭据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    buyer_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    engine: Literal["legacy", "tool_loop"]
    engine_version: str
    graph_version: str
    schema_version: Literal["agent_state_v1"] = SCHEMA_VERSION
    model_mode: Literal["mock", "langchain_openai", "scripted"]
    limits: BudgetLimits = Field(default_factory=BudgetLimits)


class ToolCall(Contract):
    """保留原生调用 ID、名称和公开参数；S2 执行边界与 S3 原生消息适配共用。"""

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class MessageEnvelope(Contract):
    """消息仅允许固定角色，工具观察必须绑定调用 ID。"""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def coherent(self):
        """禁止孤立观察或把工具调用伪装成用户消息。"""
        if (self.role == "tool") != bool(self.tool_call_id):
            raise ValueError("工具观察必须且只能携带 tool_call_id")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("仅 assistant 消息可包含工具调用")
        return self


class ToolSpec(Contract):
    """模型公开 Schema 与版本；身份和原始证据不进入参数 Schema。"""

    name: str
    description: str
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    version: str
    # 元数据由注册表声明；模型绑定时只转换公开名称、说明和参数 Schema。
    effect: Literal["read_only", "draft_write", "confirmed_write"] = "read_only"


class ToolObservation(Contract):
    """工具结果通过调用 ID 配对，业务事实使用受控引用。"""

    invocation_id: str
    tool_call_id: str
    status: Literal["success", "empty", "rejected", "failed"]
    result_ref: str | None = None
    error_code: str | None = None
    data: dict[str, JsonValue] = Field(default_factory=dict)
    message: str = ""
    retryable: bool = False


class SkillDocument(Contract):
    """S2 按需加载的操作规范及范围；哈希同时覆盖 Manifest 和 SOP。"""

    name: str
    version: str
    instructions: str
    tool_names: list[str]
    content_hash: str = ""
    completion_conditions: list[str] = Field(default_factory=list)


class ModelReply(Contract):
    """模型适配器统一输出；S3 真实消息转换保留调用 ID 与可用用量。"""

    message: MessageEnvelope
    usage: dict[str, int] | None = None


class RuntimeEvent(Contract):
    """事件携带任务归属与版本，只记录运行元数据和稳定错误码。"""

    metadata: TaskMetadata
    node: str
    status: str
    budget: BudgetUsage
    error_code: str | None = None
    duration_ms: float = Field(default=0, ge=0)
    # S2 记录 Skill 版本、开放工具和调用状态，不记录令牌或原始身份凭据。
    details: dict[str, JsonValue] = Field(default_factory=dict)


class AgentState(Contract):
    """实际执行图状态；版本不兼容时拒绝加载，恢复语义在 S4 接入。"""

    schema_version: Literal["agent_state_v1"] = SCHEMA_VERSION
    metadata: TaskMetadata
    messages: list[MessageEnvelope] = Field(default_factory=list)
    goal: str
    constraints: dict[str, JsonValue] = Field(default_factory=dict)
    loaded_skills: dict[str, str] = Field(default_factory=dict)
    observations: list[ToolObservation] = Field(default_factory=list)
    budget: BudgetUsage = Field(default_factory=BudgetUsage)
    # S3 业务完成、澄清和证据不足分别终止，不能统一冒充成功。
    status: Literal["created", "running", "runtime_ready", "completed", "clarify", "needs_manual_review", "unsupported", "failed", "waiting_confirmation", "interrupted"] = "created"
    stop_reason: str | None = None
    answer: str = ""
    events: list[RuntimeEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def versions_match(self):
        """元数据与状态必须采用同一 Schema 版本。"""
        if self.schema_version != self.metadata.schema_version:
            raise ValueError("状态与任务版本不一致")
        return self
