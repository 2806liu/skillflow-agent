from dataclasses import dataclass, field
import os
from pathlib import Path
import re

from dotenv import load_dotenv

from skillflow.agent.config import BudgetLimits

ROOT = Path(__file__).resolve().parent.parent
# 只加载本项目的配置，不读取其他项目的 .env。
load_dotenv(ROOT / ".env", override=False)


def default_data_dir() -> Path:
    """环境变量只能选择项目 B 内目录；测试代码显式传入的临时目录仍保持兼容。"""
    target = Path(os.getenv("SKILLFLOW_DATA_DIR", str(ROOT / ".skillflow"))).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError("SKILLFLOW_DATA_DIR must remain inside project B")
    return target


def default_fixtures_dir() -> Path:
    """环境配置仅能选择项目 B 内样本；在读取任何文件前拒绝跨项目路径。"""
    target = Path(os.getenv("SKILLFLOW_FIXTURES_DIR", str(ROOT / "data"))).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError("SKILLFLOW_FIXTURES_DIR must remain inside project B")
    return target


@dataclass(frozen=True)
class Settings:
    """项目 B 独立配置：业务数据与政策源均位于本项目目录。"""
    # 可选隔离演示库必须位于项目 B 内；默认仍使用原有 .skillflow，不迁走现有工单。
    data_dir: Path = field(default_factory=default_data_dir)
    # 新引擎必须显式选择；脚本模式与旧 Planner 的 mock/真实模型配置独立。
    agent_engine: str = field(default_factory=lambda: os.getenv("SKILLFLOW_AGENT_ENGINE", "legacy"))
    agent_model_mode: str = field(default_factory=lambda: os.getenv("SKILLFLOW_AGENT_MODEL_MODE", "disabled"))
    agent_limits: BudgetLimits = field(default_factory=BudgetLimits.from_env)
    # S3 以可复核字符上限限制完整上下文，包含工具 Schema；不冒称精确 token 计量。
    agent_max_context_chars: int = field(default_factory=lambda: int(os.getenv("SKILLFLOW_AGENT_MAX_CONTEXT_CHARS", "100000")))
    # 短期调查目标有效期；刷新和 SSE 回放不续期，新的有效目标确认才续期。
    agent_context_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("SKILLFLOW_AGENT_CONTEXT_TTL_SECONDS", "1800")))
    # S4 显式启用实际检查点与工单工具；S3 演示和 legacy 不隐式开放写入。
    agent_s4_enabled: bool = field(default_factory=lambda: os.getenv("SKILLFLOW_AGENT_S4_ENABLED", "false").lower() in {"1","true","yes"})
    agent_confirmation_seconds: int = field(default_factory=lambda: int(os.getenv("SKILLFLOW_AGENT_CONFIRMATION_SECONDS", "180")))
    # S5 独立装配物流协议；旧 S4 工具 Schema 和检查点保持原版本。
    agent_s5_enabled: bool = field(default_factory=lambda: os.getenv("SKILLFLOW_AGENT_S5_ENABLED", "false").lower() in {"1", "true", "yes"})
    # S6 显式允许新图接入原登录与业务库；默认保持旧环境行为。
    agent_s6_enabled: bool = field(default_factory=lambda: os.getenv("SKILLFLOW_AGENT_S6_ENABLED", "false").lower() in {"1", "true", "yes"})
    # 物流进程只读项目 B 内的指定记录，支持追加属于真实登录测试买家的模拟数据。
    agent_logistics_file: Path = field(default_factory=lambda: Path(os.getenv("SKILLFLOW_AGENT_LOGISTICS_FILE", str(ROOT / "data/logistics/delivery.json"))))
    agent_mcp_query_timeout: float = field(default_factory=lambda: float(os.getenv("SKILLFLOW_AGENT_MCP_QUERY_TIMEOUT", "10")))
    agent_mcp_startup_timeout: float = field(default_factory=lambda: float(os.getenv("SKILLFLOW_AGENT_MCP_STARTUP_TIMEOUT", "10")))
    fixtures_dir: Path = field(default_factory=default_fixtures_dir)
    # 业务仓库默认沿用旧 JSON；本机高并发演示显式切到 SQLAlchemy，避免重启时迁走已有工单。
    business_backend: str = field(default_factory=lambda: os.getenv("SKILLFLOW_BUSINESS_BACKEND", "mock"))
    # 同时执行与等待的请求都有上限；队列满时在受理前返回明确的繁忙码。
    # 本机 SQLite 只有一个并发写入者；短测显示 8 个执行位比 16 个有更低的锁等待。
    max_active_chats: int = field(default_factory=lambda: int(os.getenv("SKILLFLOW_MAX_ACTIVE_CHATS", "8")))
    max_waiting_chats: int = field(default_factory=lambda: int(os.getenv("SKILLFLOW_MAX_WAITING_CHATS", "64")))
    # 仅为相同业务路径的本机对照测试保留 global 模式；正常运行使用 session。
    chat_lock_mode: str = field(default_factory=lambda: os.getenv("SKILLFLOW_CHAT_LOCK_MODE", "session"))
    model_mode: str = field(default_factory=lambda: os.getenv("SKILLFLOW_MODEL_MODE", "mock"))
    model: str = field(default_factory=lambda: os.getenv("SKILLFLOW_MODEL", ""))
    # 模型配置只使用 B 专用变量，避免读取其他项目或通用客户端的运行配置。
    api_key: str = field(default_factory=lambda: os.getenv("SKILLFLOW_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("SKILLFLOW_BASE_URL", ""))
    timeout: float = field(default_factory=lambda: float(os.getenv("SKILLFLOW_TIMEOUT", "20")))
    model_timeout: float = field(default_factory=lambda: float(os.getenv("SKILLFLOW_MODEL_TIMEOUT", "30")))
    model_retries: int = field(default_factory=lambda: int(os.getenv("SKILLFLOW_MODEL_RETRIES", "1")))
    # 模型请求默认忽略进程继承的通用代理变量；需要系统代理时显式设为 true。
    # 这样项目 B 的出站路径由自己的配置决定，不被其他服务的代理设置影响。
    model_trust_env: bool = field(default_factory=lambda: os.getenv("SKILLFLOW_MODEL_TRUST_ENV", "false").lower() in {"1", "true", "yes", "on"})
    # 默认在首次/切换商品时追问确认；兼容测试可以显式关闭该交互门。
    confirm_product: bool = True
    # 演示身份从启动环境读取；请求体不能覆盖它，避免把客户端字段当成权限凭证。
    demo_buyer_id: str = field(default_factory=lambda: os.getenv("SKILLFLOW_DEMO_BUYER_ID", "demo-buyer"))
    # B6 将演示身份与生产登录明确分开；生产模式必须配置项目 B 自己的 OIDC 提供方。
    auth_mode: str = field(default_factory=lambda: os.getenv("SKILLFLOW_AUTH_MODE", "demo"))
    oidc_issuer: str = field(default_factory=lambda: os.getenv("SKILLFLOW_OIDC_ISSUER", ""))
    oidc_client_id: str = field(default_factory=lambda: os.getenv("SKILLFLOW_OIDC_CLIENT_ID", ""))
    oidc_client_secret: str = field(default_factory=lambda: os.getenv("SKILLFLOW_OIDC_CLIENT_SECRET", ""))
    oidc_redirect_uri: str = field(default_factory=lambda: os.getenv("SKILLFLOW_OIDC_REDIRECT_URI", ""))
    # 容器桥接网络可只改后端连接主机；对浏览器公布的 issuer 和令牌签发者保持原值。
    oidc_connect_host: str = field(default_factory=lambda: os.getenv("SKILLFLOW_OIDC_CONNECT_HOST", ""))
    # 身份提供方的出站代理独立于模型网络配置，只有显式开启才读取进程代理环境。
    oidc_trust_env: bool = field(default_factory=lambda: os.getenv("SKILLFLOW_OIDC_TRUST_ENV", "false").lower() in {"1", "true", "yes", "on"})
    auth_session_seconds: int = field(default_factory=lambda: int(os.getenv("SKILLFLOW_AUTH_SESSION_SECONDS", "3600")))

    def __post_init__(self):
        if self.agent_engine not in {"legacy", "tool_loop"}:
            raise ValueError("SKILLFLOW_AGENT_ENGINE must be legacy or tool_loop")
        if self.agent_model_mode not in {"disabled", "scripted", "langchain_openai"}:
            raise ValueError("SKILLFLOW_AGENT_MODEL_MODE must be disabled, scripted or langchain_openai")
        # 演示继续隔离；仅用户确认的 S6 就地升级允许原业务目录。
        if self.agent_engine == "tool_loop":
            target = Path(self.data_dir).resolve()
            if not target.is_relative_to(ROOT.resolve()) or (target == (ROOT / ".skillflow").resolve() and not self.agent_s6_enabled):
                raise ValueError("tool_loop requires an independent data directory inside project B")
        if self.agent_s6_enabled and not (self.agent_s5_enabled and self.auth_mode == "production"):
            raise ValueError("S6 requires S5 and production authentication")
        if not Path(self.agent_logistics_file).resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("logistics data must remain inside project B")
        if not isinstance(self.agent_limits, BudgetLimits):
            raise ValueError("agent_limits must be BudgetLimits")
        if self.agent_max_context_chars < 1000:
            raise ValueError("agent_max_context_chars must be at least 1000")
        if self.agent_context_ttl_seconds <= 0:
            raise ValueError("agent_context_ttl_seconds must be positive")
        if self.agent_confirmation_seconds <= 0:
            raise ValueError("agent_confirmation_seconds must be positive")
        if self.agent_s5_enabled and not self.agent_s4_enabled:
            raise ValueError("S5 requires the persistent S4 runtime")
        if self.agent_mcp_query_timeout <= 0 or self.agent_mcp_startup_timeout <= 0:
            raise ValueError("MCP timeouts must be positive")
        if self.agent_s4_enabled and (self.agent_engine != "tool_loop" or self.agent_model_mode != "langchain_openai" or self.business_backend != "sqlalchemy"):
            raise ValueError("S4 requires tool_loop, langchain_openai and project B SQLAlchemy backend")
        if self.business_backend not in {"mock", "sqlalchemy"}:
            raise ValueError("SKILLFLOW_BUSINESS_BACKEND must be mock or sqlalchemy")
        if self.max_active_chats < 1 or self.max_waiting_chats < 0:
            raise ValueError("聊天执行容量须为正数，等待容量不能为负数")
        if self.chat_lock_mode not in {"session", "global"}:
            raise ValueError("SKILLFLOW_CHAT_LOCK_MODE must be session or global")
        # 明确区分规则模拟与真实模型；配置错误不静默降级。
        if self.model_mode not in {"mock", "langchain_openai"}:
            raise ValueError("SKILLFLOW_MODEL_MODE must be mock or langchain_openai")
        if self.timeout <= 0 or self.model_timeout <= 0 or self.model_retries < 0:
            raise ValueError("模型与工具超时必须大于零，重试次数不能为负数")
        if self.auth_mode not in {"demo", "production"} or self.auth_session_seconds <= 0:
            raise ValueError("身份模式必须为 demo/production，认证会话期限必须大于零")
        if self.auth_mode == "production":
            # 缺少可信提供方或 HTTPS 回调时立即停机，不允许退回固定演示买家。
            if not all((self.oidc_issuer.startswith("https://"), self.oidc_client_id,
                        self.oidc_client_secret, self.oidc_redirect_uri.startswith("https://"))):
                raise ValueError("生产认证需要 HTTPS OIDC issuer、client_id、client_secret 和 redirect_uri")
        if self.oidc_connect_host and not re.fullmatch(r"[A-Za-z0-9.-]+(?::[0-9]{1,5})?", self.oidc_connect_host):
            # 只接受主机和可选端口，拒绝路径、用户信息和查询串进入后端连接目标。
            raise ValueError("SKILLFLOW_OIDC_CONNECT_HOST 只能是主机名和可选端口")
