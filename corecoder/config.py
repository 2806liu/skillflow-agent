"""配置管理：读取环境变量并提供默认值。"""

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv():
    """从当前目录向上查找并加载 .env，直到用户主目录。"""
    from dotenv import load_dotenv

    # 先检查当前目录，再逐级检查父目录，避免漏掉项目根目录的配置。
    env_path = Path(".env")
    if not env_path.exists():
        cur = Path.cwd()
        home = Path.home()
        while cur != home and cur != cur.parent:
            candidate = cur / ".env"
            if candidate.exists():
                env_path = candidate
                break
            cur = cur.parent
    load_dotenv(env_path, override=False)


@dataclass
class Config:
    """CoreCoder 的模型、API 和上下文长度配置。"""

    model: str = "gpt-5.5"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    max_context_tokens: int = 128_000
    provider: str = "openai"

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置实例，显式环境变量优先于 .env 文件。"""

        # 加载 .env，但不覆盖已经存在的环境变量。
        _load_dotenv()
        # 按优先级读取 CoreCoder、OpenAI、DeepSeek 的常见 API Key 变量。
        api_key = (
            os.getenv("CORECODER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or ""
        )
        return cls(
            model=os.getenv("CORECODER_MODEL", "gpt-5.5"),
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("CORECODER_BASE_URL"),
            max_tokens=int(os.getenv("CORECODER_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("CORECODER_TEMPERATURE", "0")),
            max_context_tokens=int(os.getenv("CORECODER_MAX_CONTEXT", "128000")),
            provider=os.getenv("CORECODER_PROVIDER", "openai"),
        )
