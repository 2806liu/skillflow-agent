"""CoreCoder：一个受 Claude Code 架构启发的精简 AI 编程代理。"""

__version__ = "0.7.0"

from corecoder.agent import Agent
from corecoder.config import Config
from corecoder.llm import LLM
from corecoder.tools import ALL_TOOLS

__all__ = ["ALL_TOOLS", "LLM", "Agent", "Config", "__version__"]
