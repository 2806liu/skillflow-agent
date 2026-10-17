"""新引擎版本与运行保护参数；保护预算不代表性能验收目标。"""

import os

from pydantic import BaseModel, ConfigDict, Field


ENGINE_VERSION = "tool_loop_s1_v1"
GRAPH_VERSION = "skeleton_v1"
SCHEMA_VERSION = "agent_state_v1"


class BudgetLimits(BaseModel):
    """任务创建时冻结上限，恢复沿用原配置；数值必须为有限正数。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    max_rounds: int = Field(default=8, ge=1)
    max_tool_attempts: int = Field(default=12, ge=1)
    max_model_requests: int = Field(default=10, ge=1)
    model_timeout_seconds: float = Field(default=30, gt=0)
    active_timeout_seconds: float = Field(default=120, gt=0)

    @classmethod
    def from_env(cls):
        """只读取项目 B 专用变量，不改变旧 Planner 的超时和重试配置。"""
        return cls(**{
            name: os.environ[env_name]
            for name, env_name in {
                "max_rounds": "SKILLFLOW_AGENT_MAX_ROUNDS",
                "max_tool_attempts": "SKILLFLOW_AGENT_MAX_TOOL_ATTEMPTS",
                "max_model_requests": "SKILLFLOW_AGENT_MAX_MODEL_REQUESTS",
                "model_timeout_seconds": "SKILLFLOW_AGENT_MODEL_TIMEOUT",
                "active_timeout_seconds": "SKILLFLOW_AGENT_ACTIVE_TIMEOUT",
            }.items() if env_name in os.environ
        })
