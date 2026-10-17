"""子 Agent 创建工具。

The idea: for complex sub-tasks, spawn an independent agent with its own
conversation history and tool access. This lets the main agent delegate
work like "go research this codebase and report back" without polluting
its own context window.

The sub-agent runs to completion and returns a text summary.
"""

from typing import ClassVar

from .base import Tool


class AgentTool(Tool):
    """为复杂子任务创建独立上下文的 Agent。"""

    name = "agent"
    description = (
        "Spawn a sub-agent to handle a complex sub-task independently. "
        "The sub-agent has its own context and tool access. Use this for: "
        "researching a codebase, implementing a multi-step change in isolation, "
        "or any task that would benefit from a fresh context window."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What the sub-agent should accomplish",
            },
        },
        "required": ["task"],
    }

    # set by Agent.__init__ after construction
    _parent_agent = None

    def execute(self, task: str) -> str:
        """创建子 Agent 执行任务，并把结果压缩后返回给父 Agent。"""

        if self._parent_agent is None:
            return "Error: agent tool not initialized (no parent agent)"

        # 延迟导入，避免 Agent 与工具模块互相导入。
        from ..agent import Agent

        parent = self._parent_agent
        sub = Agent(
            llm=parent.llm,
            tools=[t for t in parent.tools if t.name != "agent"],  # no recursive agents
            max_context_tokens=parent.context.max_tokens,
            max_rounds=20,
            # the sub-agent answers to the same consent layer and hooks as the parent
            permission=parent.permission,
            hooks=parent.hooks,
        )

        # 子 Agent 的异常转成文本；同时恢复 cwd，避免子任务的 cd 泄漏给父任务。
        from .bash import get_tracked_cwd, set_tracked_cwd

        parent_cwd = get_tracked_cwd()
        try:
            result = sub.chat(task)
            # trim long results to avoid blowing up parent's context
            if len(result) > 5000:
                result = result[:4500] + "\n... (sub-agent output truncated)"
            return f"[Sub-agent completed]\n{result}"
        except Exception as e:  # noqa: BLE001
            return f"Sub-agent error: {e}"
        finally:
            if get_tracked_cwd() != parent_cwd:
                set_tracked_cwd(parent_cwd)
