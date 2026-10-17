"""核心 Agent 循环。

This is the heart of CoreCoder.  The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text (no tool calls),
which means it's done working and ready to report back.
"""

import concurrent.futures
import inspect

from .context import ContextManager
from .llm import LLM
from .permissions import Permission
from .prompt import PLAN_MODE_PROMPT, system_prompt
from .tools import ALL_TOOLS
from .tools.agent import AgentTool
from .tools.base import Tool
from .tools.todo import TodoWriteTool


class Agent:
    """负责在用户、LLM 和工具之间循环编排的核心代理。"""

    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
        permission=None,
        hooks=None,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS
        self.permission = permission
        self.hooks = hooks
        self._tool_by_name = {t.name: t for t in self.tools}
        self.messages: list[dict] = []          ## 对话历史
        self.context = ContextManager(max_tokens=max_context_tokens)    ## 上下文管理器
        self.max_rounds = max_rounds        ## 最多转几圈
        self._system = system_prompt(self.tools)
        # /plan 命令会切换该标志；开启时所有写操作都会被拒绝。
        self.plan_mode = False      # # 计划模式开关

        # 把当前 Agent 注入子 Agent 工具，使其可以创建受控的子任务。
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

        self._todo = next((t for t in self.tools if isinstance(t, TodoWriteTool)), None)

    def _full_messages(self) -> list[dict]:
        """构造本轮发送给 LLM 的完整消息，动态注入模式和任务清单。"""

        system = self._system
        # 每轮重新注入，保证计划模式切换后下一次请求立即生效。
        if self.plan_mode:
            system += "\n\n" + PLAN_MODE_PROMPT
        # 每轮重新注入最新任务清单，避免模型看到旧的工具输出副本。
        if self._todo is not None:
            rendered = self._todo.render()
            if rendered:
                system += "\n\n# Current task list\n" + rendered
        return [{"role": "system", "content": system}] + self.messages

    def _tool_schemas(self) -> list[dict]:
        """收集所有工具的 JSON Schema，供模型生成工具调用。"""

        return [t.schema() for t in self.tools]

    def chat(self, user_input: str, on_token=None, on_tool=None, on_reasoning=None) -> str:
        """处理一条用户消息，可能经历多轮 LLM 与工具调用。"""
        #接收用户输入把用户消息加进历史。检查上下文是否太长，太长就压缩。
        self.messages.append({"role": "user", "content": user_input})
        self.context.maybe_compress(self.messages, self.llm)

        for _ in range(self.max_rounds):
            resp = self.llm.chat(
                messages=self._full_messages(),
                tools=self._tool_schemas(),
                on_token=on_token,
                on_reasoning=on_reasoning,
            )

            # 没有工具调用表示模型已经完成任务，直接返回文本答案。
            if not resp.tool_calls:
                self.messages.append(resp.message)
                return resp.content

            # 有工具调用则执行；多个互不依赖的调用会并发执行。
            self.messages.append(resp.message)

            try:
                ## 单个工具
                if len(resp.tool_calls) == 1:
                    tc = resp.tool_calls[0]
                    if on_tool:
                        on_tool(tc.name, tc.arguments)
                    result = self._pre_hooks(tc) or self._permit(tc)
                    if result is None:
                        result = self._exec_tool(tc)
                        self._post_hooks(tc, result)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                else:
                    #多个工具，并发执行
                    # parallel execution for multiple tool calls
                    results = self._exec_tools_parallel(resp.tool_calls, on_tool)
                    for tc, result in zip(resp.tool_calls, results):
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
            except KeyboardInterrupt:
                # 中断时补齐未完成的 tool 回复，避免下一次请求历史格式非法。
                self._answer_pending_tool_calls(resp.tool_calls)
                raise

            # 工具输出可能很长，每轮结束后检查是否需要压缩上下文。
            self.context.maybe_compress(self.messages, self.llm)

        return "(reached maximum tool-call rounds)"

    def _pre_hooks(self, tc) -> str | None:
        """在权限检查前执行前置钩子；返回字符串即拦截工具调用。"""
        if self.hooks is None:
            return None
        return self.hooks.run_pre(tc.name, tc.arguments)

    def _post_hooks(self, tc, result: str):
        """执行后置钩子观察调用结果；后置钩子不能阻止调用。"""
        if self.hooks is not None:
            self.hooks.run_post(tc.name, tc.arguments, result)

    def _permit(self, tc) -> str | None:
        """检查单次调用权限；None 表示放行，字符串表示拒绝原因。"""
        # 计划模式优先级高于 --yes，开启后不允许任何修改操作。
        if self.plan_mode and tc.name not in Permission.READ_ONLY:
            return (
                "Plan mode is on, so this call was refused: plan mode is "
                "read-only. Do not retry it. Keep investigating with the "
                "read-only tools, then present the plan and stop. The user can "
                'approve it by typing "approve", or exit plan mode with /plan.'
            )
        if self.permission is None:
            return None
        return self.permission.check(tc.name, tc.arguments)

    def _exec_tool(self, tc) -> str:
        """校验并执行单个工具调用，把异常转换为模型可读的错误文本。"""
        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            return f"Error: unknown tool '{tc.name}'"
        # 先校验参数，避免把工具内部产生的 TypeError 误判为调用参数错误。
        try:
            inspect.signature(tool.execute).bind(**tc.arguments)
        except TypeError as e:
            return f"Error: bad arguments for {tc.name}: {e}"
        # 工具异常作为文本返回给模型，不让单个工具终止主循环。
        try:
            return tool.execute(**tc.arguments)
        except Exception as e:  # noqa: BLE001
            return f"Error executing {tc.name}: {e}"

    def _exec_tools_parallel(self, tool_calls, on_tool=None) -> list[str]:
        """使用线程并发执行多个工具调用，并按原顺序返回结果。

        This is inspired by Claude Code's StreamingToolExecutor which starts
        executing tools while the model is still generating.  We simplify to:
        when the model returns N tool calls at once, run them in parallel.
        """
        from .tools.bash import get_tracked_cwd, set_tracked_cwd

        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, tc.arguments)

        # 先在当前线程完成钩子和权限检查，避免多个线程交错询问终端用户。
        results = [self._pre_hooks(tc) or self._permit(tc) for tc in tool_calls]
        # 工作目录是线程本地的，因此把会话 cwd 传入线程，并按调用顺序合并 cd 结果。
        session_cwd = get_tracked_cwd()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            def run(i, tc, base_cwd):
                is_bash = tc.name == "bash"
                if is_bash and base_cwd is not None:
                    set_tracked_cwd(base_cwd)
                out = self._exec_tool(tc)
                if not is_bash:
                    return i, out, None
                cwd_after = get_tracked_cwd()
                # only a worker that actually moved the dir reports back; a
                # sibling that merely inherited the base must not revert a cd
                return i, out, cwd_after if cwd_after != base_cwd else None

            futures = {
                i: pool.submit(run, i, tc, session_cwd)
                for i, tc in enumerate(tool_calls)
                if results[i] is None
            }
            for i, future in futures.items():
                i, results[i], worker_cwd = future.result()
                if worker_cwd is not None:
                    session_cwd = worker_cwd
                    set_tracked_cwd(worker_cwd)
        for i in futures:
            self._post_hooks(tool_calls[i], results[i])
        return results

    def _answer_pending_tool_calls(self, tool_calls):
        """为尚未得到回复的工具调用补写结果，保持消息历史完整。

        OpenAI-compatible APIs reject a request where an assistant message has
        tool_calls without a matching tool reply for each id, so this keeps the
        history valid when execution is interrupted partway through.
        """
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        for tc in tool_calls:
            if tc.id not in answered:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "[interrupted]",
                })

    def reset(self):
        """清空对话历史和当前任务清单。"""
        self.messages.clear()
        # 重置后不能把上一轮会话的任务清单泄漏到下一次系统提示词。
        todo = self._tool_by_name.get("todo_write")
        if isinstance(todo, TodoWriteTool):
            todo._tasks = []
