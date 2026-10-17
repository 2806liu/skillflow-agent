"""真实 LangChain 模型适配：原生工具消息与结构化业务对象分开，不执行任何工具。"""

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from ..contracts import MessageEnvelope, ModelReply, ToolCall
from ..goals import AnswerProposal, GoalProposal


def native_messages(messages):
    """保留真实调用 ID；每个工具观察都转回原生 ToolMessage。"""
    result = []
    for message in messages:
        if message.role == "assistant":
            result.append(AIMessage(content=message.content, tool_calls=[
                {"id": call.call_id, "name": call.name, "args": call.arguments, "type": "tool_call"}
                for call in message.tool_calls]))
        elif message.role == "tool":
            result.append(ToolMessage(content=message.content, tool_call_id=message.tool_call_id))
        else:
            result.append((SystemMessage if message.role == "system" else HumanMessage)(content=message.content))
    return result


class ChatModelGateway:
    """客户端生命周期归应用；禁 SDK 隐式重试，每次网络请求由图计预算。"""

    def __init__(self, settings):
        self.s4_enabled = settings.agent_s4_enabled
        # 配置缺失由 HTTP 受理预检拒绝；构造应用本身不触发 SDK 缺密钥异常。
        self.http, self.sync_http, self.model = None, None, None
        if not settings.model or not settings.api_key:
            return
        self.http = httpx.AsyncClient(timeout=settings.model_timeout, trust_env=settings.model_trust_env)
        # 同步客户端也显式关闭通用代理继承，SDK 构造时不另建自动代理传输。
        self.sync_http = httpx.Client(timeout=settings.model_timeout, trust_env=settings.model_trust_env)
        self.model = ChatOpenAI(model=settings.model, api_key=settings.api_key,
            base_url=settings.base_url or None, temperature=0, timeout=settings.model_timeout,
            max_retries=0, http_async_client=self.http, http_client=self.sync_http)

    async def interpret_goal(self, text):
        """识别目标与原文商品关键词；不输出单号授权或执行计划。"""
        return await self.model.with_structured_output(GoalProposal, method="function_calling").ainvoke([
            SystemMessage(content="只解释本条明确的新电商售后需求，不继承旧任务。requirements可多选。仅查询订单选order；退换资格选returns；质量事件/批次选incident；替代品加alternatives。含糊不清的调查目的设置needs_clarification=true；离开订单/售后话题设置supported=false。不要把未知需求猜成普通订单查询。keyword仅在没有明确单号时提取原文商品关键词，如耳机或充电宝；不要加入订单、查询、我的等词，不确定则空字符串。不生成工具步骤。" +
                ("只有用户本轮明确要求为具体订单申请退货、换货、优惠争议或价保时 requested_action=apply，并设置对应request_type。能否申请、解释、引用示例、假设、否定都只能investigate；不能从历史目标继承申请授权。" if self.s4_enabled else "当前只读阶段，requested_action始终investigate，request_type=null。")),
            HumanMessage(content=text)])

    async def generate(self, messages, tool_specs):
        """bind_tools 只声明当前工具面，实际校验执行统一交给 Harness。"""
        tools = [{"type": "function", "function": {"name": spec.name,
                  "description": spec.description, "parameters": spec.parameters}} for spec in tool_specs]
        reply = await self.model.bind_tools(tools).ainvoke(native_messages(messages))
        if reply.invalid_tool_calls:
            raise ValueError("MODEL_INVALID_TOOL_CALL")
        return ModelReply(message=MessageEnvelope(role="assistant",
            content=reply.content if isinstance(reply.content, str) else "",
            tool_calls=[ToolCall(call_id=call["id"], name=call["name"], arguments=call["args"])
                        for call in reply.tool_calls]), usage={key: value for key, value in (reply.usage_metadata or {}).items()
                                                              if isinstance(value, int)})

    async def propose_answer(self, facts, refs):
        """最终对象只能排列事实索引和引用，不接收自由事实正文。"""
        import json
        return await self.model.with_structured_output(AnswerProposal, method="function_calling").ainvoke([
            SystemMessage(content="排列可信事实，fact_indexes必须是全部事实从0开始编号的完整排列，不能遗漏或重复。result_refs必须逐项完整复制提供的短别名列表，不要使用事实编号充当引用。仅调整顺序和tone，不能添加事实。"),
            HumanMessage(content=json.dumps({"facts": [{"index": i, "text": f} for i, f in enumerate(facts)],
                "required_fact_indexes": list(range(len(facts))), "required_result_refs": refs}, ensure_ascii=False))])

    async def aclose(self):
        """应用关闭时释放 HTTP 连接池。"""
        if self.http:
            await self.http.aclose()
        if self.sync_http:
            self.sync_http.close()
