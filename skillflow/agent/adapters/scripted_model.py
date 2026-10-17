"""无需网络的脚本模型，用于验证运行时；不会产生订单或售后结论。"""

from ..contracts import MessageEnvelope, ModelReply, ToolSpec


class ScriptedModel:
    """无共享会话游标，每次调用返回独立消息，便于并行会话验证。"""

    async def generate(self, messages: list[MessageEnvelope], tool_specs: list[ToolSpec]) -> ModelReply:
        """S1 不开放业务工具；真实 tool_calls 循环留到 S3。"""
        return ModelReply(message=MessageEnvelope(role="assistant", content="S1_SCRIPTED_READY"))
