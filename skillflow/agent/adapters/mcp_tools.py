"""将发现并转换后的 MCP 工具接入原工具端口，保持订单授权和协议传输解耦。"""

from datetime import datetime, timezone

from skillflow.contracts import ToolResult
from skillflow.integrations.logistics_mcp.contracts import LogisticsResult
from .mcp_session import McpFailure


class LogisticsBusinessAdapter:
    """组合本地工具，不复制业务规则；MCP 调用前由 ToolSession 核验当前本人订单。"""
    def __init__(self, local, manager):
        self.local, self.manager = local, manager

    async def owner(self, order_id):
        """归属检查仍读取可信订单仓库。"""
        return await self.local.owner(order_id)

    async def order(self, order_id):
        """每次读取绑定服务端当前买家。"""
        return await self.local.order(order_id)

    async def source_version(self, name, order):
        """物流具体来源版本保存在结果中，不额外产生未计预算的 MCP 请求。"""
        return 'logistics-mcp-v1' if name == 'get_logistics' else await self.local.source_version(name, order)

    async def invoke(self, name, arguments):
        """畸形结果、权限拒绝、无记录和网络故障保持不同的稳定语义。"""
        if name != 'get_logistics':
            return await self.local.invoke(name, arguments)
        try:
            reply = await self.manager.call(self.local.buyer_id, arguments['order_id'])
            value = LogisticsResult.model_validate((reply.artifact or {})['structured_content'])
            if value.order_id != arguments['order_id']:
                raise ValueError()
            if value.status == 'forbidden':
                return ToolResult(status='failed', error_code='MCP_ACCESS_DENIED', message='物流服务拒绝此订单访问。')
            if ((value.status == 'empty') != (value.logistics_status == 'NO_RECORD') or
                    (value.status == 'found' and not value.updated_at)):
                raise ValueError()
            return ToolResult(status='empty' if value.status == 'empty' else 'success',
                data={**value.model_dump(mode='json'), 'queried_at': datetime.now(timezone.utc).isoformat(),
                      'transport': 'stdio', 'mcp_tool': 'logistics_get_logistics'},
                message='暂无配送物流记录。' if value.status == 'empty' else '已读取项目 B 固定模拟配送记录。')
        except McpFailure as exc:
            return ToolResult(status='failed', error_code=exc.code, retryable=exc.retryable,
                              message='物流查询未完成：' + exc.code)
        except Exception:
            return ToolResult(status='failed', error_code='MCP_INVALID_RESPONSE', message='物流服务响应不符合约定。')
