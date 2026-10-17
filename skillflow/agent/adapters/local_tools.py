"""本地业务适配：复用已有只读处理器，将同步仓库操作放入工作线程。"""

import asyncio
import hashlib
import json

from skillflow.contracts import ToolResult
from skillflow.tools import create_tools


def fingerprint(value) -> str:
    """用稳定 JSON 摘要比较事实，不把 Python 对象序列化进状态。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class LocalBusinessAdapter:
    """每任务绑定服务端买家，SQLAlchemy Session 仍按仓库每操作创建。"""

    def __init__(self, repository, policies, buyer_id: str):
        self.repository, self.policies, self.buyer_id = repository, policies, buyer_id
        self._tools = create_tools(repository, buyer_id, policies)

    async def owner(self, order_id: str):
        """归属只用于执行边界判断，不返回买家信息给模型。"""
        return await asyncio.to_thread(self.repository.owner_of_order, order_id)

    async def order(self, order_id: str):
        """即使已检查归属，实际读取仍绑定当前买家。"""
        return await asyncio.to_thread(self.repository.get_order, order_id, self.buyer_id)

    async def source_version(self, tool_name: str, order: dict) -> str:
        """保留当前政策集合或事件结果版本，供前置条件与证据过期检查。"""
        if tool_name in {"retrieve_policy", "assess_after_sales", "check_inventory", "analyze_discount", "analyze_price_adjustment"}:
            return fingerprint(sorted(doc_id for doc_id, doc in self.policies.documents.items()
                                      if doc["policy_key"] == order["policy_key"]))
        if tool_name in {"match_quality_incident", "list_alternatives"}:
            return fingerprint(await asyncio.to_thread(self.repository.evaluate, order["order_id"], self.buyer_id))
        return "order_only"

    async def invoke(self, name: str, arguments: dict) -> ToolResult:
        """只接收执行器准备的内部参数；不经过旧 Planner 或旧 Agent 图。"""
        if name == "list_my_orders":
            page = await asyncio.to_thread(self.repository.query_orders, self.buyer_id, **arguments)
            return ToolResult(status="success" if page["items"] else "empty", data=page,
                              error_code=None if page["items"] else "ORDER_NOT_FOUND")
        if name == "match_quality_incident":
            value = await asyncio.to_thread(self.repository.evaluate, arguments["order_id"], self.buyer_id)
            return ToolResult(status="success" if value else "empty", data=value or {},
                              error_code=None if value else "ORDER_NOT_FOUND")
        if name == "list_alternatives":
            order = await self.order(arguments["order_id"])
            items = await asyncio.to_thread(self.repository.alternatives, order["sku"])
            return ToolResult(status="success" if items else "empty", data={"items": items})

        # 旧处理器中的同步 SQLite/JSON 读取也在线程中运行；保留 StructuredTool 参数和返回校验。
        return await asyncio.to_thread(self._invoke_typed_tool, name, arguments)

    def _invoke_typed_tool(self, name: str, arguments: dict) -> ToolResult:
        """工作线程独享事件循环，不阻塞主 Agent 调度循环。"""
        return asyncio.run(self._tools[name].call(arguments))
