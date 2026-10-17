"""官方 SDK FastMCP stdio 入口：stdout 只传协议；只读模拟配送，不生成退货运单。"""

import os
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .contracts import LogisticsResult, verify_grant
from .repository import LogisticsRepository


def main():
    """受管子进程只读取最小环境；缺少密钥立即退出，不加载项目 .env。"""
    secret = os.environ['SKILLFLOW_LOGISTICS_SECRET']
    repository = LogisticsRepository(Path(os.environ['SKILLFLOW_LOGISTICS_DATA']))
    server = FastMCP('skillflow-logistics', log_level='ERROR')

    @server.tool()
    async def get_logistics(order_id: Annotated[str, Field(pattern=r'^A[0-9]{3,8}$')],
                            access_grant: str = '') -> LogisticsResult:
        """查询已授权买家的订单配送；access_grant 仅由可信适配器注入。"""
        buyer = verify_grant(secret, access_grant, order_id)
        return repository.get(order_id, buyer)

    server.run(transport='stdio')


if __name__ == '__main__':
    main()
