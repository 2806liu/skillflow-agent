"""仅供独立子进程故障验收的 MCP 服务；故障开关不进入正式物流服务。"""

import argparse
import asyncio
import os
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from skillflow.integrations.logistics_mcp.contracts import LogisticsResult, verify_grant


def main():
    """固定故障模式覆盖单请求超时隔离、退出、畸形结果和 Schema 漂移。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)
    mode = parser.parse_args().mode
    server = FastMCP('test-logistics', log_level='ERROR')
    if mode == 'startup_exit':
        raise SystemExit(1)

    @server.tool(structured_output=True)
    async def get_logistics(order_id: Annotated[str, Field(pattern=r'^A[0-9]{3,8}$')], access_grant: str = '') -> dict[str, Any]:
        """先授权后故障；A123 慢查询与 A124 快查询可共享一个协议连接。"""
        verify_grant(os.environ['SKILLFLOW_LOGISTICS_SECRET'], access_grant, order_id)
        if mode == 'exit':
            os._exit(11)
        if mode == 'slow' and order_id == 'A123':
            await asyncio.sleep(.6)
        if mode == 'malformed':
            return {'order_id': order_id, 'status': 'found', 'secret_unexpected': 'bad'}
        if mode == 'wrong_order':
            order_id = 'A999'
        return LogisticsResult(order_id=order_id, status='found', logistics_status='DELIVERED',
            updated_at='2026-10-16T00:00:00Z', events=[], source_version='fault-fixture-v1').model_dump()

    if mode == 'schema':
        @server.tool()
        async def arbitrary_tool(command: str) -> str:
            """额外工具必须使白名单发现失败，不能开放给模型。"""
            return 'not executable'
    server.run(transport='stdio')


if __name__ == '__main__':
    main()
