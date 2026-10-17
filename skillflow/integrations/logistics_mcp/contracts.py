"""MCP 双端共用的只读结果契约与短时能力凭据；凭据不进入模型或检查点。"""

import time
from typing import Literal

import jwt
from pydantic import BaseModel, ConfigDict, Field


class LogisticsEvent(BaseModel):
    """固定模拟运输事件；禁止服务端任意增加内部字段。"""
    model_config = ConfigDict(extra='forbid')
    time: str = Field(min_length=1, max_length=40)
    description: str = Field(min_length=1, max_length=200)


class LogisticsResult(BaseModel):
    """业务空结果与协议失败分开；时间表示来源更新时间，不承诺实时轨迹。"""
    model_config = ConfigDict(extra='forbid')
    order_id: str = Field(pattern=r'^A[0-9]{3,8}$')
    status: Literal['found', 'empty', 'forbidden']
    logistics_status: Literal['DELIVERED', 'IN_TRANSIT', 'EXCEPTION', 'NO_RECORD', 'UNKNOWN']
    updated_at: str | None
    events: list[LogisticsEvent] = Field(max_length=20)
    source: Literal['project_b_mock_logistics'] = 'project_b_mock_logistics'
    source_version: str = Field(min_length=1, max_length=80)


def grant(secret: str, buyer_id: str, order_id: str) -> str:
    """签名授权只绑定一次订单查询范围，密钥由父进程生成并经受控环境传给子进程。"""
    now = int(time.time())
    return jwt.encode({'sub': buyer_id, 'order': order_id, 'tool': 'get_logistics',
                       'iat': now, 'exp': now + 30, 'aud': 'skillflow-logistics'}, secret, algorithm='HS256')


def verify_grant(secret: str, token: str, order_id: str) -> str:
    """固定算法、用途和有效期；拒绝后只返回统一错误，不泄露凭据正文。"""
    try:
        value = jwt.decode(token, secret, algorithms=['HS256'], audience='skillflow-logistics',
                           options={'require': ['sub', 'order', 'tool', 'iat', 'exp']})
        if (value['order'] != order_id or value['tool'] != 'get_logistics' or
                not isinstance(value['sub'], str) or not value['sub'] or value['exp'] - value['iat'] > 30):
            raise ValueError()
        return value['sub']
    except Exception:
        raise ValueError('MCP_ACCESS_DENIED') from None
