"""只读 B 自有固定物流文件；服务端独立复核记录归属，不访问订单/工单数据库。"""

import json
from pathlib import Path

from .contracts import LogisticsResult


class LogisticsRepository:
    """无记录返回 NO_RECORD；存在记录但买家不同则拒绝且不返回轨迹。"""
    def __init__(self, path: Path):
        self.path = path.resolve()
        if not self.path.is_relative_to(Path(__file__).resolve().parents[3]):
            raise ValueError('LOGISTICS_PATH_FORBIDDEN')

    def get(self, order_id: str, buyer_id: str) -> LogisticsResult:
        """每次读文件，允许独立验收修改来源版本；文件损坏由协议适配层映射故障。"""
        data = json.loads(self.path.read_text(encoding='utf-8'))
        row = data['orders'].get(order_id)
        common = dict(order_id=order_id, updated_at=None, events=[], source_version=data['version'])
        if row is None:
            return LogisticsResult(**common, status='empty', logistics_status='NO_RECORD')
        if row['buyer_id'] != buyer_id:
            return LogisticsResult(**common, status='forbidden', logistics_status='UNKNOWN')
        return LogisticsResult(order_id=order_id, status='found', source_version=data['version'],
            logistics_status=row['logistics_status'], updated_at=row['updated_at'], events=row['events'])
