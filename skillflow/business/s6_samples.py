"""S6 增量模拟案例：保留原买家、订单及登录绑定，不迁移旧任务或执行图。"""

import json
from pathlib import Path
import shutil

from .surge import Buyer, Order, OrderItem, Product


def write_json(path: Path, value):
    """先写同目录临时文件再替换，避免半份 JSON 被服务读取。"""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def prepare_samples(repository, source: Path, target: Path, buyers: list[str]):
    """为已存在的两个买家补充八类案例；清单存在时校验归属并幂等返回。"""
    from skillflow.config import ROOT
    source, target = source.resolve(), target.resolve()
    if not all(p.is_relative_to(ROOT.resolve()) for p in (source, target)):
        raise ValueError('S6 样本必须位于项目 B 内')
    if len(buyers) != 2 or len(set(buyers)) != 2 or not all(repository.has_buyer(b) for b in buyers):
        raise ValueError('必须提供原库中两个不同的已存在买家，不能自动改绑或认领订单')
    manifest = target / 's6-cases.json'
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding='utf-8'))
        if data['buyers'] != buyers:
            raise ValueError('S6 已登记买家不同，拒绝覆盖样本')
        for row in data['cases']:
            if repository.owner_of_order(row['order_id']) != row['buyer_id']:
                raise ValueError('S6 订单归属已变化，拒绝继续')
        return data
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / 'policies', target / 'policies', dirs_exist_ok=True)
    files = {n: json.loads((source / n).read_text(encoding='utf-8'))
             for n in ('buyers.json', 'orders.json', 'inventory.json', 'price_adjustments.json')}
    for buyer in buyers:
        files['buyers.json'].setdefault(buyer, {'buyer_id': buyer, 'display_name': buyer})
    cases, logistics = [], {'version': 's6-delivery-v1', 'orders': {}}
    labels = [('normal', '正常退换'), ('no_stock', '换货缺货'), ('no_policy', '缺少政策'),
              ('expired', '超出期限'), ('affected', '批次受影响'), ('unaffected', '批次范围外'),
              ('unverified', '批次待核实'), ('incident_expired', '事件超期')]
    number = 610000
    # 新单号在原库查重；不把 A123 等原演示订单转给 Alice，也不删除既有工单。
    for buyer in buyers:
        for kind, label in labels:
            while repository.owner_of_order(f'A{number}'):
                number += 1
            oid = f'A{number}'
            number += 1
            incident = kind in {'affected', 'unaffected', 'unverified', 'incident_expired'}
            sku = 'S6-PB' if incident else ('S6-EAR-WHITE' if kind == 'no_stock' else 'S6-EAR-BLACK')
            policy = 's6_missing' if kind == 'no_policy' else 'earbuds_v1'
            row = dict(order_id=oid, buyer_id=buyer, product='S6 模拟充电宝' if incident else 'S6 演示耳机',
                       sku=sku, policy_key=policy, delivered_days_ago=30 if kind in {'expired', 'incident_expired'} else 2,
                       quality_issue=True, paid_cents=9900 if incident else 18900,
                       original_cents=10900 if incident else 19900, discount_cents=1000, status='已签收')
            files['orders.json'][oid] = row
            files['inventory.json'][sku] = 0 if kind == 'no_stock' or incident else 5
            case = {'buyer_id': buyer, 'order_id': oid, 'kind': kind, 'label': label}
            cases.append(case)
            if kind != 'no_policy':
                logistics['orders'][oid] = {'buyer_id': buyer, 'logistics_status': 'DELIVERED',
                    'updated_at': '2026-10-14T08:00:00+08:00', 'events': [
                        {'time': '2026-10-14T08:00:00+08:00', 'description': '项目 B 固定模拟签收记录'}]}
    with repository.sessions.begin() as db:
        for sku, name, category, stock in [('S6-PB', 'S6 模拟充电宝', 'power_bank', 0),
                ('S6-PB-ALT', 'S6 在售替代候选', 'power_bank', 8),
                ('S6-EAR-BLACK', 'S6 演示耳机', 'legacy', 5), ('S6-EAR-WHITE', 'S6 白色耳机', 'legacy', 0)]:
            if db.get(Product, sku):
                raise ValueError('S6 商品已存在但清单缺失，需要核对，不能重置原数据')
            db.add(Product(sku=sku, name=name, category=category, brand='S6 模拟品牌', model=sku,
                           stock=stock, status='active'))
        db.flush()
        for c in cases:
            r = files['orders.json'][c['order_id']]
            kind = c['kind']
            batch = None if kind == 'unverified' else ('B090' if kind == 'unaffected' else 'B025')
            db.add(Order(**{k: r[k] for k in ('order_id','buyer_id','delivered_days_ago','quality_issue',
                        'paid_cents','original_cents','discount_cents','status')}, item=OrderItem(
                            sku=r['sku'], batch_code=batch if r['sku']=='S6-PB' else None,
                            batch_source='merchant_order' if r['sku']=='S6-PB' else 'missing',
                            batch_verified=r['sku']=='S6-PB' and batch is not None)))
    repository.create_incident(incident_id='S6-INCIDENT', actor='s6-fixture-author', evidence='项目 B 固定模拟质量证据',
        explanation='S6 批次 B020 至 B030，固定模拟规则', sku='S6-PB', batch_start='B020', batch_end='B030',
        special_return_days=7)
    repository.change_incident('S6-INCIDENT', actor='s6-fixture-reviewer', action='publish', reason='模拟审核通过')
    for name, value in files.items():
        write_json(target / name, value)
    write_json(target / 'logistics.json', logistics)
    data = {'version': 's6-cases-v1', 'buyers': buyers, 'cases': cases}
    write_json(manifest, data)
    return data


def examples_for(fixtures: Path, buyer: str, repository):
    """仅返回当前买家且归属仍有效的具名案例；普通生产环境无清单时返回空。"""
    path = fixtures / 's6-cases.json'
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding='utf-8'))['cases']
    result = []
    for row in rows:
        if row['buyer_id'] != buyer or repository.owner_of_order(row['order_id']) != buyer:
            continue
        oid = row['order_id']
        message = (f'{oid} 是否受质量事件影响，能否退货，有哪些替代品？'
                   if row['kind'] in {'affected','unaffected','unverified','incident_expired'}
                   else f'查询订单 {oid}，结合店铺政策调查能否退换。')
        # 网页用结构化场景和订单号生成物流提示，不解析中文标签，也不硬编码其他买家的单号。
        result.append({'label': f'{oid} · {row["label"]}', 'message': message,
                       'kind': row['kind'], 'order_id': oid})
    return result
