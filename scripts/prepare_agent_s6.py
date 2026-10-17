"""原库就地升级的数据准备：先SQLite在线备份，再增量补样本，绝不重绑定登录身份。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import sys


def main():
    """默认只预览；服务停止后显式 --apply，所有读写路径均受项目 B 根目录约束。"""
    root=Path(__file__).resolve().parent.parent
    sys.path.insert(0,str(root))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=root/'.skillflow')
    parser.add_argument('--buyers',nargs=2,default=['alice-buyer','other-buyer'])
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    data=args.data_dir.resolve()
    if not data.is_relative_to(root) or not (data/'skillflow.sqlite3').is_file():
        parser.error('必须选择项目B已有业务库，不创建新库冒充原库')
    database=data/'skillflow.sqlite3'
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as db:
        owners={r[0] for r in db.execute('SELECT DISTINCT buyer_id FROM identity_bindings')}
        if not set(args.buyers)<=owners:
            parser.error('买家必须已被原身份绑定引用，不在准备脚本中改绑')
        if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':
            parser.error('原数据库检查失败')
        counts={'orders':db.execute('SELECT COUNT(*) FROM business_orders').fetchone()[0],
                'tickets':db.execute('SELECT COUNT(*) FROM tickets').fetchone()[0]}
    print(json.dumps({'data_dir':str(data),'buyers':args.buyers,'before':counts,'apply':args.apply},ensure_ascii=False))
    if not args.apply:
        return 0
    backup=data/'deploy-backups'/('pre-s6-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.sqlite3')
    backup.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(database) as source,sqlite3.connect(backup) as destination:
        source.backup(destination)
    # 备份完成后才允许ORM初始化/增量表与案例写入；模型和身份提供方均不参与。
    from skillflow.business.surge import BusinessRepository
    from skillflow.business.s6_samples import prepare_samples
    repo=BusinessRepository(database,root/'data')
    try:
        manifest=prepare_samples(repo,root/'data',data/'s6-fixtures',args.buyers)
    finally:
        repo.engine.dispose()
    print('BACKUP='+str(backup))
    print('S6_CASES='+str(len(manifest['cases'])))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
