"""离线建立项目 B 的 OIDC 主体与买家绑定，不经浏览器接受自报 buyer_id。"""

import argparse
from pathlib import Path

from skillflow.business.repository import MockOrderRepository
from skillflow.business.surge import BusinessRepository
from skillflow.config import ROOT, Settings
from skillflow.sqlite_store import SqliteStore


def main():
    """管理员指定已核验的 OIDC sub 与现有买家；未知买家或缺少生产配置直接失败。"""
    parser = argparse.ArgumentParser(description="离线绑定项目 B 的可信 OIDC 主体与买家")
    parser.add_argument("--subject", required=True, help="身份提供方的稳定 sub")
    parser.add_argument("--buyer", required=True, help="项目 B 的已验证买家 ID")
    parser.add_argument("--role", choices=("buyer", "admin"), default="buyer")
    parser.add_argument("--data-dir", type=Path,
                        help="可选：项目 B 内的隔离数据目录；省略时使用原有 .skillflow")
    args = parser.parse_args()
    # 本机质量事件演示可使用独立 SQLite；禁止把身份映射写到项目 B 之外。
    data_dir = args.data_dir.resolve() if args.data_dir else None
    if data_dir and not data_dir.is_relative_to(ROOT.resolve()):
        parser.error("数据目录必须位于项目 B 内")
    settings = Settings(data_dir=data_dir) if data_dir else Settings()
    if settings.auth_mode != "production":
        parser.error("请先设置 SKILLFLOW_AUTH_MODE=production 并配置身份提供方")
    store = SqliteStore(settings.data_dir)
    # SQLAlchemy 模式检查隔离库买家；旧 Mock 模式保留现有检查。
    repository = (BusinessRepository(store.database_path, settings.fixtures_dir)
                  if settings.business_backend == "sqlalchemy" else MockOrderRepository(settings.fixtures_dir))
    if not repository.has_buyer(args.buyer):
        parser.error("项目 B 中不存在该买家；不得自动认领旧演示数据")
    store.provision_identity(settings.oidc_issuer, args.subject, args.buyer, args.role)
    print("身份映射已保存；认证令牌与 Cookie 未写入业务记录。")


if __name__ == "__main__":
    main()
