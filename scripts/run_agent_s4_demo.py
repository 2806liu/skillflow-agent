"""项目 B 独立 S4 演示：真实模型、模拟订单、检查点与确认卡片；不修改 .env 或 S3 数据。"""

import argparse
import os
from pathlib import Path
import sys


def main():
    """独立目录和端口，业务样本仅首次初始化；模型配置缺失时明确停止。"""
    root=Path(__file__).resolve().parent.parent
    sys.path.insert(0,str(root))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8006)
    args=parser.parse_args()
    os.environ.update({'SKILLFLOW_AGENT_ENGINE':'tool_loop','SKILLFLOW_AGENT_MODEL_MODE':'langchain_openai',
        'SKILLFLOW_AGENT_S4_ENABLED':'true','SKILLFLOW_AGENT_CONFIRMATION_SECONDS':'180',
        'SKILLFLOW_MODEL_MODE':'mock','SKILLFLOW_AUTH_MODE':'demo','SKILLFLOW_DEMO_BUYER_ID':'demo-buyer',
        'SKILLFLOW_BUSINESS_BACKEND':'sqlalchemy','SKILLFLOW_DATA_DIR':str(root/'.skillflow/agent-s4-demo'),
        'SKILLFLOW_CHAT_LOCK_MODE':'session','SKILLFLOW_MAX_ACTIVE_CHATS':'8','SKILLFLOW_MAX_WAITING_CHATS':'64'})
    from skillflow.api import create_app
    from skillflow.business.surge import Product
    from skillflow.config import Settings
    from evaluation.agent_runtime.s2_fixtures import seed_quality_demo
    import uvicorn
    settings=Settings()
    if not (settings.model and settings.api_key):
        raise SystemExit('请配置项目 B 自有模型服务；不会回退为脚本。')
    app=create_app(settings)
    with app.state.repository.sessions() as db:
        seeded=db.get(Product,'S2-PB-A') is not None
    if not seeded:
        seed_quality_demo(app.state.repository)
    uvicorn.run(app,host='127.0.0.1',port=args.port)


if __name__=='__main__':
    main()
