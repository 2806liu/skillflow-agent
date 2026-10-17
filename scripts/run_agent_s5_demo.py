"""S5 独立网页演示：真实模型及 stdio MCP，复用 B 模拟数据，使用独立 8007 端口和数据目录。"""

import os
from pathlib import Path
import sys


def main():
    """只设置当前启动进程；模型配置由 B 原配置读取，S4 数据和 .env 不修改。"""
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8007)
    args = parser.parse_args()
    os.environ.update({'SKILLFLOW_AGENT_ENGINE': 'tool_loop', 'SKILLFLOW_AGENT_MODEL_MODE': 'langchain_openai',
        'SKILLFLOW_AGENT_S4_ENABLED': 'true', 'SKILLFLOW_AGENT_S5_ENABLED': 'true',
        'SKILLFLOW_AGENT_CONFIRMATION_SECONDS': '180', 'SKILLFLOW_AGENT_MCP_QUERY_TIMEOUT': '10',
        'SKILLFLOW_AGENT_MCP_STARTUP_TIMEOUT': '10', 'SKILLFLOW_MODEL_MODE': 'mock',
        'SKILLFLOW_AUTH_MODE': 'demo', 'SKILLFLOW_DEMO_BUYER_ID': 'demo-buyer',
        'SKILLFLOW_BUSINESS_BACKEND': 'sqlalchemy', 'SKILLFLOW_DATA_DIR': str(root / '.skillflow/agent-s5-demo'),
        'SKILLFLOW_CHAT_LOCK_MODE': 'session', 'SKILLFLOW_MAX_ACTIVE_CHATS': '8', 'SKILLFLOW_MAX_WAITING_CHATS': '64'})
    from skillflow.config import Settings
    from skillflow.api import create_app
    from skillflow.business.surge import Product
    from evaluation.agent_runtime.s2_fixtures import seed_quality_demo
    import uvicorn
    settings = Settings()
    if not (settings.model and settings.api_key):
        raise SystemExit('请配置项目 B 自有模型服务；不会回退成脚本。')
    app = create_app(settings)
    with app.state.repository.sessions() as db:
        seeded = db.get(Product, 'S2-PB-A') is not None
    if not seeded:
        seed_quality_demo(app.state.repository)
    uvicorn.run(app, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
