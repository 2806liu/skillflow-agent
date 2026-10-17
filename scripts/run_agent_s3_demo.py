"""启动项目 B 独立 S3 真实模型只读演示，保留 legacy 和 S1 的数据与进程。"""

import argparse
import os
from pathlib import Path
import sys


def main():
    """只写 B 自有演示目录；首次补充固定质量样本，重复启动不重复建事件。"""
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8005)
    args = parser.parse_args()
    os.environ.update({"SKILLFLOW_AGENT_ENGINE": "tool_loop", "SKILLFLOW_AGENT_MODEL_MODE": "langchain_openai",
        "SKILLFLOW_MODEL_MODE": "mock", "SKILLFLOW_AUTH_MODE": "demo", "SKILLFLOW_DEMO_BUYER_ID": "demo-buyer",
        "SKILLFLOW_BUSINESS_BACKEND": "sqlalchemy", "SKILLFLOW_DATA_DIR": str(root / ".skillflow/agent-s3-demo"),
        "SKILLFLOW_CHAT_LOCK_MODE": "session", "SKILLFLOW_MAX_ACTIVE_CHATS": "8", "SKILLFLOW_MAX_WAITING_CHATS": "64"})
    from skillflow.api import create_app
    from skillflow.business.surge import Product
    from skillflow.config import Settings
    from evaluation.agent_runtime.s2_fixtures import seed_quality_demo
    import uvicorn
    settings = Settings()
    if not (settings.model and settings.api_key):
        raise SystemExit("请在项目 B .env 中配置模型和 API Key；不会回退到脚本模式。")
    app = create_app(settings)
    repository = app.state.repository
    with repository.sessions() as db:
        seeded = db.get(Product, "S2-PB-A") is not None
    if not seeded:
        seed_quality_demo(repository)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
