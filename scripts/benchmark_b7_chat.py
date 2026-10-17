"""项目 B 的固定 Mock `/chat` 延迟验收；只写入并清理本项目临时测试数据。"""

import json
import logging
import math
import shutil
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from skillflow.api import create_app
from skillflow.config import ROOT, Settings


SAMPLES = 30
WARMUP = 3
THRESHOLD_MS = 1500
MESSAGE = "查询订单 A123，结合店铺政策，调查这件商品能否退换。"


def percentile(values: list[float], percent: float) -> float:
    """对已排序样本使用 nearest-rank，避免不同统计工具的插值规则造成歧义。"""
    return values[math.ceil(percent * len(values)) - 1]


def main() -> int:
    """请求完整 `/chat` 响应，单独统计失败数及 P50/P95，不混入其他路由。"""
    temporary_base = (ROOT / ".skillflow").resolve()
    temporary_base.mkdir(exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="b7-chat-benchmark-", dir=temporary_base)).resolve()
    if not temporary_root.is_relative_to(temporary_base):
        raise RuntimeError("基准测试临时目录超出项目 B 范围")

    logger = logging.getLogger("uvicorn.error")
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        # 明确锁定 Mock、demo、B 自有 fixtures；每次使用新会话，避免历史对话改变任务。
        settings = Settings(data_dir=temporary_root / "state", fixtures_dir=ROOT / "data",
                            model_mode="mock", auth_mode="demo")
        app = create_app(settings)
        durations = []
        failures = []
        with TestClient(app) as client:
            for index in range(WARMUP + SAMPLES):
                started = time.perf_counter()
                response = client.post("/chat", json={
                    "session_id": f"b7-chat-benchmark-{index}", "message": MESSAGE,
                })
                # TestClient 返回时完整 JSON 响应已收到；失败和超时也计入耗时及失败数。
                duration_ms = (time.perf_counter() - started) * 1000
                try:
                    outcome = response.json() if response.status_code == 200 else {}
                except ValueError:
                    outcome = {}
                if response.status_code != 200 or outcome.get("status") != "completed":
                    failures.append({"sample": index, "http_status": response.status_code,
                                     "run_status": outcome.get("status")})
                if index >= WARMUP:
                    durations.append(duration_ms)

        durations.sort()
        p95_ms = percentile(durations, 0.95)
        passed = p95_ms <= THRESHOLD_MS and not failures
        # 只输出环境、统计和失败状态；不输出订单内容、会话或凭据。
        print(json.dumps({
            "mode": "mock", "auth_mode": "demo", "endpoint": "/chat",
            "transport": "TestClient in-process", "concurrency": 1,
            "warmup": WARMUP, "samples": SAMPLES,
            "p50_ms": round(percentile(durations, 0.50), 2),
            "p95_ms": round(p95_ms, 2), "max_ms": round(durations[-1], 2),
            "threshold_ms": THRESHOLD_MS, "failure_count": len(failures),
            "failed_samples": failures, "pass": passed,
        }, ensure_ascii=False))
        return 0 if passed else 1
    finally:
        logger.setLevel(previous_level)
        # 删除前再次核对绝对路径，绝不清理生产 SQLite、证书或其他项目目录。
        if temporary_root.is_relative_to(temporary_base) and temporary_root != temporary_base:
            shutil.rmtree(temporary_root)


if __name__ == "__main__":
    raise SystemExit(main())
