"""项目 B 的 50 稳态／200 突发质量事件验收。

默认在 B 的独立目录生成固定 1 万订单和 1000 买家，分别运行全局锁基线与会话锁版本。
模型模式固定 Mock；本脚本使用同一 ASGI 应用内的模拟 OIDC 登录，不触碰现有 Alice/Bob 库。
完整验收用 --duration 600；短跑只能验证脚本和安全路径，不能声称十分钟指标通过。
"""

import argparse
import asyncio
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import httpx

from skillflow.auth import CSRF_COOKIE
from skillflow.business.surge import BusinessRepository
from skillflow.config import ROOT, Settings
from skillflow.sqlite_store import SqliteStore
from scripts.surge_admin import seed


class LocalIdentityProvider:
    """仅验收用的本机身份提供者；不发网络请求，不模拟真实模型。"""

    async def authorization_url(self, state, nonce, challenge):
        return "https://identity.test/authorize?state=" + state

    async def exchange_code(self, code, verifier, nonce):
        return {"subject": code}

    async def end_session_url(self):
        return "https://identity.test/logout"


def percentile(values, fraction):
    """对完整响应与排队样本使用最近秩法，空样本返回 None。"""
    ordered = sorted(values)
    return round(ordered[max(0, int(len(ordered) * fraction + 0.999999) - 1)], 2) if ordered else None


def peak_rss_bytes():
    """Windows 读取进程峰值工作集；其他系统尽量使用 ru_maxrss。"""
    if sys.platform == "win32":
        import ctypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        # Windows 64 位进程句柄必须按指针传递；ctypes 默认 int 会截断句柄并导致指标为 None。
        current_process = ctypes.windll.kernel32.GetCurrentProcess
        current_process.restype = ctypes.c_void_p
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        get_memory.restype = ctypes.c_int
        ok = get_memory(current_process(), ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize) if ok else None
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except ImportError:
        return None


def parse_final(sse_text):
    """只把持久化 final 帧算作成功，error/interrupted 不伪装成业务完成。"""
    for block in reversed(sse_text.replace("\r\n", "\n").split("\n\n")):
        lines = block.splitlines()
        if "event: final" in lines:
            data = next((line[6:] for line in lines if line.startswith("data: ")), None)
            return json.loads(data) if data else None
    return None


def prepare_database(path: Path):
    """两个模式各用一份同种子独立库；保留结果供复核，不修改用户现有库。"""
    if path.exists():
        raise ValueError(f"目标隔离目录已存在：{path}")
    store = SqliteStore(path)
    repository = BusinessRepository(store.database_path)
    seed(repository, 10000, 1000)
    repository.change_incident("SIM-POWERBANK-001", actor="acceptance-reviewer",
                               action="publish", reason="固定模拟证据审核生效")
    repository.engine.dispose()


async def login(client, subject):
    """通过应用 OIDC 路由取得真实本地会话与 CSRF，不直接伪造买家请求头。"""
    begin = await client.get("/auth/login", follow_redirects=False)
    state = parse_qs(urlparse(begin.headers["location"]).query)["state"][0]
    callback = await client.post("/auth/callback", data={"state": state, "code": subject},
                                 follow_redirects=False)
    if callback.status_code != 303:
        raise RuntimeError(f"模拟登录失败：{callback.status_code}")
    headers = {"X-CSRF-Token": client.cookies.get(CSRF_COOKIE)}
    session = await client.post("/sessions", headers=headers)
    session.raise_for_status()
    return headers, session.json()["session_id"]


def expected_decision(index):
    return {0: "AFFECTED", 1: "NOT_AFFECTED", 2: "UNVERIFIED", 3: "UNVERIFIED", 4: "AFFECTED"}[index % 5]


async def journey(client, index, headers, session_id, repeat=0, reconnect=False):
    """60% 批次调查、20% 替代品、20% 退货草稿与确认；记录完整用户旅程。"""
    order_id = f"A{100000+index}"
    case = index % 5
    message = f"查询订单 {order_id} 的批次"
    if case == 1:
        message += "，有没有替代品"
    elif case == 0:
        message += "，能退货吗"
    run_id = str(uuid4())
    job = {"run_id": run_id, "session_id": session_id, "message": message}
    started = time.perf_counter()
    response = await client.post("/chat/stream", json=job, headers=headers)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if response.status_code == 429:
        return {"accepted": False, "busy": response.json().get("detail", {}).get("code") == "CHAT_BUSY",
                "elapsed_ms": elapsed_ms}
    if response.status_code != 200:
        return {"accepted": False, "error": f"HTTP_{response.status_code}", "elapsed_ms": elapsed_ms}
    final = parse_final(response.text)
    if not final:
        return {"accepted": True, "completed": False, "run_id": run_id, "error": "NO_FINAL",
                "elapsed_ms": elapsed_ms}
    outcome = final.get("result", {}).get("incident", {})
    correct = outcome.get("decision") == expected_decision(index)
    if case == 1:
        correct = correct and bool(final.get("result", {}).get("alternatives"))
    if case == 0 and correct:
        draft = await client.post("/tickets/draft", json={"session_id": session_id,
                                  "order_id": order_id, "request_type": "return",
                                  "reason": "模拟质量事件退货"}, headers=headers)
        if draft.status_code != 200 or not draft.json().get("ticket"):
            correct = False
        else:
            ticket_id = draft.json()["ticket"]["ticket_id"]
            confirmed = await client.post(f"/tickets/{ticket_id}/confirm", json={
                "session_id": session_id, "confirmed": True}, headers=headers)
            correct = (confirmed.status_code == 200 and
                       confirmed.json().get("status") in {"submitted", "already_submitted"})
            if repeat == 0:
                # 重复点击是同一业务请求，不应创建第二张工单或正式确认事件。
                second = await client.post(f"/tickets/{ticket_id}/confirm", json={
                    "session_id": session_id, "confirmed": True}, headers=headers)
                correct = correct and second.json().get("status") == "already_submitted"
    if reconnect:
        replay = await client.get(f"/runs/{run_id}/stream", params={"session_id": session_id})
        correct = correct and replay.status_code == 200 and parse_final(replay.text) == final
    return {"accepted": True, "completed": correct, "run_id": run_id,
            "error": None if correct else "UNEXPECTED_BUSINESS_RESULT", "elapsed_ms": elapsed_ms}


def run_timing(store, run_id):
    """排队与首进度从持久化任务／帧时间计算，不把响应头时间当完整 SSE。"""
    with closing(store.connect()) as conn:
        row = conn.execute("SELECT queued_at, started_at FROM run_streams WHERE run_id=?", (run_id,)).fetchone()
        first = conn.execute("SELECT MIN(created_at) AS first_at FROM stream_frames "
                             "WHERE run_id=? AND kind<>'started'", (run_id,)).fetchone()
    if not row:
        return None, None
    queue_ms = ((row["started_at"] - row["queued_at"]) * 1000
                if row["started_at"] is not None else None)
    first_ms = ((first["first_at"] - row["queued_at"]) * 1000
                if first and first["first_at"] else None)
    return queue_ms, first_ms


def ticket_integrity(store):
    """从持久库核对有效工单和正式确认事件，避免只凭 HTTP 返回判断幂等。"""
    with closing(store.connect()) as conn:
        duplicate_tickets = conn.execute("""
            SELECT COUNT(*) FROM (
                SELECT buyer_id, order_id, request_type FROM tickets
                WHERE status IN ('draft', 'submitted')
                GROUP BY buyer_id, order_id, request_type HAVING COUNT(*) > 1
            )
        """).fetchone()[0]
        duplicate_confirms = conn.execute("""
            SELECT COUNT(*) FROM (
                SELECT ticket_id FROM ticket_events WHERE action='confirm' AND result='submitted'
                GROUP BY ticket_id HAVING COUNT(*) > 1
            )
        """).fetchone()[0]
        submitted_without_confirm = conn.execute("""
            SELECT COUNT(*) FROM tickets AS t WHERE t.status='submitted' AND NOT EXISTS (
                SELECT 1 FROM ticket_events AS e WHERE e.ticket_id=t.ticket_id
                    AND e.action='confirm' AND e.result='submitted'
            )
        """).fetchone()[0]
        submitted = conn.execute("SELECT COUNT(*) FROM tickets WHERE status='submitted'").fetchone()[0]
    return {"submitted": submitted, "duplicate_active_tickets": duplicate_tickets,
            "duplicate_confirm_events": duplicate_confirms,
            "submitted_without_confirm": submitted_without_confirm}


async def exercise_mode(root: Path, lock_mode: str, concurrency: int, duration: float,
                        think_time: float, burst: int):
    """同一 ASGI 应用内运行多买家会话；全局锁基线与会话锁只差锁配置。"""
    from skillflow.api import create_app

    settings = Settings(data_dir=root, fixtures_dir=ROOT / "data", model_mode="mock",
                        auth_mode="production", business_backend="sqlalchemy", chat_lock_mode=lock_mode,
                        oidc_issuer="https://identity.test", oidc_client_id="surge-acceptance",
                        oidc_client_secret="local-test-secret", oidc_redirect_uri="https://testserver/auth/callback")
    app = create_app(settings, identity_provider=LocalIdentityProvider())
    store = app.state.store
    total_buyers = max(concurrency, burst)
    for index in range(total_buyers):
        store.provision_identity(settings.oidc_issuer, f"subject-{index}", f"surge-buyer-{index:04d}")
    transport = httpx.ASGITransport(app=app)
    clients = [httpx.AsyncClient(transport=transport, base_url="https://testserver", timeout=60,
                                 follow_redirects=False, trust_env=False) for _ in range(total_buyers)]
    try:
        accounts = await asyncio.gather(*(login(clients[i], f"subject-{i}") for i in range(total_buyers)))
        store.write_wait_ms.clear()
        store.busy_errors = 0
        started = time.perf_counter()
        deadline = started + duration

        async def worker(index):
            samples = []
            repeat = 0
            while repeat == 0 or time.perf_counter() < deadline:
                headers, session_id = accounts[index]
                samples.append(await journey(clients[index], index, headers, session_id,
                                             repeat, reconnect=repeat == 0 and index % 10 == 0))
                repeat += 1
                if think_time:
                    await asyncio.sleep(think_time)
            return samples

        groups = await asyncio.gather(*(worker(index) for index in range(concurrency)))
        steady_samples = [item for group in groups for item in group]
        actual_duration = time.perf_counter() - started
        burst_samples = []
        burst_elapsed = None
        if burst:
            burst_started = time.perf_counter()
            burst_samples = await asyncio.gather(*(journey(clients[index], index, *accounts[index],
                                                          reconnect=index % 20 == 0)
                                                   for index in range(burst)))
            burst_elapsed = time.perf_counter() - burst_started

        # 使用已登录的第一位买家访问第二位买家的资源；所有拒绝都必须带稳定错误码。
        isolation = {}
        if total_buyers >= 2:
            foreign_order = await clients[0].get("/orders/A100001")
            isolation["order"] = (foreign_order.status_code == 403 and
                                   foreign_order.json().get("detail", {}).get("code") == "ORDER_FORBIDDEN")
            other_run = groups[1][0].get("run_id") if concurrency >= 2 else None
            if other_run:
                foreign_run = await clients[0].get(f"/runs/{other_run}")
                isolation["run"] = (foreign_run.status_code == 403 and
                                    foreign_run.json().get("detail", {}).get("code") == "RUN_FORBIDDEN")
            with closing(store.connect()) as conn:
                row = conn.execute("SELECT ticket_id FROM tickets WHERE buyer_id='surge-buyer-0005' "
                                   "ORDER BY created_at DESC LIMIT 1").fetchone()
            if row:
                foreign_ticket = await clients[0].get(f"/tickets/{row['ticket_id']}",
                                                      params={"session_id": accounts[0][1]})
                isolation["ticket"] = (foreign_ticket.status_code == 403 and
                                       foreign_ticket.json().get("detail", {}).get("code") == "TICKET_FORBIDDEN")

        queue, first = [], []
        for item in [*steady_samples, *burst_samples]:
            if item.get("run_id"):
                wait_ms, first_ms = run_timing(store, item["run_id"])
                if wait_ms is not None:
                    queue.append(wait_ms)
                if first_ms is not None:
                    first.append(first_ms)
        accepted = [item for item in steady_samples if item.get("accepted")]
        completed = [item for item in accepted if item.get("completed")]
        burst_accepted = [item for item in burst_samples if item.get("accepted")]
        report = {
            "lock_mode": lock_mode, "model_mode": "mock", "business_route": "incident_deterministic",
            "steady_seconds": round(actual_duration, 2), "concurrent_sessions": concurrency,
            "accepted": len(accepted), "completed_expected": len(completed),
            "completion_rate": round(len(completed) / len(accepted), 4) if accepted else None,
            "busy_rejected": sum(item.get("busy", False) for item in steady_samples),
            "unexpected_errors": sorted({item.get("error") for item in steady_samples if item.get("error")}),
            "throughput_per_second": round(len(completed) / actual_duration, 2),
            "complete_response_ms": {key: percentile([item["elapsed_ms"] for item in accepted], quantile)
                                     for key, quantile in (("p50", .5), ("p95", .95), ("p99", .99))},
            "queue_p95_ms": percentile(queue, .95), "sse_first_progress_p95_ms": percentile(first, .95),
            "sqlite_write_wait_p95_ms": percentile(list(store.write_wait_ms), .95),
            "sqlite_busy_errors": store.busy_errors,
            "peak_rss_bytes": peak_rss_bytes(), "admission": app.state.chat_admission.snapshot(),
            "ticket_integrity": ticket_integrity(store), "isolation_probes": isolation,
            "burst": {"sessions": burst, "elapsed_seconds": round(burst_elapsed, 2) if burst_elapsed else None,
                      "accepted": len(burst_accepted),
                      "completed_expected": sum(item.get("completed", False) for item in burst_accepted),
                      "busy_rejected": sum(item.get("busy", False) for item in burst_samples),
                      "unexpected_errors": sorted({item.get("error") for item in burst_samples if item.get("error")})},
        }
        return report
    finally:
        await asyncio.gather(*(client.aclose() for client in clients))
        app.state.repository.engine.dispose()


async def main():
    parser = argparse.ArgumentParser(description="项目 B 质量事件并发验收")
    parser.add_argument("--duration", type=float, default=600, help="50 并发稳态秒数，正式验收需 600")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--burst", type=int, default=200)
    parser.add_argument("--think-time", type=float, default=0.2)
    parser.add_argument("--skip-baseline", action="store_true", help="短跑排障时可跳过全局锁基线")
    args = parser.parse_args()
    if args.duration <= 0 or not 1 <= args.concurrency <= 1000 or not 0 <= args.burst <= 1000:
        parser.error("时长和并发人数不在允许范围")
    base = (ROOT / ".skillflow" / "surge-tests" / ("accept-" + uuid4().hex)).resolve()
    base.mkdir(parents=True)
    modes = ["session"] if args.skip_baseline else ["global", "session"]
    report = {"scenario": "simulated_power_bank_incident", "isolated_data_dir": str(base),
              "formal_duration_reached": args.duration >= 600 and args.concurrency == 50 and args.burst == 200,
              "modes": {}}
    for mode in modes:
        data_dir = base / mode
        prepare_database(data_dir)
        print(f"开始 {mode}：{args.concurrency} 并发，{args.duration} 秒；随后 {args.burst} 突发。", flush=True)
        result = await exercise_mode(data_dir, mode, args.concurrency, args.duration,
                                     args.think_time, args.burst if mode == "session" else 0)
        report["modes"][mode] = result
        (base / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not args.skip_baseline:
        baseline = report["modes"]["global"]["complete_response_ms"]["p95"]
        improved = report["modes"]["session"]["complete_response_ms"]["p95"]
        report["p95_ratio_session_to_global"] = round(improved / baseline, 3) if baseline else None
    (base / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"验收报告：{base / 'report.json'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
