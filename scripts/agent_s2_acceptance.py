"""S2 独立验收：运行安全用例和实际本地工具，生成可读 Markdown 与 JSON。"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import unittest
from uuid import uuid4


ROOT = Path(__file__).resolve().parent.parent


async def business_demo(root):
    """脚本显式调用工具组件，不用 LLM，不通过旧 Agent 固定计划冒充新循环。"""
    from evaluation.agent_runtime.s2_fixtures import seed_quality_demo
    from skillflow.agent.bootstrap import create_tool_runtime
    from skillflow.agent.contracts import TaskMetadata, ToolCall
    from skillflow.agent.tool_executor import TargetScope
    from skillflow.business.policies import LocalPolicyRepository
    from skillflow.business.repository import MockOrderRepository
    from skillflow.business.surge import BusinessRepository
    from skillflow.sqlite_store import SqliteStore

    store = SqliteStore(root)
    policies = LocalPolicyRepository(ROOT / "data/policies")
    runtime = create_tool_runtime(store, MockOrderRepository(ROOT / "data"), policies)
    traces, task_records = [], []

    async def new_task(owner_runtime=runtime, order_id="A123", scope=None):
        """模拟已认证应用层登记；任务身份不出现在公开工具参数中。"""
        metadata = TaskMetadata(task_id=uuid4().hex, run_id=str(uuid4()), buyer_id="demo-buyer",
            session_id=uuid4().hex, engine="tool_loop", engine_version="s2_tools_v1",
            graph_version="s2_diagnostic_v1", model_mode="scripted")
        store.register_run(metadata.run_id, metadata.buyer_id, metadata.session_id, metadata.model_dump(mode="json"))
        task = await owner_runtime.open_task(metadata.run_id, metadata.buyer_id, metadata.session_id,
                                              scope or TargetScope(explicit_order_ids=(order_id,)))
        task_records.append((owner_runtime, task))
        return task

    async def call(task, tool_name, **arguments):
        """每次工具面代表下一轮；当前报告是脚本驱动，不代表模型自主选工具。"""
        observation = await task.dispatch(ToolCall(call_id=uuid4().hex, name=tool_name, arguments=arguments),
                                           task.begin_turn(uuid4().hex))
        traces.append({"task_id": task.metadata.task_id, "tool": tool_name, "observation": observation.model_dump(mode="json")})
        return observation

    async def load(task, skill_name):
        return await call(task, "load_skill", name=skill_name, reason="S2 固定场景验收")

    order_cases = []
    for order_id in ("A123", "A124", "A125", "A126"):
        task = await new_task(order_id=order_id)
        skill = await load(task, "return_exchange")
        order = await call(task, "get_order", order_id=order_id)
        policy = await call(task, "retrieve_policy", order_id=order_id, query="退换货资格 店铺政策")
        row = {"order_id": order_id, "order_status": order.status, "skill_version": skill.data.get("version"),
               "skill_hash": skill.data.get("content_hash"), "return_eligible": None, "exchange_eligible": None,
               "inventory": None, "error_code": policy.error_code, "task_id": task.metadata.task_id}
        if policy.status == "success":
            refs = [item["ref"] for item in policy.data["evidence"]]
            assessed = await call(task, "assess_after_sales", order_id=order_id, evidence_refs=refs)
            inventory = await call(task, "check_inventory", order_id=order_id)
            row.update(return_eligible=assessed.data.get("return_eligible"), exchange_eligible=assessed.data.get("exchange_eligible"),
                       inventory=inventory.data.get("available"), error_code=assessed.error_code or inventory.error_code)
        order_cases.append(row)

    # 每种攻击使用清晰的可信任务上下文，报告展示实际拒绝码。
    rejected = []
    task = await new_task()
    unloaded = await call(task, "get_order", order_id="A123")
    rejected.append({"case": "未加载 Skill", "expected": "SKILL_NOT_LOADED", "actual": unloaded.error_code})
    await load(task, "return_exchange")
    for label, tool_name, arguments, expected in (
        ("注入买家身份", "get_order", {"order_id": "A123", "buyer_id": "other-buyer"}, "INVALID_TOOL_ARGUMENTS"),
        ("跨买家订单", "get_order", {"order_id": "A900"}, "ORDER_FORBIDDEN"),
        ("超出任务目标", "get_order", {"order_id": "A124"}, "TARGET_NOT_ALLOWED"),
        ("未知工具", "shell", {"command": "dir"}, "UNKNOWN_TOOL"),
        ("提前创建草稿", "create_ticket_draft", {"order_id": "A123", "request_type": "return", "reason": "申请"}, "WRITE_NOT_ENABLED"),
        ("提前提交", "submit_ticket", {"ticket_id": "fake"}, "WRITE_NOT_ENABLED"),
    ):
        observation = await call(task, tool_name, **arguments)
        rejected.append({"case": label, "expected": expected, "actual": observation.error_code})

    discoveries = []
    for keyword in ("手柄", "耳机"):
        task = await new_task(scope=TargetScope(keyword=keyword))
        await load(task, "order_tracking")
        page = await call(task, "list_my_orders", keyword=keyword)
        target = page.data["items"][0]["order_id"]
        result = await call(task, "get_order", order_id=target)
        discoveries.append({"keyword": keyword, **page.data, "followup_status": result.status, "followup_code": result.error_code})

    sql = BusinessRepository(store.database_path, ROOT / "data")
    try:
        seed_quality_demo(sql)
        quality_runtime = create_tool_runtime(store, sql, policies)
        task = await new_task(quality_runtime, scope=TargetScope(keyword="S2 模拟充电宝", quality_batch=True))
        await load(task, "quality_incident")
        page = await call(task, "list_my_orders", keyword="S2 模拟充电宝")
        incident_cases = []
        for item in page.data["items"]:
            await call(task, "get_order", order_id=item["order_id"])
            observation = await call(task, "match_quality_incident", order_id=item["order_id"])
            incident_cases.append({"status": observation.status, **observation.data})
        alternatives = await call(task, "list_alternatives", order_id="A700")

        tasks = []
        for owner_runtime, task in task_records:
            await task.finish_diagnostic()
            tasks.append({"metadata": task.metadata.model_dump(mode="json"), "scope": task.scope.model_dump(mode="json"),
                          "skills": {name: doc.model_dump() for name, doc in task.loaded.items()},
                          "opened_tool_schemas": [spec.model_dump() for spec in task.offered_tools()],
                          "budget": task.budget.snapshot().model_dump(), "events": store.agent_events(task.metadata.run_id)})
            owner_runtime.close_task(task.metadata.run_id)
    finally:
        sql.engine.dispose()
    valid = (
        order_cases[0]["return_eligible"] is True and order_cases[0]["inventory"] > 0
        and order_cases[1]["exchange_eligible"] is True and order_cases[1]["inventory"] == 0
        and order_cases[2]["error_code"] == "POLICY_NOT_REGISTERED"
        and order_cases[3]["return_eligible"] is False
        and all(case["expected"] == case["actual"] for case in rejected)
        and discoveries[0]["followup_status"] == "success"
        and discoveries[1]["followup_code"] == "CLARIFICATION_REQUIRED"
        and [case.get("decision") for case in incident_cases] == ["AFFECTED", "NOT_AFFECTED", "UNVERIFIED", "AFFECTED"]
        and alternatives.status == "success" and bool(alternatives.data.get("items"))
    )
    return {"validated": valid, "database": str(root.relative_to(ROOT) / "skillflow.sqlite3"),
            "catalog": await runtime.skills.list_catalog(), "order_cases": order_cases, "rejections": rejected,
            "discovery_cases": discoveries, "incident_cases": incident_cases,
            "alternatives": alternatives.model_dump(mode="json"), "tasks": tasks, "tool_trace": traces}


def markdown_report(report):
    """将实际结果汇总成业务可读表；完整 SOP、Schema 和事件留在配套 JSON。"""
    yes = lambda value: "无法判断" if value is None else "是" if value else "否"
    lines = ["# S2 技能与工具边界验收报告", "", "**脚本驱动 · 模拟订单 · 实际执行本地业务工具 · 未调用 LLM**", "",
             "本报告验收 Skill SOP、加载范围、参数和权限边界。聊天页面仍是 S1 骨架，模型自主调用留到 S3。", "",
             f"- 记录时间：{report['recorded_at']}",
             f"- 自动测试：{report['tests']['tests_run']} 项；{'全部通过' if report['tests']['successful'] else '未全部通过'}",
             f"- 本地业务场景：{'通过' if report['demo']['validated'] else '失败'}", "",
             "## 1. 退换调查：实际工具结果", "", "| 订单 | 退货资格 | 换货资格 | 换货库存 | 错误码 |", "| --- | --- | --- | --- | --- |"]
    for row in report["demo"]["order_cases"]:
        stock = row["inventory"] if row["inventory"] is not None else "未查询"
        lines.append(f"| {row['order_id']} | {yes(row['return_eligible'])} | {yes(row['exchange_eligible'])} | {stock} | {row['error_code'] or '无'} |")
    lines.extend(["", "A124 的资格与库存分别判断；缺货不改变退货资格。A125 缺政策，停止资格结论；A126 保留超期结果。未创建或提交工单。", "",
                  "## 2. 技能目录与版本", "", "| Skill | 版本 | 内容哈希（前 12 位） |", "| --- | --- | --- |"])
    for row in report["demo"]["catalog"]:
        lines.append(f"| {row['name']} | {row['version']} | {row['content_hash'][:12]} |")
    lines.extend(["", "首次只开放 load_skill；规范加载后，下一轮才能使用新工具。写入工具禁用，物流工具待 S5 MCP 接入。", "",
                  "## 3. 实际拦截", "", "| 案例 | 预期 | 实际 |", "| --- | --- | --- |"])
    for row in report["demo"]["rejections"]:
        lines.append(f"| {row['case']} | {row['expected']} | {row['actual']} |")
    lines.extend(["", "## 4. 本人订单候选发现", "", "| 关键词 | 候选数 | 是否完整扫描 | 后续实际结果 |", "| --- | --- | --- | --- |"])
    for row in report["demo"]["discovery_cases"]:
        lines.append(f"| {row['keyword']} | {row['candidate_count']} | {row['discovery_complete']} | {row['followup_code'] or row['followup_status']} |")
    lines.extend(["", "## 5. SQLAlchemy 确定性质量事件工具", "", "| 订单 | 三态判断 | 事件售后处理 | 原因码 | 事件版本 |", "| --- | --- | --- | --- | --- |"])
    for row in report["demo"]["incident_cases"]:
        # 待核实/需人工复核不能渲染为“无退货资格”，原始布尔值仍完整保留在 JSON。
        outcome = ("待核实" if row["decision"] == "UNVERIFIED" else "需人工复核" if row["manual_review"]
                   else "符合事件退货条件" if row["return_eligible"] else "不在事件影响范围")
        lines.append(f"| {row['order_id']} | {row['decision']} | {outcome} | {row['reason_code']} | {row['event_version'] or '—'} |")
    lines.extend(["", "### 实际替代品结果", "", "| SKU | 商品 | 可用库存 |", "| --- | --- | --- |"])
    for item in report["demo"]["alternatives"]["data"].get("items", []):
        lines.append(f"| {item['sku']} | {item['name']} | {item['available']} |")
    lines.extend(["", "UNVERIFIED 表示待核实。替代品只是本地目录中同类、在售、有库存且未被当前生效事件排除的候选，不作安全保证。", "",
                  "## 6. 证据与阶段边界", "", "[完整 JSON（SOP、公开 Schema、工具结果、事件）](latest.json)", "",
                  f"隔离诊断数据库：{report['demo']['database']}", "",
                  "自动测试另覆盖跨任务证据、政策变更、跳页、超时、预算、重复调用及任务隔离。快照用于诊断，尚不能恢复执行图；本报告不证明真实模型能力或并发容量。", ""])
    return "\n".join(lines)


def main():
    """冻结当前测试进程配置，不改用户 .env；输出范围严格限制在项目 B。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-regression", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation/agent_runtime/s2")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("报告目录必须位于项目 B 内")
    os.environ.update({"SKILLFLOW_AUTH_MODE": "demo", "SKILLFLOW_MODEL_MODE": "mock",
        "SKILLFLOW_AGENT_ENGINE": "legacy", "SKILLFLOW_AGENT_MODEL_MODE": "disabled",
        "SKILLFLOW_DEMO_BUYER_ID": "demo-buyer", "SKILLFLOW_BUSINESS_BACKEND": "mock",
        "SKILLFLOW_CHAT_LOCK_MODE": "session", "SKILLFLOW_MAX_ACTIVE_CHATS": "8", "SKILLFLOW_MAX_WAITING_CHATS": "64",
        "SKILLFLOW_AGENT_MAX_ROUNDS": "8", "SKILLFLOW_AGENT_MAX_TOOL_ATTEMPTS": "12",
        "SKILLFLOW_AGENT_MAX_MODEL_REQUESTS": "10", "SKILLFLOW_AGENT_MODEL_TIMEOUT": "30", "SKILLFLOW_AGENT_ACTIVE_TIMEOUT": "120"})
    sys.path.insert(0, str(ROOT))
    modules = ["tests_skillflow.test_agent_skills"]
    if args.include_regression:
        modules.extend(["tests_skillflow.test_agent_foundation", "tests_skillflow.test_agent",
                        "tests_skillflow.test_streaming", "tests_skillflow.test_surge"])
    started = perf_counter()
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    demo = asyncio.run(business_demo(ROOT / ".skillflow/agent-s2-tests" / ("report-" + uuid4().hex)))
    report = {"stage": "S2", "recorded_at": datetime.now(timezone.utc).isoformat(),
        "execution_mode": "scripted_driver_local_business_tools", "real_model_called": False,
        "tests": {"scope": modules, "tests_run": result.testsRun, "successful": result.wasSuccessful(),
                  "failures": [test.id() for test, _ in result.failures], "errors": [test.id() for test, _ in result.errors],
                  "skipped": len(result.skipped)}, "elapsed_seconds": round(perf_counter() - started, 3), "demo": demo}
    output.mkdir(parents=True, exist_ok=True)
    (output / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "latest.md").write_text(markdown_report(report), encoding="utf-8")
    print("S2_REPORT=" + str(output / "latest.md"))
    return 0 if result.wasSuccessful() and demo["validated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
