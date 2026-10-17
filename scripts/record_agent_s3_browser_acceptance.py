"""导出已经由浏览器触发并人工核对的 S3 运行，不创建请求、不调用模型或改业务数据。"""

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


def main():
    """只读打开 B 的独立演示库；导出完整运行证据及不掩盖失败的汇总。"""
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], help="已核对的 case=run_id，可重复")
    parser.add_argument("--model-label", default="未记录", help="本轮已核对的启动模型名，仅作配置标签")
    parser.add_argument("--context", action="store_true", help="导出短期会话目标补验，保留原 S3 单次验收报告")
    args = parser.parse_args()
    database = (root / ".skillflow/agent-s3-demo/skillflow.sqlite3").resolve()
    output = root / "evaluation/agent_runtime/s3"
    output.mkdir(parents=True, exist_ok=True)
    records, rows = [], []
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        for item in args.run:
            case, run_id = item.split("=", 1)
            raw = db.execute("SELECT value_json FROM kv WHERE name=?", ("run_" + run_id,)).fetchone()
            if raw is None:
                raise SystemExit("指定执行记录不存在：" + run_id)
            record = json.loads(raw[0])
            if record["result"].get("execution_mode") != "real_model":
                raise SystemExit("不是 S3 真实模型记录：" + run_id)
            state_raw = db.execute("SELECT state_json FROM agent_runs WHERE run_id=?", (run_id,)).fetchone()
            state = json.loads(state_raw[0])
            # 配对校验来自实际图消息，不把脚本报告当作模型调用证据。
            calls = [c["call_id"] for m in state["messages"] for c in m.get("tool_calls", [])]
            observations = [m["tool_call_id"] for m in state["messages"] if m["role"] == "tool"]
            paired = sorted(calls) == sorted(observations) and len(calls) == len(set(calls))
            records.append({"case": case, "tool_messages_paired": paired, "response": record,
                            "messages": state["messages"]})
            r = record["result"]
            rows.append(f"| {case} | {record['status']} / {r['stop_reason']} | {r['budget']['model_requests']} | {r['budget']['tool_attempts']} | {r['budget']['active_seconds']:.2f} | {run_id} |")
        tickets = db.execute("SELECT count(*) FROM tickets").fetchone()[0]
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "source": "browser_ui_manual_verification",
              "configured_model_label": args.model_label, "business_data": "project_b_simulated", "tickets_in_demo_db": tickets,
              "capacity_test": False, "records": records}
    # 补验另存文件，不能覆盖原有失败和重测证据。
    report_name = "context-browser-acceptance" if args.context else "browser-acceptance"
    (output / (report_name + ".json")).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    text = ("# S3 短期上下文真实模型网页补验" if args.context else "# S3 真实模型网页验收") + "\n\n2026-10-16，亲自在浏览器提交并检查页面结果，再从 B 自有数据库导出实际执行记录。\n\n"
    text += "模型使用 B 已配置的外部服务；业务数据为固定模拟订单，未重复 Keycloak 验收。不是模型容量测试。\n\n"
    if args.context:
        text += "本组在真实模型配置下检查网页流程；会话策略可以直接澄清，模型请求计数为 0 的回合未请求外部模型。\n\n"
    text += "| 场景 | 状态 / 终止原因 | 模型请求 | 工具尝试 | 有效执行秒数 | run_id |\n|---|---|---:|---:|---:|---|\n" + "\n".join(rows)
    text += f"\n\n演示库工单数量：{tickets}。所有完整运行均核对原生调用 ID 与 ToolMessage 配对。有效执行秒数含模型等待、工具、审计及完成门，不等于并发容量或 P95。\n"
    for row in records:
        record = row["response"]
        text += "\n## " + row["case"] + "\n\n" + record["answer"] + "\n\n"
        text += "Skill：" + json.dumps(record["result"]["loaded_skills"], ensure_ascii=False) + "。\n\n"
        if args.context:
            text += "目标要求：" + json.dumps((record["result"].get("task_goal") or {}).get("requirements"), ensure_ascii=False) + "。\n\n"
            text += "目标来源：" + json.dumps(record["result"].get("conversation_context"), ensure_ascii=False) + "。\n\n"
        text += "实际工具链：" + " → ".join(r["name"] for r in record["result"]["tool_records"]) + "。\n\n"
        text += "工具调用消息配对：" + str(row["tool_messages_paired"]) + "。\n"
    if args.context:
        text += "\n## 补验说明\n\n- 原缺口：完成一单后清空待澄清目标，连续裸单号退化成普通订单查询。现改为持久化白名单目标摘要，继承只替换本轮单号，每笔重新加载 Skill 和取证。\n- 已确认目标在工具前保存，查无订单和需人工不清空；明确新意图切换，歧义/未知话题暂停继承。\n- 刷新和服务重启后的继续是关联新任务，不是 S4 图恢复；过期边界、故障、隔离和 SSE 不续期由注入时钟及模型替身的自动回归验证。\n- 本报告展示真实模型功能补验，不是并发容量或总体成功率评测。\n"
        # 首次模型超时与后续重试分别保留，避免把重试后的通过误写成首轮全通过。
        text += "- 本轮明确切换为订单状态的首次目标识别请求触发 MODEL_TIMEOUT；旧目标已暂停，随后裸单号返回 CONTEXT_PAUSED，未请求模型或工具。完整重述后重试成功，后续单号只查订单。保留首次失败、暂停澄清及重试三类记录，未放宽 30 秒超时或回退脚本。\n"
    else:
        text += "\n## 发现和修正\n\n- A124 首轮最终提案索引/引用校验失败，系统未展示不受信任结论。改用短引用别名并由服务端映射真实结果，增加显式提案审计与最多一次共享修正额度；重测结果与失败记录均保留。\n- A126 首轮最终模型请求触发 MODEL_TIMEOUT，保持单次30秒预算，未回退到脚本或掩盖失败；重测单独记录。\n- 多候选先只列本人订单；刷新后选择 A124，关联新 task/run，重新读取订单、政策、资格和库存。该行为不属于 S4 执行图恢复。\n- 展开政策工具核对 evidence.ref、原文 document_id 与后续 assess_after_sales 参数；展开质量工具核对事件版本、三态和确定性原因。\n"
    (output / (report_name + ".md")).write_text(text, encoding="utf-8")
    print(str(output / (report_name + ".md")))


if __name__ == "__main__":
    main()
