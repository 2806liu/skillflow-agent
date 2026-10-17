import asyncio
import logging
import re
from time import perf_counter
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from skillflow.contracts import Evidence, Plan, ToolResult, TraceEvent
from skillflow.planner import validate_plan
from skillflow.streaming import publish, publish_trace

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    """一次任务的共享状态：请求、计划、工具事实、证据、回答和执行事件。"""
    message: str
    context: dict
    plan: Plan | None
    events: list[TraceEvent]
    results: dict[str, Any]
    citations: list[Evidence]
    answer: str
    status: str
    stopped: ToolResult | None
    # 服务端从对话决策编译的计划仍经过 validate，不接受客户端传入此字段。
    supplied_plan: Plan
    skip_llm_answer: bool


class AgentRuntime:
    """LangGraph 控制业务步骤；各节点只依赖项目 B 注入的模块。"""
    def __init__(self, planner, skills, tools, timeout):
        self.plan_service, self.skills, self.tools, self.timeout = planner, skills, tools, timeout
        builder = StateGraph(AgentState)
        # 规划失败直接进入回答；计划合法且可执行时才进入工具节点。
        for name in ("planner", "validate", "execute", "synthesize", "llm_answer", "critic"):
            builder.add_node(name, getattr(self, name))
        builder.add_edge(START, "planner")
        builder.add_conditional_edges("planner", lambda s: "validate" if s.get("plan") else "synthesize")
        builder.add_conditional_edges("validate", lambda s: "execute" if s["status"] == "ready" else "synthesize")
        builder.add_edge("execute", "synthesize")
        builder.add_edge("synthesize", "llm_answer")
        builder.add_edge("llm_answer", "critic")
        builder.add_edge("critic", END)
        self.graph = builder.compile()

    @staticmethod
    def event(state, node, status, summary, **kwargs):
        event = TraceEvent(node=node, status=status, summary=summary, **kwargs)
        publish_trace(event)
        return [*state.get("events", []), event]

    async def planner(self, state):
        started = perf_counter()
        try:
            plan = state.get("supplied_plan") or await asyncio.wait_for(
                self.plan_service.create(state["message"], state["context"]),
                getattr(getattr(self.plan_service, "settings", None), "model_timeout", self.timeout))
            # 计划先于工具事件发布；计划不含回答正文，仍须通过后续校验。
            publish("plan", plan.model_dump())
            return {"plan": plan, "status": plan.decision,
                    "events": self.event(state, "planner", "success", "已生成结构化计划。", duration_ms=round((perf_counter()-started)*1000, 2))}
        except Exception:
            logger.exception("planner_failed")
            return {"plan": None, "status": "planner_failed", "answer": "计划生成失败。请检查模型配置或稍后重试，尚未执行工具。",
                    "events": self.event(state, "planner", "failed", "模型或计划结构异常，未自动切换 Mock。")}

    async def validate(self, state):
        plan = state["plan"]
        if plan.decision != "execute":
            return {"answer": plan.message, "events": self.event(state, "validate", "skipped", "该请求需要补充信息或超出当前能力。")}
        try:
            validate_plan(plan, state["message"], state["context"], self.skills)
            return {"status": "ready", "events": self.event(state, "validate", "success", "订单来源、工具白名单、参数和步骤依赖校验通过。")}
        except ValueError as exc:
            return {"status": "plan_rejected", "answer": "计划未通过校验，尚未执行工具：" + str(exc),
                    "events": self.event(state, "validate", "failed", str(exc))}

    async def execute(self, state):
        """按计划依赖执行本项目工具，遇到空结果或失败时跳过剩余步骤。

        中文数据流说明：
        LangGraph execute 节点 → TypedCoreTool.call() → LangChain StructuredTool.ainvoke()
        → 项目 B 的业务函数 → ToolResult → 写回 LangGraph 状态。
        execute 只负责按计划推进，不让模型直接伪造订单、政策或库存事实。
        """
        results, events, citations = {}, list(state["events"]), []
        completed = set()
        plan = state["plan"]
        for step in plan.steps:
            if not set(step.depends_on) <= completed:
                return {"status": "execution_failed", "answer": "执行依赖未完成，调查已停止。", "results": results, "events": events}
            arguments = {"order_id": plan.order_id}
            # 检索词和证据绑定实际前置结果，不信任 Planner 伪造的业务参数。
            if step.tool == "retrieve_policy":
                if plan.skill == "coupon_dispute":
                    keywords = "优惠 折扣 优惠券 实付 原价 退款"
                elif plan.skill == "price_adjustment":
                    keywords = "购买价 当前价 降价 补差价 价保 价格保护 期限"
                else:
                    keywords = "质量问题 退货 换货 七天 条件 政策"
                arguments["query"] = f"{results['get_order']['order']['product']} {results['get_order']['order']['policy_key']} {keywords}"
            elif step.tool == "analyze_discount":
                # B3 金额核对同样只能使用政策检索节点产生的证据。
                arguments["evidence"] = [hit.model_dump() for hit in citations]
            elif step.tool == "analyze_price_adjustment":
                # 价差工具只能读取订单和政策检索结果，不能接受模型计算的价格。
                arguments["evidence"] = [hit.model_dump() for hit in citations]
            elif step.tool == "assess_after_sales":
                arguments["evidence"] = [hit.model_dump() for hit in citations]
            # 这里是 LangGraph 到工具适配层的入口：
            # self.tools[step.tool] 是 TypedCoreTool，call() 内部会继续调用
            # LangChain StructuredTool.ainvoke()，而不是直接执行模型返回的工具名。
            started = perf_counter()
            result = None
            for attempt in range(2):
                # 工具开始事件只含工具标识，不提前泄漏未经核验的业务结论。
                publish("tool_start", {"tool": step.tool, "step_id": step.step_id, "attempt": attempt + 1})
                # 仅对标记为可重试的只读失败重试一次，避免无限循环。
                try:
                    # call() 最终返回经过 Pydantic 校验的 ToolResult，
                    # execute 再把其中的 data、状态和错误写入 LangGraph 的 state。
                    result = await asyncio.wait_for(self.tools[step.tool].call(arguments), self.timeout)
                except asyncio.TimeoutError:
                    result = ToolResult(status="failed", error_code="TOOL_TIMEOUT", message="工具执行超时。", retryable=True)
                except Exception:
                    logger.exception("tool_failed tool=%s", step.tool)
                    result = ToolResult(status="failed", error_code="TOOL_FAILED", message="工具执行失败，请稍后重试。")
                events.append(TraceEvent(node="execute", status=result.status, summary=result.message or "工具已返回实际结果。",
                                         tool=step.tool, step_id=step.step_id,
                                         duration_ms=round((perf_counter()-started)*1000, 2),
                                         details={"attempt": attempt+1, "input": arguments, "output": result.model_dump()}))
                publish_trace(events[-1])
                if result.status != "failed" or not result.retryable or attempt == 1:
                    break
            results[step.tool] = result.data
            if step.tool == "retrieve_policy":
                citations = [Evidence.model_validate(hit) for hit in result.data.get("evidence", [])]
            if result.status != "success":
                remaining = plan.steps[plan.steps.index(step)+1:]
                for skipped in remaining:
                    events.append(TraceEvent(node="execute", status="skipped", summary="前置工具未成功，未执行此步骤。", tool=skipped.tool, step_id=skipped.step_id))
                    publish_trace(events[-1])
                return {"status": "tool_failed" if result.status == "failed" else "insufficient_evidence" if step.tool != "get_order" else "order_not_found",
                        "stopped": result, "results": results, "citations": citations, "events": events}
            completed.add(step.step_id)
        return {"status": "completed", "results": results, "citations": citations, "events": events}

    async def synthesize(self, state):
        """用实际订单、政策规则和库存组装回答，不把计划本身当作执行事实。"""
        answer = state.get("answer", "")
        order = state.get("results", {}).get("get_order", {}).get("order")
        if state.get("stopped"):
            answer = state["stopped"].message
            if order:
                if state.get("plan") and state["plan"].skill == "coupon_dispute":
                    reason = answer.replace("无法判断退换条件。", "无法判断优惠金额和退款依据。")
                    answer = f"订单 {order['order_id']}：暂时无法完成优惠核对。" + reason + "不能在缺少优惠政策证据时判断金额或退款。"
                    return {"answer": answer, "events": self.event(state, "synthesize", "success", "优惠证据不足，已停止金额结论。")}
                if state.get("plan") and state["plan"].skill == "price_adjustment":
                    answer = (f"订单 {order['order_id']}：暂时无法完成价差比较。{answer}"
                              "不能在缺少价保政策或价格凭证时判断补差金额。")
                    return {"answer": answer, "events": self.event(state, "synthesize", "success", "价保证据不足，已停止价差结论。")}
                answer = f"订单 {order['order_id']}：{order['product']}，{order['status']}，签收 {order['delivered_days_ago']} 天。\n" + answer + "尚不能确认能否退换。"
        elif state["status"] == "completed" and order:
            discount = state["results"].get("analyze_discount")
            price = state["results"].get("analyze_price_adjustment")
            if price:
                marks = "".join(f"[{number}]" for number in price["citations_used"])
                purchase = price["purchase_price_cents"] / 100
                current = price["current_price_cents"] / 100
                difference = price["difference_cents"] / 100
                if price["eligible"]:
                    answer = (f"订单 {order['order_id']}：购买价 ¥{purchase:.2f}，当前价 ¥{current:.2f}，"
                              f"价差 ¥{difference:.2f}；降价发生在 {price['price_drop_days_ago']} 天前，"
                              f"仍在演示价保 {price['price_protection_days']} 天期限内。{marks}\n"
                              "可按演示政策申请补差，当前只完成价格比较，未执行退款或提交工单。")
                elif difference <= 0:
                    answer = (f"订单 {order['order_id']}：购买价 ¥{purchase:.2f}，当前价 ¥{current:.2f}，"
                              "当前价未低于购买价，没有可计算的演示补差金额。" + marks)
                else:
                    answer = (f"订单 {order['order_id']}：购买价 ¥{purchase:.2f}，当前价 ¥{current:.2f}，"
                              f"价差 ¥{difference:.2f}，但降价已发生 {price['price_drop_days_ago']} 天，"
                              f"超过演示价保 {price['price_protection_days']} 天期限，需要人工复核。{marks}")
                return {"answer": answer, "events": self.event(state, "synthesize", "success", "已完成购买价与当前价的只读比较")}
            if discount:
                marks = "".join(f"[{number}]" for number in discount["citations_used"])
                original = discount["original_cents"] / 100
                saved = discount["discount_cents"] / 100
                paid = discount["paid_cents"] / 100
                expected = discount["expected_paid_cents"] / 100
                refund = discount["refund_amount_cents"] / 100
                if discount["consistent"]:
                    answer = (f"订单 {order['order_id']}：原价 ¥{original:.2f}，优惠 ¥{saved:.2f}，"
                              f"实际支付 ¥{paid:.2f}。金额核对一致：{original:.2f} - {saved:.2f} = {expected:.2f}。{marks}\n"
                              f"如发生退款，演示政策要求按实际支付金额计算，预计退款 ¥{refund:.2f}。{marks}")
                else:
                    difference = abs(discount["difference_cents"]) / 100
                    answer = (f"订单 {order['order_id']}：原价 ¥{original:.2f}，优惠 ¥{saved:.2f}，"
                              f"实际支付 ¥{paid:.2f}；按政策应支付 ¥{expected:.2f}，差额 ¥{difference:.2f}，"
                              f"优惠结果存在差异，需要人工复核。{marks}\n"
                              f"在复核完成前不能自动退款；政策中的退款依据为实际支付金额。{marks}")
                return {"answer": answer, "events": self.event(state, "synthesize", "success", "已完成优惠金额计算并生成带引用的核对结果。")}
            answer = f"订单 {order['order_id']}：{order['product']}，{order['status']}，签收 {order['delivered_days_ago']} 天，实际支付 ¥{order['paid_cents']/100:.2f}。"
            assessment = state["results"].get("assess_after_sales")
            if assessment:
                marks = "".join(f"[{number}]" for number in assessment["citations_used"])
                answer += f"\n演示订单记录的质量问题：{'是' if order['quality_issue'] else '否'}；检索政策的退货期限 {assessment['return_window_days']} 天、换货期限 {assessment['exchange_window_days']} 天。{marks}"
                answer += f"\n退货调查：{'符合演示政策的申请条件' if assessment['return_eligible'] else '未满足演示政策的申请条件，需要人工复核'}。{marks}"
                stock = state["results"].get("check_inventory", {}).get("available", 0)
                exchange = "符合申请条件，当前有换货库存" if assessment["exchange_eligible"] and stock > 0 else "符合政策条件，但当前缺货，需补货或选择退货" if assessment["exchange_eligible"] else "未满足演示政策条件，需要人工复核"
                answer += f"\n换货调查：{exchange}（演示库存 {stock} 件）。{marks}\n本次已完成调查，尚未创建工单或退款。"
        return {"answer": answer, "events": self.event(state, "synthesize", "success", "根据工具结果组装回答；政策结论带引用。")}

    async def llm_answer(self, state):
        """模型组织已核验的事实段落，程序保留原文金额、状态和引用。

        模型不能自由改写事实；索引必须完整且唯一，引用编号必须与段落内标记
        一致。超时、漏段落或引用错误都回退到事实模板并在 Trace 标明原因。
        """
        answer = state.get("answer", "")
        settings = getattr(self.plan_service, "settings", None)
        if not answer or state.get("skip_llm_answer") or getattr(settings, "model_mode", "mock") == "mock":
            return {"events": self.event(state, "llm_answer", "skipped", "Mock 模式保留确定性回答")}
        try:
            response = await asyncio.wait_for(self.plan_service.answer(
                answer, state.get("context", {}),
                [item.model_dump() for item in state.get("citations", [])], state.get("results", {})),
                getattr(settings, "model_timeout", self.timeout))
            allowed = {item.citation_number for item in state.get("citations", [])}
            blocks = [line for line in answer.splitlines() if line.strip()]
            expected = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
            if (sorted(response.fact_indexes) != list(range(len(blocks)))
                    or len(response.citation_numbers) != len(set(response.citation_numbers))
                    or set(response.citation_numbers) != expected or expected - allowed):
                return {"events": self.event(state, "llm_answer", "failed", "模型段落或引用未通过校验，回退到事实模板。")}
            # 丢弃模型自由生成的 answer 字段，使用原文段落；语气只能来自有限的服务用语。
            opening = {"neutral": "", "empathic": "理解你希望尽快解决这个问题。\n",
                       "acknowledge": "我来帮你核对。\n"}[response.tone]
            return {"answer": opening + "\n".join(blocks[index] for index in response.fact_indexes),
                    "events": self.event(state, "llm_answer", "success", "已校验事实段落、金额和状态原文及政策引用。",
                                         details={"fact_indexes": response.fact_indexes, "tone": response.tone})}
        except Exception:
            logger.exception("llm_answer_failed")
            return {"events": self.event(state, "llm_answer", "failed", "模型回答失败，回退到事实模板")}

    async def critic(self, state):
        """检查规则核验所用引用是否存在于实际检索结果，缺引用时拒绝结论。"""
        assessment = state.get("results", {}).get("assess_after_sales")
        discount = state.get("results", {}).get("analyze_discount")
        price = state.get("results", {}).get("analyze_price_adjustment")
        available = {hit.citation_number for hit in state.get("citations", [])}
        if discount and (not discount["citations_used"] or not set(discount["citations_used"]) <= {hit.citation_number for hit in state.get("citations", [])}):
            return {"status": "critic_failed", "answer": "优惠核对引用检查失败，无法给出金额结论。",
                    "events": self.event(state, "critic", "failed", "优惠核对缺少有效政策引用。")}
        if price and (not price["citations_used"] or not set(price["citations_used"]) <= available):
            return {"status": "critic_failed", "answer": "价保分析缺少有效政策引用，无法给出价差结论。",
                    "events": self.event(state, "critic", "failed", "价保分析缺少有效政策引用。")}
        if assessment and (not assessment["citations_used"] or not set(assessment["citations_used"]) <= available):
            return {"status": "critic_failed", "answer": "政策引用检查失败，无法提供退换结论。",
                    "events": self.event(state, "critic", "failed", "政策结论缺少有效引用。")}
        return {"events": self.event(state, "critic", "success", "引用关联检查通过。" if assessment else "没有输出缺乏依据的政策结论。")}
