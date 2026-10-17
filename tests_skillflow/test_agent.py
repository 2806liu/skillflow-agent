"""验收项目 B 独立闭环：真实本地政策检索、计划校验、会话和失败路径。"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from skillflow.api import create_app
from skillflow.config import ROOT, Settings
from skillflow.contracts import DialogueDecision, ToolResult
from skillflow.planner import classify_intent, extract_order_ids, mock_plan, validate_plan
from skillflow.skills.registry import load_skills
from skillflow.sqlite_store import SqliteStore
from skillflow.business.sqlite_tickets import SqliteTicketStore


class AgentTests(unittest.TestCase):
    def setUp(self):
        # 每个测试复制 B 自己的数据到独立目录，修改坏案例不会污染演示文件。
        self.test_root = (ROOT / ".skillflow" / "tests").resolve()
        self.directory = self.test_root / str(uuid4())
        fixtures = self.directory / "fixtures"
        (fixtures / "policies").mkdir(parents=True)
        for name in ("buyers.json", "orders.json", "inventory.json"):
            shutil.copyfile(ROOT / "data" / name, fixtures / name)
        self.policy_path = fixtures / "policies" / "earbuds_v1.md"
        shutil.copyfile(ROOT / "data/policies/earbuds_v1.md", self.policy_path)
        # 价格保护测试使用项目 B 自己的演示凭证；缺少该文件时工具应走证据不足分支。
        shutil.copyfile(ROOT / "data/price_adjustments.json", fixtures / "price_adjustments.json")
        # 单元回归固定使用 Mock 路由，避免开发者 .env 启用真实 LLM 后
        # 测试结果受网络、模型随机性或代理状态影响；真实模型另用连通性脚本验收。
        self.settings = Settings(data_dir=self.directory / "state", fixtures_dir=fixtures,
                                 timeout=1, model_mode="mock")
        self.app = create_app(self.settings)
        self.policies = self.app.state.policies
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        # 删除前核对绝对路径仍位于本次测试根目录，只清理自己生成的数据。
        if self.directory.resolve().is_relative_to(self.test_root) and self.directory != self.test_root:
            shutil.rmtree(self.directory)

    def ask(self, message, session="test"):
        """通过完整 API 和 LangGraph 流程验收，不直接调用回答函数。"""
        response = self.client.post("/chat", json={"session_id": session, "message": message})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_normal_executes_all_tools_with_local_policy_citation(self):
        result = self.ask("查询订单A123，结合政策调查能否退换")
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["result"]["assess_after_sales"]["return_eligible"])
        executed = [event["tool"] for event in result["events"] if event["node"] == "execute"]
        self.assertEqual(executed, ["get_order", "retrieve_policy", "assess_after_sales", "check_inventory"])
        self.assertIn("[1]", result["answer"])
        document = result["citations"][0]
        self.assertEqual(self.client.get("/policies/" + document["document_id"] + "/source").text, self.policy_path.read_text(encoding="utf-8"))
        self.assertEqual(self.client.get("/runs/" + result["run_id"]).json(), result)

    def test_missing_order_then_number_preserves_task(self):
        """补号后先确认商品，确认后才恢复原退换调查任务。"""
        initial = self.ask("调查这件商品能否退换")
        self.assertEqual(initial["status"], "clarify")
        self.assertFalse(any(e["node"] == "execute" for e in initial["events"]))
        product = self.ask("A123")
        self.assertEqual(product["status"], "clarify")
        self.assertIn("是否就是这件商品", product["answer"])
        self.assertEqual([event["tool"] for event in product["events"] if event["node"] == "execute"], ["get_order"])
        final = self.ask("对的")
        self.assertEqual(final["plan"]["skill"], "return_exchange")
        self.assertEqual(final["status"], "completed")

    def test_followup_uses_order_and_sessions_are_isolated(self):
        self.ask("查询 A123")
        self.assertEqual(self.ask("那这单能换货吗")["plan"]["order_id"], "A123")
        self.assertEqual(self.ask("这单能换货吗", "other")["status"], "clarify")

    def test_ambiguous_product_reference_does_not_reuse_order(self):
        # 已确认商品后，“这件商品”默认指向当前商品；明确切换订单才重新追问。
        self.ask("查询 A123")
        result = self.ask("这件商品能否退换")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["plan"]["order_id"], "A123")
        self.assertEqual(result["plan"]["skill"], "return_exchange")

    def test_out_of_stock(self):
        self.assertIn("缺货", self.ask("A124 能否退换")["answer"])

    def test_outside_policy_window(self):
        result = self.ask("A126 能否退换")
        self.assertFalse(result["result"]["assess_after_sales"]["return_eligible"])
        self.assertIn("人工复核", result["answer"])

    def test_missing_policy_skips_decision(self):
        result = self.ask("A125 能否退换")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertNotIn("assess_after_sales", result["result"])
        self.assertFalse(result["citations"])

    def test_unknown_and_foreign_orders_do_not_leak(self):
        for order in ("A999", "A900"):
            result = self.ask(order + " 能否退换")
            self.assertEqual(result["status"], "order_not_found")
            self.assertNotIn("其他用户", result["answer"])
            self.assertNotIn("retrieve_policy", result["result"])

    def test_unrelated_query_returns_no_policy_evidence(self):
        # 验证本地检索真实空结果，而非构造一个成功的展示响应。
        result = asyncio.run(self.policies.retrieve("earbuds_v1", "nonexistent_query_token_xyz"))
        self.assertEqual(result.status, "empty")

    def test_normalized_policy_paragraph_retains_rules(self):
        self.policy_path.write_text(" ".join(self.policy_path.read_text(encoding="utf-8").split()), encoding="utf-8")
        self.policies.reload()
        self.assertEqual(self.ask("A123 能否退换")["status"], "completed")

    def test_timeout_retries_once_and_never_fakes_evidence(self):
        # 只注入检索超时，保留实际执行器，验证有限重试与后续步骤跳过。
        with patch.object(self.policies, "retrieve", side_effect=asyncio.TimeoutError) as retrieve:
            result = self.ask("A123 能否退换")
        self.assertEqual(retrieve.call_count, 2)
        self.assertEqual(result["status"], "tool_failed")
        self.assertFalse(result["citations"])

    def test_policy_scope_does_not_use_other_product_rules(self):
        # 即使查询词包含已有耳机政策，也不能给无政策手柄返回耳机规则。
        result = asyncio.run(self.policies.retrieve("controller_missing", "earbuds_v1 退货 换货"))
        self.assertEqual(result.status, "empty")

    def test_incomplete_and_conflicting_policy_refuse_conclusion(self):
        snippet = "policy_key: earbuds_v1\nreturn_window_days: 7\n"
        self.policy_path.write_text(snippet, encoding="utf-8")
        self.policies.reload()
        self.assertEqual(self.ask("A123 能否退换")["status"], "insufficient_evidence")
        self.policy_path.write_text(snippet + "return_window_days: 30\nexchange_window_days: 7\nquality_only: true\n", encoding="utf-8")
        self.policies.reload()
        result = self.ask("A123 能否退换")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIn("冲突", result["answer"])

    def test_coupon_dispute_calculates_paid_amount_with_policy_citation(self):
        # B3 只核对金额并给出退款依据，不执行退款或修改订单。
        result = self.ask("A123 优惠券扣款有争议，为什么实际支付 189 元？")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["plan"]["skill"], "coupon_dispute")
        self.assertEqual([event["tool"] for event in result["events"] if event["node"] == "execute"],
                         ["get_order", "retrieve_policy", "analyze_discount"])
        analysis = result["result"]["analyze_discount"]
        self.assertTrue(analysis["consistent"])
        self.assertEqual(analysis["expected_paid_cents"], 18900)
        self.assertEqual(analysis["refund_amount_cents"], 18900)
        self.assertIn("189.00", result["answer"])
        self.assertTrue(result["citations"])

    def test_coupon_dispute_without_order_clarifies(self):
        result = self.ask("优惠券扣款有争议")
        self.assertEqual(result["status"], "clarify")
        self.assertFalse(result["result"])

    def test_coupon_dispute_without_policy_refuses_amount_conclusion(self):
        result = self.ask("A125 优惠券扣款有争议")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertNotIn("analyze_discount", result["result"])
        self.assertIn("优惠核对", result["answer"])

    def test_intent_classifier_separates_task_types(self):
        """意图分类按用户任务判断，退货退款不能被退款金额关键词覆盖。"""
        self.assertEqual(classify_intent("查一下订单 A123"), "order_tracking")
        self.assertEqual(classify_intent("这件商品能退吗"), "return_exchange")
        self.assertEqual(classify_intent("退货退款"), "return_exchange")
        self.assertEqual(classify_intent("退款金额怎么算"), "coupon_dispute")
        self.assertEqual(classify_intent("金额怎么算"), "coupon_dispute")
        self.assertEqual(classify_intent("我要退款"), "return_exchange")
        self.assertEqual(classify_intent("我要退东西"), "return_exchange")
        self.assertEqual(classify_intent("这个质量不好"), "return_exchange")
        self.assertEqual(classify_intent("质量不好我要退了"), "return_exchange")
        self.assertEqual(classify_intent("我要投诉优惠券没生效"), "unsupported_coupon")
        self.assertEqual(classify_intent("但是你这个降价了"), "price_adjustment")
        self.assertEqual(classify_intent("补差价"), "price_adjustment")
        self.assertEqual(classify_intent("价格保护"), "price_adjustment")
        self.assertEqual(classify_intent("退差价"), "coupon_dispute")
        self.assertEqual(classify_intent("我要退差价"), "coupon_dispute")

    def test_difference_request_routes_to_coupon_dispute(self):
        """“退差价”属于订单优惠差额核对；补齐订单号后沿用 B3 工具链。"""
        clarify = self.ask("退差价", session="difference-intent")
        self.assertEqual(clarify["status"], "clarify")
        self.assertEqual(clarify["plan"]["skill"], "coupon_dispute")
        result = self.ask("A127", session="difference-intent")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["plan"]["skill"], "coupon_dispute")
        self.assertFalse(result["result"]["analyze_discount"]["consistent"])
        self.assertEqual([event["tool"] for event in result["events"] if event["node"] == "execute"],
                         ["get_order", "retrieve_policy", "analyze_discount"])

    def test_price_adjustment_without_order_clarifies(self):
        """没有订单号时先追问，不执行价格工具，也不猜测当前商品。"""
        result = self.ask("补差价")
        self.assertEqual(result["status"], "clarify")
        self.assertEqual(result["plan"]["skill"], "price_adjustment")
        self.assertFalse(result["result"])
        self.assertFalse(any(event["node"] == "execute" for event in result["events"]))
        self.assertIn("订单号", result["answer"])

    def test_price_adjustment_followup_compares_prices(self):
        """已有当前订单时，按政策和本地演示凭证比较购买价与当前价。"""
        self.ask("查询订单 A123", session="price-followup")
        result = self.ask("但是你这个降价了", session="price-followup")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["plan"]["skill"], "price_adjustment")
        self.assertEqual(result["plan"]["order_id"], "A123")
        self.assertEqual([event["tool"] for event in result["events"] if event["node"] == "execute"],
                         ["get_order", "retrieve_policy", "analyze_price_adjustment"])
        self.assertEqual(result["result"]["analyze_price_adjustment"]["difference_cents"], 1000)
        self.assertIn("179.00", result["answer"])

    def test_model_misroutes_are_corrected_before_tools(self):
        """模拟模型错分任务，校验对话门按明确词义恢复正确的只读工具链。"""
        planner = self.app.state.runtime.plan_service
        wrong_difference = DialogueDecision(intent="price_adjustment", order_id="A127",
                                            reference="explicit", requested_action="investigate")
        with patch.object(planner, "decide", return_value=wrong_difference):
            dispute = self.ask("退差价 A127", session="model-wrong-difference")
        self.assertEqual(dispute["plan"]["skill"], "coupon_dispute")
        self.assertEqual([event["tool"] for event in dispute["events"] if event["node"] == "execute"],
                         ["get_order", "retrieve_policy", "analyze_discount"])
        wrong_status = DialogueDecision(intent="order_query", order_id="A123",
                                        reference="explicit", requested_action="status")
        with patch.object(planner, "decide", return_value=wrong_status):
            order = self.ask("查询订单 A123", session="model-wrong-status")
        self.assertEqual(order["plan"]["skill"], "order_tracking")
        self.assertEqual([event["tool"] for event in order["events"] if event["node"] == "execute"], ["get_order"])

    def test_order_id_variants_are_normalized_without_accepting_bare_numbers(self):
        """订单号允许常见书写差异，但裸数字仍必须被拒绝。"""
        self.assertEqual(extract_order_ids("订单 a-123 和 A 124"), ["A123", "A124"])
        self.assertEqual(extract_order_ids("订单 123"), [])

    def test_confirmed_product_is_default_for_amount_and_return_followups(self):
        """首次查询确认商品后，后续金额和质量追问默认沿用该商品。"""
        first = self.ask("查询订单 A123", session="active-product")
        self.assertEqual(first["plan"]["order_id"], "A123")
        amount = self.ask("金额怎么算", session="active-product")
        self.assertEqual(amount["plan"]["skill"], "coupon_dispute")
        self.assertEqual(amount["plan"]["order_id"], "A123")
        self.assertEqual([event["tool"] for event in amount["events"] if event["node"] == "execute"],
                         ["get_order", "retrieve_policy", "analyze_discount"])
        quality = self.ask("这个质量不好", session="active-product")
        self.assertEqual(quality["plan"]["skill"], "return_exchange")
        self.assertEqual(quality["plan"]["order_id"], "A123")

    def test_switch_order_semantic_requires_new_order_id(self):
        """明确切换其他订单时不能静默复用当前商品。"""
        self.ask("查询订单 A123", session="switch-product")
        result = self.ask("换一个订单调查退货", session="switch-product")
        self.assertEqual(result["status"], "clarify")
        self.assertEqual(result["plan"]["skill"], "return_exchange")

    def test_colloquial_return_request_preserves_goal_until_order_arrives(self):
        """口语退货先追问订单号和商品确认，再执行资格调查并提供工单依据。"""
        for message in ("要退东西", "我要退东西", "我要退款", "这个质量不好"):
            session = "colloquial-" + message
            clarify = self.ask(message, session=session)
            self.assertEqual(clarify["status"], "clarify")
            self.assertEqual(clarify["plan"]["skill"], "return_exchange")
            product = self.ask("A123", session=session)
            self.assertEqual(product["status"], "clarify")
            self.assertEqual([event["tool"] for event in product["events"] if event["node"] == "execute"], ["get_order"])
            result = self.ask("对的", session=session)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["plan"]["skill"], "return_exchange")
            self.assertTrue(result["result"]["assess_after_sales"]["return_eligible"])
            self.assertEqual([event["tool"] for event in result["events"] if event["node"] == "execute"],
                             ["get_order", "retrieve_policy", "assess_after_sales", "check_inventory"])

    def test_explicit_return_followup_uses_confirmed_order_in_same_session(self):
        """查询 A123 后的明确退货动作沿用本会话订单，其他会话不能借用。"""
        self.ask("查询 A123", session="return-followup")
        result = self.ask("质量不好我要退了", session="return-followup")
        self.assertEqual(result["plan"]["order_id"], "A123")
        self.assertEqual(result["plan"]["skill"], "return_exchange")
        self.assertTrue(result["result"]["assess_after_sales"]["return_eligible"])
        self.assertEqual(self.ask("质量不好我要退了", session="other-return-followup")["status"], "clarify")

    def test_unsupported_coupon_complaint_does_not_call_tools(self):
        """优惠券未生效投诉属于暂不支持能力，直接结束而不是误进金额核对。"""
        result = self.ask("我要投诉优惠券没生效")
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["plan"]["skill"], "coupon_dispute")
        self.assertFalse(result["result"])
        self.assertFalse(any(event["node"] == "execute" for event in result["events"]))

    def test_refund_intent_routes_to_coupon_dispute_and_keeps_pending_goal(self):
        """单独说“退款”先进入金额核对；补订单号后执行 B3 工具链。"""
        clarify = self.ask("退款", session="refund-intent")
        self.assertEqual(clarify["status"], "clarify")
        self.assertEqual(clarify["plan"]["skill"], "coupon_dispute")
        result = self.ask("A123", session="refund-intent")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["plan"]["skill"], "coupon_dispute")
        self.assertEqual([event["tool"] for event in result["events"] if event["node"] == "execute"],
                         ["get_order", "retrieve_policy", "analyze_discount"])

    def test_return_refund_intent_routes_to_return_exchange(self):
        """“申请退货”和“退货退款”经商品确认后进入售后调查，不能误路由为优惠核对。"""
        for message in ("申请退货", "退货退款"):
            clarify = self.ask(message, session="return-intent-" + message)
            self.assertEqual(clarify["status"], "clarify")
            self.assertEqual(clarify["plan"]["skill"], "return_exchange")
            product = self.ask("A123", session="return-intent-" + message)
            self.assertEqual(product["status"], "clarify")
            result = self.ask("对的", session="return-intent-" + message)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["plan"]["skill"], "return_exchange")
            self.assertEqual([event["tool"] for event in result["events"] if event["node"] == "execute"],
                             ["get_order", "retrieve_policy", "assess_after_sales", "check_inventory"])

    def test_multitarget_requires_clarification(self):
        self.assertEqual(self.ask("A123 和 A124 能否退换")["status"], "clarify")

    def test_plan_validation_rejects_identity_tool_and_dependency_injection(self):
        message = "A123 能否退换"
        # 每次只污染一处，确认权限与依赖的约束独立生效。
        mutations = [lambda p: p.steps[0].arguments.update({"buyer_id": "other-buyer"}),
                     lambda p: setattr(p.steps[1], "tool", "shell"),
                     lambda p: setattr(p.steps[1], "depends_on", ["inventory"]),
                     lambda p: setattr(p.steps[2], "depends_on", ["order"]),
                     lambda p: setattr(p, "order_id", "A900"), lambda p: p.steps.pop()]
        for mutate in mutations:
            plan = mock_plan({"message": message, "context": {}})
            mutate(plan)
            with self.assertRaises(ValueError):
                validate_plan(plan, message, {}, load_skills())

    def test_blank_message_is_rejected(self):
        self.assertEqual(self.client.post("/chat", json={"session_id": "a", "message": "   "}).status_code, 422)

    def test_mock_workflow_makes_no_external_http_requests(self):
        # 阻断异步 HTTP 客户端：独立 B 的健康检查、调查和引用仍应全部成功。
        with patch("httpx.AsyncClient.request", side_effect=AssertionError("禁止外部 HTTP")):
            self.assertEqual(self.client.get("/health").json()["policy_source"], "project_b_local")
            self.assertEqual(self.ask("A123 能否退换")["status"], "completed")

    def test_unknown_policy_source_is_not_a_file_path(self):
        self.assertEqual(self.client.get("/policies/unknown/source").status_code, 404)

    def test_discount_dispute_ticket_requires_real_amount_difference(self):
        """优惠争议只允许金额不一致订单建单，并复用确认、幂等、取消和历史接口。"""
        consistent = self.client.post("/tickets/draft", json={
            "session_id": "discount-consistent", "order_id": "A123",
            "request_type": "discount_dispute", "reason": "申请核对优惠差异"}).json()
        self.assertEqual(consistent["status"], "rejected")
        self.assertIsNone(consistent["ticket"])
        self.assertIn("一致", consistent["message"])
        self.assertTrue(any(event["node"] == "ticket_gate" and event["status"] == "failed"
                            for event in consistent["events"]))

        draft_response = self.client.post("/tickets/draft", json={
            "session_id": "discount-dispute", "order_id": "A127",
            "request_type": "discount_dispute", "reason": "实际支付与优惠计算不一致"})
        self.assertEqual(draft_response.status_code, 200)
        draft = draft_response.json()
        self.assertEqual(draft["status"], "draft_created")
        self.assertEqual(draft["ticket"]["request_type"], "discount_dispute")
        self.assertEqual(draft["ticket"]["metadata"]["difference_cents"], -500)
        self.assertEqual(draft["ticket"]["metadata"]["expected_paid_cents"], 19400)
        self.assertEqual([event["tool"] for event in draft["events"] if event["node"] == "execute"],
                         ["get_order", "retrieve_policy", "analyze_discount"])
        self.assertFalse(any(event["tool"] == "refund" for event in draft["events"] if event["node"] == "execute"))

        ticket_id = draft["ticket"]["ticket_id"]
        submitted = self.client.post(f"/tickets/{ticket_id}/confirm", json={
            "session_id": "discount-dispute", "confirmed": True}).json()
        self.assertEqual(submitted["status"], "submitted")
        self.assertEqual(submitted["ticket"]["request_type"], "discount_dispute")
        repeated = self.client.post(f"/tickets/{ticket_id}/confirm", json={
            "session_id": "discount-dispute", "confirmed": True}).json()
        self.assertEqual(repeated["status"], "already_submitted")
        self.assertEqual(repeated["ticket"]["ticket_id"], ticket_id)
        history = self.client.get(f"/tickets/{ticket_id}/events?session_id=discount-dispute")
        self.assertEqual(history.status_code, 200)
        self.assertTrue(any(event["action"] == "confirm_repeated" for event in history.json()))

        missing_policy = self.client.post("/tickets/draft", json={
            "session_id": "discount-missing-policy", "order_id": "A125",
            "request_type": "discount_dispute", "reason": "缺少政策时不应建单"}).json()
        self.assertEqual(missing_policy["status"], "rejected")
        self.assertIsNone(missing_policy["ticket"])
        self.assertIn("优惠争议", missing_policy["message"])
        self.assertFalse(any(event["node"] == "draft" and event["status"] == "success"
                             for event in missing_policy["events"]))

    def test_ticket_draft_requires_confirmation_and_is_idempotent(self):
        # 第一次请求只生成草稿；重复生成和重复确认都不能创建第二张工单。
        payload = {"session_id": "ticket-session", "order_id": "A123", "request_type": "return", "reason": "商品质量问题"}
        draft = self.client.post("/tickets/draft", json=payload)
        self.assertEqual(draft.status_code, 200, draft.text)
        draft_data = draft.json()
        self.assertEqual(draft_data["status"], "draft_created")
        ticket_id = draft_data["ticket"]["ticket_id"]
        self.assertEqual(draft_data["ticket"]["status"], "draft")
        duplicate_draft = self.client.post("/tickets/draft", json=payload).json()
        self.assertEqual(duplicate_draft["ticket"]["ticket_id"], ticket_id)
        kept = self.client.post(f"/tickets/{ticket_id}/confirm", json={"session_id": "ticket-session", "confirmed": False}).json()
        self.assertEqual(kept["ticket"]["status"], "draft")
        submitted = self.client.post(f"/tickets/{ticket_id}/confirm", json={"session_id": "ticket-session", "confirmed": True}).json()
        self.assertEqual(submitted["status"], "submitted")
        repeated = self.client.post(f"/tickets/{ticket_id}/confirm", json={"session_id": "ticket-session", "confirmed": True}).json()
        self.assertEqual(repeated["status"], "already_submitted")
        self.assertEqual(repeated["ticket"]["ticket_id"], ticket_id)

    def test_chat_draft_rechecks_with_fixed_plan_when_model_plan_fails(self):
        """用户确认商品后说“申请”，二次调查使用固定计划，不受模型漏步骤影响。"""
        session = "chat-fixed-ticket-plan"
        self.assertEqual(self.ask("质量不好，我不要了", session=session)["status"], "clarify")
        self.assertEqual(self.ask("A123", session=session)["status"], "clarify")
        investigated = self.ask("对的", session=session)
        self.assertEqual(investigated["status"], "completed")
        planner = self.app.state.runtime.plan_service
        request = DialogueDecision(intent="ticket_action", order_id="A123", reference="current_order",
                                   requested_action="create_draft", request_type="return")
        # 对话模型可以建议创建草稿；服务端复核不能再依赖模型重新编排业务步骤。
        with patch.object(planner, "decide", return_value=request), patch.object(
                planner, "create", side_effect=AssertionError("工单复核不应调用模型规划")):
            draft = self.ask("申请", session=session)
        self.assertEqual(draft["status"], "draft_created")
        self.assertEqual(draft["result"]["ticket"]["status"], "draft")
        self.assertEqual([event["tool"] for event in draft["events"] if event["node"] == "execute"],
                         ["get_order", "retrieve_policy", "assess_after_sales", "check_inventory"])
        self.assertIn("尚未执行真实退款", draft["answer"])

    def test_cancel_order_guides_to_real_ticket_and_requires_confirmation(self):
        """模型声称没有工单时仍按买家查询数据库；取消原订单须先引导并确认工单。"""
        session = "cancel-existing-return"
        payload = {"session_id": session, "order_id": "A124", "request_type": "return", "reason": "质量问题"}
        draft = self.client.post("/tickets/draft", json=payload).json()
        ticket_id = draft["ticket"]["ticket_id"]
        self.client.post(f"/tickets/{ticket_id}/confirm", json={"session_id": session, "confirmed": True})
        self.assertEqual(self.ask("有价保吗", session=session)["status"], "completed")

        planner = self.app.state.runtime.plan_service
        wrong = DialogueDecision(intent="ticket_action", order_id="A124", ticket_id=ticket_id,
                                 reference="current_ticket", requested_action="none", request_type="return",
                                 needs_clarification=True, clarification_question="没有查到工单。")
        with patch.object(planner, "decide", return_value=wrong):
            guided = self.ask("取消订单", session=session)
        self.assertEqual(guided["status"], "clarify")
        self.assertIn("不能取消原订单", guided["answer"])
        self.assertIn(ticket_id, guided["answer"])
        self.assertEqual(self.client.get(f"/tickets/{ticket_id}?session_id={session}").json()["ticket"]["status"], "submitted")

        wrong_refund = DialogueDecision(intent="ticket_action", order_id="A124", reference="explicit",
                                        requested_action="cancel", needs_clarification=True,
                                        clarification_question="目前没有检测到 A124 的工单。")
        with patch.object(planner, "decide", return_value=wrong_refund):
            clarified = self.ask("取消A124的退款", session=session)
        self.assertEqual(clarified["status"], "clarify")
        self.assertIn("尚未执行退款", clarified["answer"])
        self.assertIn(ticket_id, clarified["answer"])
        self.assertNotIn("没有检测到", clarified["answer"])
        cancelled = self.ask("确认取消", session=session)
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(cancelled["result"]["ticket"]["ticket_id"], ticket_id)

    def test_cancel_target_can_change_with_order_only_followup(self):
        """取消确认时改报另一订单号，应重查该订单工单，不沿用旧工单。"""
        session = "cancel-switch-order"
        tickets = {}
        for order_id in ("A123", "A124"):
            draft = self.client.post("/tickets/draft", json={
                "session_id": session, "order_id": order_id, "request_type": "return", "reason": "质量问题"}).json()
            tickets[order_id] = draft["ticket"]["ticket_id"]
            self.client.post(f"/tickets/{tickets[order_id]}/confirm", json={
                "session_id": session, "confirmed": True})
        self.assertIn(tickets["A124"], self.ask("取消订单", session=session)["answer"])
        switched = self.ask("A123", session=session)
        self.assertEqual(switched["status"], "clarify")
        self.assertIn(tickets["A123"], switched["answer"])
        self.assertNotIn(tickets["A124"], switched["answer"])
        self.assertEqual(self.client.get(f"/tickets/{tickets['A124']}?session_id={session}").json()["ticket"]["status"], "submitted")

    def test_ticket_blocks_cross_type_active_conflict(self):
        """同一订单已有一种已提交售后时，另一种类型必须被互斥规则拦截。"""
        session = "ticket-conflict"
        exchange = self.client.post("/tickets/draft", json={
            "session_id": session, "order_id": "A123", "request_type": "exchange", "reason": "商品换货申请"}).json()
        self.assertEqual(exchange["status"], "draft_created")
        ticket_id = exchange["ticket"]["ticket_id"]
        submitted = self.client.post(f"/tickets/{ticket_id}/confirm", json={
            "session_id": session, "confirmed": True}).json()
        self.assertEqual(submitted["status"], "submitted")

        conflict = self.client.post("/tickets/draft", json={
            "session_id": session, "order_id": "A123", "request_type": "return", "reason": "商品退货申请"}).json()
        self.assertEqual(conflict["status"], "rejected")
        self.assertIsNone(conflict["ticket"])
        self.assertIn("换货工单", conflict["message"])
        self.assertIn(ticket_id, conflict["message"])

        cancelled = self.client.post(f"/tickets/{ticket_id}/cancel", json={"session_id": session}).json()
        self.assertEqual(cancelled["status"], "cancelled")
        cancelled_again = self.client.post(f"/tickets/{ticket_id}/confirm", json={"session_id": session, "confirmed": True}).json()
        self.assertEqual(cancelled_again["status"], "rejected")
        self.assertEqual(cancelled_again["error_code"], "TICKET_CANCELLED")
        after_cancel = self.client.post("/tickets/draft", json={
            "session_id": session, "order_id": "A123", "request_type": "return", "reason": "商品退货申请"}).json()
        self.assertEqual(after_cancel["status"], "draft_created")

    def test_ticket_gate_rejects_ineligible_or_foreign_orders(self):
        # A124 换货缺货、A125 无政策、A126 超期和 A900 越权都不得写入工单。
        for order_id, request_type in (("A124", "exchange"), ("A125", "return"), ("A126", "return"), ("A900", "return")):
            response = self.client.post("/tickets/draft", json={
                "session_id": "ticket-reject-" + order_id, "order_id": order_id,
                "request_type": request_type, "reason": "申请售后"})
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()
            self.assertEqual(data["status"], "rejected")
            self.assertIsNone(data["ticket"])
            self.assertTrue(any(event["node"] == "ticket_gate" and event["status"] == "failed" for event in data["events"]))

    def test_same_buyer_can_confirm_from_another_session(self):
        draft = self.client.post("/tickets/draft", json={
            "session_id": "ticket-owner", "order_id": "A123", "request_type": "exchange", "reason": "换货申请"}).json()
        ticket_id = draft["ticket"]["ticket_id"]
        response = self.client.post(f"/tickets/{ticket_id}/confirm", json={"session_id": "other-session", "confirmed": True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "submitted")

    def test_same_buyer_can_recover_draft_from_another_session(self):
        # 同一演示用户的另一个会话不能通过重复草稿请求读取已有工单内容。
        first = self.client.post("/tickets/draft", json={
            "session_id": "ticket-owner-2", "order_id": "A123", "request_type": "return", "reason": "质量问题"}).json()
        self.assertEqual(first["status"], "draft_created")
        second = self.client.post("/tickets/draft", json={
            "session_id": "other-session-2", "order_id": "A123", "request_type": "return", "reason": "质量问题"}).json()
        self.assertEqual(second["status"], "draft_created")
        self.assertEqual(second["ticket"]["ticket_id"], first["ticket"]["ticket_id"])
        submitted = self.client.post(f"/tickets/{first['ticket']['ticket_id']}/confirm", json={
            "session_id": "other-session-2", "confirmed": True}).json()
        self.assertEqual(submitted["status"], "submitted")
        recovered_submitted = self.client.post("/tickets/draft", json={
            "session_id": "new-session-3", "order_id": "A123", "request_type": "return", "reason": "商品退货申请"}).json()
        self.assertEqual(recovered_submitted["status"], "already_submitted")
        self.assertEqual(recovered_submitted["ticket"]["ticket_id"], first["ticket"]["ticket_id"])

    def test_other_buyer_cannot_read_or_confirm_ticket(self):
        draft = self.client.post("/tickets/draft", json={
            "session_id": "buyer-owner", "order_id": "A123", "request_type": "exchange", "reason": "商品换货申请"}).json()
        ticket_id = draft["ticket"]["ticket_id"]
        foreign_settings = Settings(data_dir=self.settings.data_dir, fixtures_dir=self.settings.fixtures_dir,
                                     timeout=1, demo_buyer_id="other-buyer")
        with TestClient(create_app(foreign_settings)) as foreign_client:
            # 演示模式也用明确的 TICKET_FORBIDDEN 拒绝已存在的他人工单。
            denied_read = foreign_client.get(f"/tickets/{ticket_id}?session_id=foreign")
            self.assertEqual(denied_read.status_code, 403)
            self.assertEqual(denied_read.json()["detail"]["code"], "TICKET_FORBIDDEN")
            denied_confirm = foreign_client.post(
                f"/tickets/{ticket_id}/confirm", json={"session_id": "foreign", "confirmed": True})
            self.assertEqual(denied_confirm.status_code, 403)
            self.assertEqual(denied_confirm.json()["detail"]["code"], "TICKET_FORBIDDEN")

    def test_multiple_demo_buyers_can_only_see_their_own_orders(self):
        """不同服务端买家身份只能查询自己名下订单，不能依靠 session_id 越权。"""
        other_settings = Settings(data_dir=self.settings.data_dir / "other", fixtures_dir=self.settings.fixtures_dir,
                                  timeout=1, demo_buyer_id="other-buyer")
        alice_settings = Settings(data_dir=self.settings.data_dir / "alice", fixtures_dir=self.settings.fixtures_dir,
                                  timeout=1, demo_buyer_id="alice-buyer")
        with TestClient(create_app(other_settings)) as other_client, TestClient(create_app(alice_settings)) as alice_client:
            other_orders = {order["order_id"] for order in other_client.get("/demo/orders").json()}
            alice_orders = {order["order_id"] for order in alice_client.get("/demo/orders").json()}
            self.assertEqual(other_orders, {"A900"})
            self.assertEqual(alice_orders, {"A901"})
            self.assertEqual(other_client.get("/health").json()["buyer_id"], "other-buyer")
            self.assertEqual(alice_client.get("/health").json()["buyer_id"], "alice-buyer")
            self.assertEqual(other_client.get("/demo/buyers").status_code, 200)

    def test_ticket_events_preserve_cross_session_audit_history(self):
        """工单事件保存 buyer_id、操作会话和状态结果，并受买家权限保护。"""
        draft = self.client.post("/tickets/draft", json={
            "session_id": "audit-session-a", "order_id": "A123",
            "request_type": "exchange", "reason": "审计验证"}).json()
        ticket_id = draft["ticket"]["ticket_id"]
        self.client.post(f"/tickets/{ticket_id}/confirm", json={
            "session_id": "audit-session-b", "confirmed": True})
        repeated = self.client.post(f"/tickets/{ticket_id}/confirm", json={
            "session_id": "audit-session-c", "confirmed": True}).json()
        self.assertEqual(repeated["status"], "already_submitted")
        self.client.post(f"/tickets/{ticket_id}/cancel", json={"session_id": "audit-session-d"})

        history = self.client.get(f"/tickets/{ticket_id}/events?session_id=audit-reader")
        self.assertEqual(history.status_code, 200)
        events = history.json()
        self.assertEqual([event["action"] for event in events], [
            "draft_created", "confirm", "confirm_repeated", "cancel"])
        self.assertEqual([event["session_id"] for event in events], [
            "audit-session-a", "audit-session-b", "audit-session-c", "audit-session-d"])
        self.assertTrue(all(event["buyer_id"] == "demo-buyer" for event in events))

        foreign_settings = Settings(data_dir=self.settings.data_dir, fixtures_dir=self.settings.fixtures_dir,
                                     timeout=1, demo_buyer_id="alice-buyer")
        with TestClient(create_app(foreign_settings)) as foreign_client:
            denied_history = foreign_client.get(f"/tickets/{ticket_id}/events?session_id=alice")
            self.assertEqual(denied_history.status_code, 403)
            self.assertEqual(denied_history.json()["detail"]["code"], "TICKET_FORBIDDEN")

    def test_restart_recovers_product_draft_and_cancel_confirmations(self):
        """重建应用模拟服务重启；商品、提交及取消确认都从检查点恢复，并核验真实工单。"""
        session = "restart-confirmations"
        self.assertEqual(self.ask("商品质量不好，我要退货", session)["status"], "clarify")
        self.assertEqual(self.ask("A123", session)["status"], "clarify")
        self.client.close()
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)
        self.assertEqual(self.ask("对的", session)["status"], "completed")
        draft = self.ask("申请吧", session)
        self.assertEqual(draft["status"], "draft_created")
        ticket_id = draft["result"]["ticket"]["ticket_id"]
        self.client.close()
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)
        submitted = self.ask("对的，我接受", session)
        self.assertEqual(submitted["status"], "submitted")
        self.assertEqual(submitted["result"]["ticket"]["ticket_id"], ticket_id)
        self.assertEqual(self.ask("取消这笔退货工单", session)["status"], "clarify")
        self.client.close()
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)
        cancelled = self.ask("确认取消", session)
        self.assertEqual(cancelled["status"], "cancelled")
        history = self.client.get(f"/tickets/{ticket_id}/events?session_id={session}").json()
        self.assertEqual([item["action"] for item in history].count("confirm"), 1)
        self.assertEqual([item["action"] for item in history].count("cancel"), 1)
        self.assertEqual(self.client.get("/health").json()["session_mode"], "sqlite_langgraph_checkpointer")

    def test_concurrent_instances_reuse_one_ticket(self):
        """两个应用实例同时建草稿与提交，仅产生一张工单和一次状态转换。"""
        payload = {"session_id": "parallel-owner", "order_id": "A123",
                   "request_type": "return", "reason": "商品质量问题"}
        with TestClient(create_app(self.settings)) as second:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(self.client.post, "/tickets/draft", json=payload)
                other = pool.submit(second.post, "/tickets/draft", json=payload)
                drafts = [first.result().json(), other.result().json()]
            ids = {item["ticket"]["ticket_id"] for item in drafts}
            self.assertEqual(len(ids), 1)
            ticket_id = ids.pop()
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(self.client.post, f"/tickets/{ticket_id}/confirm",
                                    json={"session_id": "parallel-owner", "confirmed": True})
                other = pool.submit(second.post, f"/tickets/{ticket_id}/confirm",
                                    json={"session_id": "parallel-owner", "confirmed": True})
                results = [first.result().json(), other.result().json()]
            self.assertEqual({item["status"] for item in results}, {"submitted", "already_submitted"})
            history = self.client.get(f"/tickets/{ticket_id}/events?session_id=parallel-owner").json()
            self.assertEqual([item["action"] for item in history].count("draft_created"), 1)
            self.assertEqual([item["action"] for item in history].count("confirm"), 1)
            self.assertEqual(len(self.app.state.tickets.for_order(buyer_id="demo-buyer", order_id="A123")), 1)

    def test_ticket_state_and_history_rollback_together(self):
        """审计写入失败时提交和取消都回滚，防止状态变化却缺少历史。"""
        draft = self.client.post("/tickets/draft", json={
            "session_id": "rollback", "order_id": "A123", "request_type": "return",
            "reason": "商品质量问题"}).json()
        ticket_id = draft["ticket"]["ticket_id"]
        ticket_store = self.app.state.tickets
        with patch.object(type(ticket_store), "_event", side_effect=RuntimeError("audit unavailable")):
            with self.assertRaisesRegex(RuntimeError, "audit unavailable"):
                ticket_store.confirm(ticket_id, session_id="rollback", buyer_id="demo-buyer")
        self.assertEqual(ticket_store.get(ticket_id)["status"], "draft")
        ticket_store.confirm(ticket_id, session_id="rollback", buyer_id="demo-buyer")
        with patch.object(type(ticket_store), "_event", side_effect=RuntimeError("audit unavailable")):
            with self.assertRaisesRegex(RuntimeError, "audit unavailable"):
                ticket_store.cancel(ticket_id, session_id="rollback", buyer_id="demo-buyer")
        record = ticket_store.get(ticket_id)
        self.assertEqual(record["status"], "submitted")
        self.assertEqual([item["action"] for item in record["ticket_events"]], ["draft_created", "confirm"])

    def test_legacy_json_import_is_incremental_and_preserves_sources(self):
        """旧 JSON 未变化时不重放，旧进程停机前的增量状态仍能接收。"""
        legacy_root = self.directory / "legacy"
        legacy_root.mkdir()
        key = SqliteStore.session_key("demo-buyer:legacy-session")
        session_file = legacy_root / f"{key}.json"
        session_file.write_text(json.dumps({"order_id": "A123", "pending_action": "confirm_product"}), encoding="utf-8")
        record, _, _ = self.app.state.tickets.create_draft(
            buyer_id="demo-buyer", session_id="legacy-session", order_id="A123", sku="earbuds-black",
            request_type="return", reason="质量问题", source_run_id="legacy-test")
        ticket_file = legacy_root / "tickets.json"
        ticket_file.write_text(json.dumps({record["ticket_id"]: record}, ensure_ascii=False), encoding="utf-8")
        migrated = SqliteStore(legacy_root)
        self.assertEqual(migrated.session("legacy-session", "demo-buyer")["pending_action"], "confirm_product")
        self.assertEqual(len(SqliteTicketStore(migrated).events(SqliteTicketStore(migrated).get(record["ticket_id"]))), 1)
        self.assertTrue(session_file.exists())
        self.assertTrue(ticket_file.exists())
        # 第二次启动不得重放历史；随后旧文件出现变化时应接收最后一次写入。
        SqliteStore(legacy_root)
        self.assertEqual(len(SqliteTicketStore(migrated).get(record["ticket_id"])["ticket_events"]), 1)
        session_file.write_text(json.dumps({"order_id": "A124", "pending_action": "confirm_product"}), encoding="utf-8")
        record["status"] = "submitted"
        record["updated_at"] = "2099-01-01T00:00:00+00:00"
        record["ticket_events"].append({**record["ticket_events"][0], "event_id": str(uuid4()),
                                         "action": "confirm", "result": "submitted"})
        ticket_file.write_text(json.dumps({record["ticket_id"]: record}, ensure_ascii=False), encoding="utf-8")
        SqliteStore(legacy_root)
        self.assertEqual(migrated.session("legacy-session", "demo-buyer")["order_id"], "A124")
        self.assertEqual(SqliteTicketStore(migrated).get(record["ticket_id"])["status"], "submitted")
        self.assertEqual(len(SqliteTicketStore(migrated).get(record["ticket_id"])["ticket_events"]), 2)

    def test_restart_revalidates_stale_cancel_confirmation(self):
        """检查点里的取消待确认不能覆盖数据库中已取消的真实工单。"""
        session = "stale-cancel"
        draft = self.client.post("/tickets/draft", json={"session_id": session, "order_id": "A123",
                                                        "request_type": "return", "reason": "质量问题"}).json()
        ticket_id = draft["ticket"]["ticket_id"]
        self.client.post(f"/tickets/{ticket_id}/confirm", json={"session_id": session, "confirmed": True})
        self.assertEqual(self.ask("取消A123的退货工单", session)["status"], "clarify")
        # 模拟另一个服务实例先完成取消，而本会话检查点仍等待用户确认。
        self.app.state.tickets.cancel(ticket_id, session_id="other-instance", buyer_id="demo-buyer")
        self.client.close()
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)
        self.ask("确认取消", session)
        record = self.app.state.tickets.get(ticket_id)
        self.assertEqual(record["status"], "cancelled")
        self.assertEqual([event["action"] for event in record["ticket_events"]].count("cancel"), 1)

    def test_quality_complaint_cannot_be_upgraded_to_draft_by_model(self):
        """模型把含糊质量抱怨误判为建单时，守卫仍只允许调查。"""
        session = "quality-not-draft"
        self.assertEqual(self.ask("查询订单 A124，结合店铺政策调查能否退换", session)["status"], "completed")
        mistaken = DialogueDecision(intent="ticket_action", order_id="A124", reference="current_order",
                                    requested_action="create_draft", request_type="return")
        with patch.object(self.app.state.runtime.plan_service, "decide", return_value=mistaken):
            answer = self.ask("商品质量不好，我不要了", session)
        self.assertEqual(answer["status"], "completed")
        self.assertEqual(answer["plan"]["skill"], "return_exchange")
        self.assertNotIn("ticket", answer["result"])
        self.assertEqual(self.app.state.tickets.for_order(buyer_id="demo-buyer", order_id="A124"), [])

    def test_not_this_product_detaches_old_draft(self):
        """用户否认商品后解除旧订单和草稿绑定，后续不能再把 A124 当目标。"""
        session = "reject-product"
        draft = self.client.post("/tickets/draft", json={"session_id": session, "order_id": "A124",
                                                        "request_type": "return", "reason": "质量问题"}).json()
        ticket_id = draft["ticket"]["ticket_id"]
        rejected = self.ask("不是它", session)
        self.assertEqual(rejected["status"], "clarify")
        self.assertIn("正确的订单号", rejected["answer"])
        self.assertEqual(self.app.state.tickets.get(ticket_id)["status"], "draft")
        followup = self.ask("我要退货", session)
        self.assertEqual(followup["status"], "clarify")
        self.assertNotIn("A124", followup["answer"])
        self.assertEqual(self.ask("A123", session)["status"], "clarify")

    def test_submit_and_accept_without_pending_confirmation(self):
        """已提交时“提交”返回原工单；无待确认动作时“我接受”不得执行新写入。"""
        session = "submit-after-done"
        draft = self.client.post("/tickets/draft", json={"session_id": session, "order_id": "A123",
                                                        "request_type": "return", "reason": "质量问题"}).json()
        ticket_id = draft["ticket"]["ticket_id"]
        self.client.post(f"/tickets/{ticket_id}/confirm", json={"session_id": session, "confirmed": True})
        repeated = self.ask("提交", session)
        self.assertEqual(repeated["status"], "already_submitted")
        self.assertEqual(repeated["result"]["ticket"]["ticket_id"], ticket_id)
        self.assertEqual(self.ask("对的我接受", session)["status"], "clarify")
        record = self.app.state.tickets.get(ticket_id)
        self.assertEqual([item["action"] for item in record["ticket_events"]].count("confirm"), 1)

    def test_price_adjustment_ticket_requires_evidence_and_confirmation(self):
        """A123 价保正差额可建草稿；A124 无差额、A125 缺证据和 A126 超期均不得建单。"""
        payload = {"session_id": "price-owner", "order_id": "A123",
                   "request_type": "price_adjustment", "reason": "购买后降价申请价保"}
        draft = self.client.post("/tickets/draft", json=payload).json()
        self.assertEqual(draft["status"], "draft_created")
        ticket_id = draft["ticket"]["ticket_id"]
        self.assertEqual(draft["ticket"]["metadata"]["difference_cents"], 1000)
        self.assertTrue(draft["ticket"]["metadata"]["policy_document_ids"])
        self.assertEqual([event["tool"] for event in draft["events"] if event["node"] == "execute"],
                         ["get_order", "retrieve_policy", "analyze_price_adjustment"])
        self.assertEqual(self.client.post(f"/tickets/{ticket_id}/confirm", json={
            "session_id": "price-owner", "confirmed": False}).json()["ticket"]["status"], "draft")
        submitted = self.client.post(f"/tickets/{ticket_id}/confirm", json={
            "session_id": "price-owner", "confirmed": True}).json()
        self.assertEqual(submitted["status"], "submitted")
        repeated = self.client.post("/tickets/draft", json={**payload, "session_id": "other-session"}).json()
        self.assertEqual(repeated["ticket"]["ticket_id"], ticket_id)
        self.assertEqual(repeated["status"], "already_submitted")
        for order_id in ("A124", "A125", "A126"):
            denied = self.client.post("/tickets/draft", json={**payload, "order_id": order_id}).json()
            self.assertEqual(denied["status"], "rejected")
            self.assertIsNone(denied["ticket"])
        cancelled = self.client.post(f"/tickets/{ticket_id}/cancel", json={"session_id": "price-owner"}).json()
        self.assertEqual(cancelled["status"], "cancelled")
        history = self.client.get(f"/tickets/{ticket_id}/events?session_id=price-owner").json()
        self.assertEqual([item["action"] for item in history].count("confirm"), 1)
        self.assertEqual([item["action"] for item in history].count("cancel"), 1)
        self.assertFalse(any(event.get("tool") == "refund" for event in draft["events"]))

    def test_price_adjustment_and_return_cannot_be_active_together(self):
        """同一订单的退货与价保补差不能并行建单；取消后可改选另一种申请。"""
        return_draft = self.client.post("/tickets/draft", json={"session_id": "price-conflict",
            "order_id": "A123", "request_type": "return", "reason": "质量问题"}).json()
        return_id = return_draft["ticket"]["ticket_id"]
        payload = {"session_id": "price-conflict", "order_id": "A123",
                   "request_type": "price_adjustment", "reason": "降价申请"}
        conflict = self.client.post("/tickets/draft", json=payload).json()
        self.assertEqual(conflict["status"], "rejected")
        self.assertIn(return_id, conflict["message"])
        self.client.post(f"/tickets/{return_id}/cancel", json={"session_id": "price-conflict"})
        price_draft = self.client.post("/tickets/draft", json=payload).json()
        self.assertEqual(price_draft["status"], "draft_created")
        blocked_return = self.client.post("/tickets/draft", json={"session_id": "price-conflict",
            "order_id": "A123", "request_type": "return", "reason": "质量问题"}).json()
        self.assertEqual(blocked_return["status"], "rejected")
        self.assertEqual(blocked_return["error_code"], "TICKET_CONFLICT")

    def test_price_query_explains_existing_return_before_application(self):
        """已有退货工单时仍返回价差证据，但不引导用户创建冲突的价保草稿。"""
        session = "price-existing-return"
        draft = self.client.post("/tickets/draft", json={"session_id": session,
            "order_id": "A123", "request_type": "return", "reason": "商品质量问题"}).json()
        self.assertEqual(draft["status"], "draft_created")
        result = self.ask("A123当前有价差吗", session)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["plan"]["skill"], "price_adjustment")
        self.assertEqual(result["result"]["analyze_price_adjustment"]["difference_cents"], 1000)
        self.assertTrue(result["result"]["price_application_blocked"])
        self.assertIn(draft["ticket"]["ticket_id"], result["answer"])
        self.assertNotIn("要为你创建价保补差", result["answer"])
        self.assertNotIn("ticket", result["result"])

    def test_return_inquiry_exposes_price_ticket_conflict_before_draft(self):
        """已有价保工单时调查保留政策结论，但隐藏申请入口；显式申请停在资格门。"""
        session = "price-blocks-return"
        price = self.client.post("/tickets/draft", json={"session_id": session,
            "order_id": "A123", "request_type": "price_adjustment", "reason": "购买后降价"}).json()
        price_id = price["ticket"]["ticket_id"]
        submitted = self.client.post(f"/tickets/{price_id}/confirm", json={
            "session_id": session, "confirmed": True}).json()
        self.assertEqual(submitted["status"], "submitted")
        investigated = self.ask("查询订单 A123，结合店铺政策，调查这件商品能否退换", session)
        self.assertEqual(investigated["status"], "completed")
        self.assertTrue(investigated["result"]["assess_after_sales"]["return_eligible"])
        self.assertEqual(investigated["result"]["ticket_conflict"]["ticket_id"], price_id)
        self.assertIn(price_id, investigated["answer"])
        self.assertNotIn("要为你创建退货申请草稿", investigated["answer"])
        rejected = self.ask("A123申请退货", session)
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(rejected["result"]["ticket_error_code"], "TICKET_CONFLICT")
        self.assertTrue(any(event["node"] == "ticket_gate" and event["status"] == "failed"
                            for event in rejected["events"]))
        self.assertFalse(any(event["node"] == "draft" for event in rejected["events"]))
        self.assertEqual(len(self.app.state.tickets.for_order(
            buyer_id=self.settings.demo_buyer_id, order_id="A123")), 1)
        self.client.post(f"/tickets/{price_id}/cancel", json={"session_id": session})
        after_cancel = self.ask("A123能退吗", session)
        self.assertNotIn("ticket_conflict", after_cancel["result"])

    def test_inquiry_shows_existing_return_without_repeat_draft_history(self):
        """同类草稿或已提交工单在调查阶段直接展示；只读调查不产生复用历史。"""
        session = "existing-return-inquiry"
        draft = self.client.post("/tickets/draft", json={"session_id": session,
            "order_id": "A124", "request_type": "return", "reason": "质量问题"}).json()
        ticket_id = draft["ticket"]["ticket_id"]
        record = self.app.state.tickets.get(ticket_id)
        before = len(self.app.state.tickets.events(record))
        for expected_status in ("draft", "submitted"):
            if expected_status == "submitted":
                self.client.post(f"/tickets/{ticket_id}/confirm", json={
                    "session_id": session, "confirmed": True})
                record = self.app.state.tickets.get(ticket_id)
                before = len(self.app.state.tickets.events(record))
            investigated = self.ask("查询订单 A124，结合政策调查能否退换", session)
            self.assertEqual(investigated["status"], "completed")
            self.assertEqual(investigated["result"]["ticket_existing"]["ticket_id"], ticket_id)
            self.assertEqual(investigated["result"]["ticket_existing"]["status"], expected_status)
            self.assertIn(ticket_id, investigated["answer"])
            self.assertNotIn("要为你创建退货申请草稿", investigated["answer"])
            self.assertIn("未创建新工单", investigated["answer"])
            self.assertEqual(len(self.app.state.tickets.events(
                self.app.state.tickets.get(ticket_id))), before)
        self.client.post(f"/tickets/{ticket_id}/cancel", json={"session_id": session})
        after_cancel = self.ask("A124能退吗", session)
        self.assertNotIn("ticket_existing", after_cancel["result"])
        self.assertIn("要为你创建退货申请草稿", after_cancel["answer"])

    def test_price_inquiry_shows_existing_same_type_ticket(self):
        """已有价保工单时只给价格核对与原工单状态，不邀请再次创建价保草稿。"""
        session = "existing-price-inquiry"
        draft = self.client.post("/tickets/draft", json={"session_id": session,
            "order_id": "A123", "request_type": "price_adjustment", "reason": "购买后降价"}).json()
        ticket_id = draft["ticket"]["ticket_id"]
        self.client.post(f"/tickets/{ticket_id}/confirm", json={
            "session_id": session, "confirmed": True})
        investigated = self.ask("A123当前有价差吗", session)
        self.assertEqual(investigated["status"], "completed")
        self.assertEqual(investigated["result"]["ticket_existing"]["ticket_id"], ticket_id)
        self.assertIn(ticket_id, investigated["answer"])
        self.assertNotIn("要为你创建价保补差申请草稿", investigated["answer"])
        self.assertNotIn("未执行退款或提交工单", investigated["answer"])

    def test_explicit_return_application_reaches_conflict_gate_when_model_misclassifies(self):
        """模型把明确申请误判为追问时仍执行到资格门，冲突不得进入草稿节点。"""
        session = "model-misreads-return-application"
        price = self.client.post("/tickets/draft", json={"session_id": session,
            "order_id": "A123", "request_type": "price_adjustment", "reason": "购买后降价"}).json()
        ticket_id = price["ticket"]["ticket_id"]
        self.client.post(f"/tickets/{ticket_id}/confirm", json={
            "session_id": session, "confirmed": True})
        self.ask("查询订单 A123，结合政策调查能否退换", session)
        mistaken = DialogueDecision(intent="return_exchange", order_id="A123", reference="explicit",
                                    requested_action="investigate", needs_clarification=True,
                                    clarification_question="请说明要调查什么")
        with patch.object(self.app.state.runtime.plan_service, "decide", return_value=mistaken):
            rejected = self.ask("A123申请退货", session)
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(rejected["result"]["ticket_error_code"], "TICKET_CONFLICT")
        self.assertTrue(any(event["node"] == "ticket_gate" and event["status"] == "failed"
                            for event in rejected["events"]))
        self.assertFalse(any(event["node"] == "draft" for event in rejected["events"]))
        self.assertEqual(len(self.app.state.tickets.for_order(
            buyer_id=self.settings.demo_buyer_id, order_id="A123")), 1)

    def test_price_confirmation_rechecks_changed_evidence_after_restart(self):
        """草稿后价格发生变化时，重启恢复的确认必须拒绝并留在草稿状态。"""
        payload = {"session_id": "price-recheck", "order_id": "A123",
                   "request_type": "price_adjustment", "reason": "购买后降价"}
        draft = self.client.post("/tickets/draft", json=payload).json()
        ticket_id = draft["ticket"]["ticket_id"]
        price_file = self.settings.fixtures_dir / "price_adjustments.json"
        prices = json.loads(price_file.read_text(encoding="utf-8"))
        prices["A123"]["current_price_cents"] = 17800
        price_file.write_text(json.dumps(prices, ensure_ascii=False), encoding="utf-8")
        self.client.close()
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)
        rejected = self.client.post(f"/tickets/{ticket_id}/confirm", json={
            "session_id": "price-recheck", "confirmed": True}).json()
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(rejected["error_code"], "PRICE_EVIDENCE_CHANGED")
        self.assertEqual(self.app.state.tickets.get(ticket_id)["status"], "draft")
        self.assertIn("confirm_revalidation_failed", [item["action"] for item in rejected["ticket_events"]])

    def test_chat_price_application_has_draft_and_confirmation_steps(self):
        """“申请价差”从商品确认进入价保调查，再经草稿和明确确认提交。"""
        session = "price-chat"
        first = self.ask("A123申请价差", session)
        self.assertEqual(first["status"], "clarify")
        self.assertIn("是否就是这件商品", first["answer"])
        investigated = self.ask("对的", session)
        self.assertEqual(investigated["status"], "completed")
        self.assertEqual(investigated["plan"]["skill"], "price_adjustment")
        self.assertIn("申请草稿", investigated["answer"])
        draft = self.ask("申请吧", session)
        self.assertEqual(draft["status"], "draft_created")
        self.assertEqual(draft["result"]["ticket"]["request_type"], "price_adjustment")
        submitted = self.ask("确认提交", session)
        self.assertEqual(submitted["status"], "submitted")
        self.assertEqual(submitted["result"]["ticket"]["ticket_id"], draft["result"]["ticket"]["ticket_id"])

    def test_price_question_cannot_be_promoted_to_draft_by_model(self):
        """模型误把“有价差吗”判为申请时，守卫仍只执行价格调查。"""
        session = "price-query-only"
        self.assertEqual(self.ask("A123当前有价差吗", session)["status"], "completed")
        mistaken = DialogueDecision(intent="ticket_action", order_id="A123", reference="current_order",
                                    requested_action="create_draft", request_type="price_adjustment")
        with patch.object(self.app.state.runtime.plan_service, "decide", return_value=mistaken):
            queried = self.ask("当前有价差吗", session)
        self.assertEqual(queried["status"], "completed")
        self.assertEqual(queried["plan"]["skill"], "price_adjustment")
        self.assertNotIn("ticket", queried["result"])


if __name__ == "__main__":
    unittest.main()
