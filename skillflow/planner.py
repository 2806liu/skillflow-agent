import re

import httpx

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from skillflow.contracts import AssistantResponse, DialogueDecision, Plan, PlanStep


_ORDER_ID_PATTERN = re.compile(r"(?<![A-Z0-9])A[\s-]?[0-9]{3,8}(?![A-Z0-9])", re.IGNORECASE)


def extract_order_ids(message: str) -> list[str]:
    """提取并规范项目 B 的订单号，允许大小写、短横线和单个空格写法。

    例如 A123、a123、A-123、A 123 都归一化为 A123；裸数字 123 不接受，
    避免把普通数字误当成订单并越过订单归属校验。
    """
    ids = []
    for raw in _ORDER_ID_PATTERN.findall(message):
        normalized = re.sub(r"[\s-]", "", raw.upper())
        if normalized not in ids:
            ids.append(normalized)
    return ids


def price_adjustment_message(order_id: str | None) -> str:
    """缺少订单或价格凭证时，说明价保调查需要的下一步。"""
    if order_id:
        return (f"订单 {order_id} 的价保调查需要购买价、当前价和降价时间凭证；"
                "系统会先检索演示价保政策，再比较两者差额，缺少凭证时转人工复核。")
    return ("请提供订单号，例如 A123，并准备购买价、当前价和降价时间凭证；"
            "系统会检索演示价保政策后比较差额，缺少凭证时需要人工复核。")


def classify_intent(goal: str) -> str:
    """按用户目标分类任务，不把意图识别简化为订单号或错别字纠正。

    中文决策顺序：先拦截当前不支持的“优惠券未生效/不能使用”投诉；
    再识别退东西、质量问题、申请退款等售后动作；随后识别优惠金额和退款金额核对；
    没有上述业务动作时才归入普通订单查询。该顺序避免“退货退款”被误判为金额核对。
    """
    unsupported_coupon = re.search(
        r"(?:投诉|优惠券).*(?:没生效|未生效|不生效|不能用|用不了|无法使用)"
        r"|(?:没生效|未生效|不生效|不能用|用不了|无法使用).*(?:优惠券|折扣)",
        goal,
    )
    if unsupported_coupon:
        return "unsupported_coupon"

    # 降价、补差价和价格保护必须进入独立价保链，不能复用优惠核对或订单摘要。
    # 具体是否可补差由本地价格凭证和政策证据决定，缺失时由工具返回证据不足。
    price_adjustment = re.search(
        r"补(?:我|下|一下)?差价|价差|价格保护|价保|保价|降价|降了|价格下降|买贵了",
        goal,
    )
    if price_adjustment:
        return "price_adjustment"

    # “退差价”表示核对优惠差额；必须先于宽泛的“退货/退款”口语规则判断。
    discount_difference = re.search(r"退差价|差价退款|优惠差额|金额差异|金额不一致", goal)
    if discount_difference:
        return "coupon_dispute"

    # “我要退款”表达申请售后；单独“退款”和“退款金额怎么算”仍表示金额核对。
    # 质量不好、退东西等口语描述先做只读资格调查，不直接创建工单。
    return_query = re.search(
        r"退货退款|申请退货|申请换货|退货|换货|退换|售后申请|申请退|能退|能换"
        r"|退东西|想退|要退|退了|(?:我要|我想|申请|给我)退款"
        r"|质量(?:不好|问题|有问题|太差|差)|坏了|故障",
        goal,
    )
    if return_query:
        return "return_exchange"

    discount_query = re.search(
        r"优惠|折扣|优惠券|实付|支付|多收|少收|退款金额|退款|退差价|差价退款|金额差异|金额不一致|营销|促销|优惠争议|金额(?:怎么算|计算|核对|明细|不对)|原价|扣款有争议",
        goal,
    )
    if discount_query:
        return "coupon_dispute"
    return "order_tracking"


def resolve_order(message: str, context: dict):
    """优先提取当前订单；明确指代或连续售后动作可借用已确认订单。"""
    ids = extract_order_ids(message)
    if len(ids) > 1:
        return None
    if ids:
        return ids[0]
    # 普通“这件商品能否退换”仍需订单号；同一会话中已经查到订单后，
    # “这个质量不好”“我要退了”等明确接续的售后动作可沿用该订单。
    # 没有已确认订单时仍返回 None，由 Planner 追问，不猜测订单号。
    if re.search(r"刚才(?:的)?订单|这单|该订单|那单|这笔订单|这个订单|上一单", message):
        return context.get("order_id")
    if context.get("order_id") and re.search(r"这个质量|这款.*质量|我要退|我想退|申请退|退东西|质量不好.*退了", message):
        return context["order_id"]
    # 同一会话中已经确认过商品时，金额、优惠和售后追问默认沿用该商品。
    # 只有明确要求切换订单，才返回 None 让 Planner 重新索要订单号。
    if context.get("order_id") and not re.search(r"另一笔|另一个订单|其他订单|换个订单|换一个订单|换一单|新订单", message):
        return context["order_id"]
    return None


def mock_plan(values: dict):
    """规则 Planner 用于先验证闭环；用户仅补订单号时延续此前待完成目标。"""
    message, context = values["message"], values["context"]
    ids = set(extract_order_ids(message))
    order_id = resolve_order(message, context)
    goal = (context.get("pending_goal") or message) if re.fullmatch(r"\s*A[0-9]{3,8}\s*", message.upper()) else message
    intent = classify_intent(goal)
    if intent == "unsupported_coupon":
        return Plan(decision="unsupported", skill="coupon_dispute", goal=goal[:500],
                    message="优惠券未生效投诉暂不支持。当前可以核对订单优惠金额，或调查商品能否退换。")
    if intent == "price_adjustment":
        if not order_id:
            return Plan(decision="clarify", skill="price_adjustment", goal=goal[:500],
                        message="请提供订单号，例如 A123，并准备购买价、当前价或降价凭证。")
        return Plan(decision="execute", skill="price_adjustment", goal=goal[:500], order_id=order_id,
                    steps=[PlanStep(step_id="order", tool="get_order", arguments={"order_id": order_id}),
                           PlanStep(step_id="policy", tool="retrieve_policy", depends_on=["order"]),
                           PlanStep(step_id="price", tool="analyze_price_adjustment", depends_on=["order", "policy"])])
    skill = "return_exchange" if intent == "return_exchange" else "order_tracking"
    if not order_id:
        return Plan(decision="clarify", skill=skill, goal=goal[:500],
                    message="请指定一笔订单进行调查。" if len(ids) > 1 else "请提供订单号，例如 A123。")
    steps = [PlanStep(step_id="order", tool="get_order", arguments={"order_id": order_id})]
    if skill == "return_exchange":
        steps += [PlanStep(step_id="policy", tool="retrieve_policy", depends_on=["order"]),
                  PlanStep(step_id="assessment", tool="assess_after_sales", depends_on=["order", "policy"]),
                  PlanStep(step_id="inventory", tool="check_inventory", depends_on=["order", "assessment"])]
    return Plan(decision="execute", skill=skill, goal=goal[:500], order_id=order_id, steps=steps)


def mock_plan_b3(values: dict):
    """B3 优惠争议 Planner：先识别订单，再规划政策检索和金额核对。"""
    message, context = values["message"], values["context"]
    ids = set(extract_order_ids(message))
    order_id = resolve_order(message, context)
    goal = (context.get("pending_goal") or message) if re.fullmatch(r"\s*A[0-9]{3,8}\s*", message.upper()) else message
    intent = classify_intent(goal)
    if intent == "unsupported_coupon":
        return Plan(decision="unsupported", skill="coupon_dispute", goal=goal[:500],
                    message="优惠券未生效投诉暂不支持。当前可以核对订单优惠金额，或调查商品能否退换。")
    if intent == "price_adjustment":
        if not order_id:
            return Plan(decision="clarify", skill="price_adjustment", goal=goal[:500],
                        message="请提供订单号，例如 A123，并准备购买价、当前价或降价凭证。")
        return Plan(decision="execute", skill="price_adjustment", goal=goal[:500], order_id=order_id,
                    steps=[PlanStep(step_id="order", tool="get_order", arguments={"order_id": order_id}),
                           PlanStep(step_id="policy", tool="retrieve_policy", depends_on=["order"]),
                           PlanStep(step_id="price", tool="analyze_price_adjustment", depends_on=["order", "policy"])])
    if intent == "return_exchange":
        return mock_plan({"message": message, "context": context})
    if intent == "coupon_dispute":
        if not order_id:
            return Plan(decision="clarify", skill="coupon_dispute", goal=goal[:500],
                        message="请提供需要核对优惠的订单号，例如 A123。")
        return Plan(decision="execute", skill="coupon_dispute", goal=goal[:500], order_id=order_id,
                    steps=[PlanStep(step_id="order", tool="get_order", arguments={"order_id": order_id}),
                           PlanStep(step_id="policy", tool="retrieve_policy", depends_on=["order"]),
                           PlanStep(step_id="discount", tool="analyze_discount", depends_on=["order", "policy"])])
    return mock_plan(values)


class Planner:
    """LangChain 规划入口：规则模拟与真实模型共用 Plan 输出契约。"""
    def __init__(self, settings, skills):
        self.settings = settings
        self.skills = skills
        # Mock 规则同样走 LangChain Runnable，B3 优惠计划由 mock_plan_b3 负责。
        self.mock_chain = RunnableLambda(mock_plan_b3)
        self.real_chain = None
        self.dialogue_chain = None
        self.answer_chain = None

    def _chat_model(self):
        """构造项目 B 专用模型客户端，并显式控制是否继承系统代理环境。"""
        client = httpx.AsyncClient(timeout=self.settings.model_timeout,
                                   trust_env=self.settings.model_trust_env)
        return ChatOpenAI(model=self.settings.model, api_key=self.settings.api_key,
                          base_url=self.settings.base_url or None, temperature=0,
                          timeout=self.settings.model_timeout, max_retries=self.settings.model_retries,
                          http_async_client=client)

    async def decide(self, message, context):
        """先识别对话动作，再交给计划图或工单图执行。

        Mock 模式使用确定性识别保证演示可回归；真实模式使用 LangChain
        结构化输出。模型只能提出动作，后续仍由 API 和工单状态门授权。
        """
        if self.settings.model_mode == "mock":
            from skillflow.dialogue import decide_dialogue
            return decide_dialogue(message, context)
        if not self.settings.api_key or not self.settings.model:
            raise ValueError("真实模型模式需要 SKILLFLOW_API_KEY 和 SKILLFLOW_MODEL")
        if self.dialogue_chain is None:
            llm = self._chat_model()
            # 提示词要求模型识别取消动作但不判断工单是否存在；数据库查询和二次确认由对话状态图负责。
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是售后客服对话理解器，只输出 DialogueDecision。"
                 "请区分业务查询和工单动作；退款字样不代表真实退款。"
                 "确认提交只能在上下文 pending_action=confirm_submit 时提出；取消、撤销、取消订单或取消退款优先输出 ticket_action/cancel。"
                 "单纯问价差是 price_adjustment/investigate；明确申请价差、补差或价保是 ticket_action/create_draft，request_type=price_adjustment，仍须政策复核和用户确认。"
                 "你不能自行断言工单不存在；服务端会按买家和订单查询真实工单。原订单和真实退款不能由本流程取消，需引导用户确认是否取消售后工单。"
                 "订单号只能使用消息或可信上下文中的 A 加数字编号。上下文：{context}"),
                ("human", "用户消息：{message}")
            ])
            self.dialogue_chain = prompt | llm.with_structured_output(DialogueDecision, method="function_calling")
        return await self.dialogue_chain.ainvoke({"message": message, "context": context})

    async def answer(self, draft: str, context: dict, evidence: list[dict], results: dict):
        """模型安排已核验事实段落并选择语气；执行器不会展示模型自由生成的事实。"""
        if self.settings.model_mode == "mock":
            return AssistantResponse(answer=draft, citation_numbers=[])
        if not self.settings.api_key or not self.settings.model:
            return AssistantResponse(answer=draft, citation_numbers=[])
        if self.answer_chain is None:
            llm = self._chat_model()
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是售后客服回答编排器。根据用户上下文选择语气和事实段落顺序。"
                 "fact_indexes 必须包含所有段落索引一次且只一次。确认问题放最后。"
                 "citation_numbers 必须完整包含段落中所有引用编号且不重复。"
                 "tone 选择 neutral/empathic/acknowledge。answer 填草稿原文，禁止编造金额、资格、工单或退款。"),
                ("human", "上下文：{context}\n工具结果：{results}\n证据：{evidence}\n编号段落：{blocks}\n草稿：{draft}")
            ])
            self.answer_chain = prompt | llm.with_structured_output(AssistantResponse, method="function_calling")
        response = await self.answer_chain.ainvoke({"draft": draft, "context": context,
                                                     "evidence": evidence, "results": results,
                                                     "blocks": list(enumerate(line for line in draft.splitlines() if line.strip()))})
        return response

    async def create(self, message, context):
        """只生成计划；工具权限、订单来源与依赖由下方校验器再次检查。"""
        values = {"message": message, "context": context}
        if self.settings.model_mode == "mock":
            return await self.mock_chain.ainvoke(values)
        if not self.settings.api_key or not self.settings.model:
            raise ValueError("真实模型模式需要 SKILLFLOW_API_KEY 和 SKILLFLOW_MODEL；不会自动切换 Mock。")
        if self.real_chain is None:
            llm = self._chat_model()
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是演示数码小店的调查 Planner。只规划，不执行、不编造结果。技能：{skills}。"
                 "订单查询使用 get_order；退换调查依次 get_order、retrieve_policy、assess_after_sales、check_inventory。"
                 "get_order 参数只能是 order_id；其他工具参数必须为空，由执行器绑定实际结果。"
                 "依赖必须引用先前 step_id，调查必须依赖政策与订单，库存必须依赖调查。"
                 "订单缺失/多笔时 clarify；优惠争议先执行 get_order、retrieve_policy、analyze_discount，只返回核对结果，不直接退款、提交工单或修改数据。"
                 "优惠券未生效、优惠券不能使用或投诉优惠券属于暂不支持，直接返回 unsupported，不调用工具。"
                 "降价、补差价、价格保护或价保使用 price_adjustment，按 get_order、retrieve_policy、analyze_price_adjustment 顺序只读比较；缺少价格凭证时返回证据不足，不退款、不提交工单。"
                 "退差价、优惠差额或金额不一致属于 coupon_dispute，按 get_order、retrieve_policy、analyze_discount 顺序规划；不要把它当成购买后降价价保。"
                 "订单号必须来自当前请求或可信上下文；补订单号时保持 pending_goal 的原任务。"),
                ("system", "B3 优惠争议必须按 get_order、retrieve_policy、analyze_discount 顺序规划；不得直接退款或修改订单。"),
                ("human", "可信上下文：{context}\n请求：{message}")])
            self.real_chain = prompt | llm.with_structured_output(Plan, method="function_calling")
        return await self.real_chain.ainvoke({**values, "skills": list(self.skills.values())})

def validate_plan(plan: Plan, message: str, context: dict, skills: dict):
    """阻止未知工具、越权订单、伪造结果、循环依赖及缺少必需步骤的计划。"""
    if plan.decision != "execute":
        return
    if plan.skill not in skills:
        raise ValueError("该技能尚未实现。")
    if plan.order_id != resolve_order(message, context):
        raise ValueError("计划订单号必须来自当前请求或已确认的会话上下文。")
    seen = {}
    for step in plan.steps:
        if step.step_id in seen or any(dep not in seen for dep in step.depends_on):
            raise ValueError("步骤 ID 重复、依赖不存在或依赖成环。")
        if step.tool not in skills[plan.skill]["allowed_tools"]:
            raise ValueError("计划调用了技能白名单以外的工具。")
        if step.tool in seen.values():
            raise ValueError("同一个只读工具不可重复执行。")
        if step.tool == "get_order":
            if seen or step.arguments != {"order_id": plan.order_id}:
                raise ValueError("第一步必须查询该订单，且不能传入用户身份或其他参数。")
        elif step.arguments:
            raise ValueError("业务工具参数由执行器绑定，Planner 不得注入订单或政策结果。")
        dependencies = {seen[dep] for dep in step.depends_on}
        required = {"retrieve_policy": {"get_order"},
                    "assess_after_sales": {"get_order", "retrieve_policy"},
                    "analyze_discount": {"get_order", "retrieve_policy"},
                    "analyze_price_adjustment": {"get_order", "retrieve_policy"},
                    "check_inventory": {"get_order", "assess_after_sales"}}.get(step.tool, set())
        if not required <= dependencies:
            raise ValueError("缺少必须的业务前置步骤。")
        seen[step.step_id] = step.tool
    required_tools = ({"get_order"} if plan.skill == "order_tracking" else
                      {"get_order", "retrieve_policy", "analyze_discount"} if plan.skill == "coupon_dispute" else
                      {"get_order", "retrieve_policy", "analyze_price_adjustment"} if plan.skill == "price_adjustment" else
                      {"get_order", "retrieve_policy", "assess_after_sales", "check_inventory"})
    if set(seen.values()) != required_tools:
        raise ValueError("调查计划缺少验收所需的步骤。")
