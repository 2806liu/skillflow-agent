import asyncio
import re

from langchain_core.tools import StructuredTool
from pydantic import Field

from corecoder.tools.base import Tool
from skillflow.contracts import Contract, Evidence, ToolResult


class OrderInput(Contract):
    """工具只能接收订单号，用户身份由服务端注入，避免越权查询。"""
    order_id: str = Field(pattern=r"^A[0-9]{3,8}$")


class PolicyInput(OrderInput):
    """检索词由执行器绑定订单事实，本地政策模块负责商品范围过滤。"""
    query: str = Field(min_length=1, max_length=1000)


class AssessmentInput(OrderInput):
    """退换核验只接受前置检索工具返回的证据。"""
    evidence: list[Evidence]


class DiscountInput(OrderInput):
    """优惠核对只接收前置订单和政策证据，金额必须来自订单数据。"""
    evidence: list[Evidence]


class PriceAdjustmentInput(OrderInput):
    """价差分析只接收订单和前一步政策证据，禁止模型自行传入价格结果。"""
    evidence: list[Evidence]


class TypedCoreTool(Tool):
    """复用本项目 CoreCoder 副本的 Tool 协议，增加类型校验和 LangChain 适配。

    中文调用链说明：
    1. LangGraph execute 节点调用本类的 call()；
    2. call() 调用 LangChain StructuredTool.ainvoke()；
    3. StructuredTool 根据 args_schema 校验参数，并进入 invoke_json()；
    4. invoke_json() 调用项目 B 的业务 handler；
    5. handler 返回 ToolResult，最后恢复为类型化对象交给 LangGraph。
    """

    def __init__(self, name, description, input_model, handler):
        # 保留 CoreCoder 的工具描述协议，同时注册可异步执行的 LangChain 工具。
        self.name, self.description = name, description
        self.input_model, self.handler = input_model, handler
        self.parameters = input_model.model_json_schema()
        self.langchain_tool = StructuredTool.from_function(
            func=self.execute, coroutine=self.invoke_json, name=name,
            description=description, args_schema=input_model)

    async def invoke_json(self, **kwargs):
        # StructuredTool.ainvoke() 会进入这里。先把模型或执行器传入的字典
        # 转成对应的 Pydantic 输入模型，阻止非法订单号、额外身份字段等数据继续流动。
        inputs = self.input_model.model_validate(kwargs)
        # handler 是项目 B 的真实业务函数，例如 get_order、retrieve 或 inventory；
        # 它只访问本项目自己的 JSON 订单、库存和 Markdown 政策数据。
        result = await self.handler(inputs)
        # 业务函数必须返回符合 ToolResult 契约的结果；序列化为 JSON 是为了适配
        # LangChain StructuredTool 的统一工具返回格式，调用方随后会恢复成 ToolResult。
        return ToolResult.model_validate(result).model_dump_json()

    def execute(self, **kwargs):
        """同步适配入口；LangGraph 在异步执行时使用下方 call。"""
        return asyncio.run(self.invoke_json(**kwargs))

    async def call(self, arguments):
        """真正经过 LangChain 工具调用，并将 JSON 结果恢复为类型化契约。

        这是 LangGraph execute 节点和 LangChain 工具之间的适配边界：
        execute 不直接调用业务函数，而是统一走 StructuredTool.ainvoke()，
        这样每个工具都经过同一套输入校验、异步调用和输出校验。
        """
        tool_json = await self.langchain_tool.ainvoke(arguments)
        # LangChain 返回 JSON 字符串后，在回到 LangGraph 前恢复成 ToolResult，
        # 让 runtime 可以统一读取 status、data、message 和 retryable 字段。
        return ToolResult.model_validate_json(tool_json)


def create_tools(repository, buyer_id, policies):
    """组装项目 B 自有业务工具；不注册跨项目调用、文件编辑或 Shell 工具。

    中文映射说明：definitions 把“工具名、说明、输入模型、业务函数”绑定起来，
    TypedCoreTool 再将这组定义同时暴露为 CoreCoder Tool 和 LangChain StructuredTool。
    """
    def order_for(order_id):
        # 工具绑定经认证的买家实例；模型参数只有订单号，不能覆盖 buyer_id。
        return repository.get_order(order_id, buyer_id)

    async def get_order(inputs):
        """不存在和不属于当前用户的订单统一返回空结果。"""
        order = order_for(inputs.order_id)
        return ToolResult(status="success", data={"order": order}) if order else ToolResult(
            status="empty", error_code="ORDER_NOT_FOUND", message="未查询到当前买家的这笔订单，请核对订单号。")

    async def retrieve(inputs):
        """从实际订单取得 policy_key，只检索本项目对应商品的政策。"""
        order = order_for(inputs.order_id)
        if not order:
            return await get_order(inputs)
        return await policies.retrieve(order["policy_key"], inputs.query)

    async def assess(inputs):
        """解析检索到的完整规则；规则缺失或相互冲突时停止判断。"""
        order = order_for(inputs.order_id)
        if not order:
            return await get_order(inputs)
        rules = {}
        used = set()
        # 只使用属于该商品的政策片段，且每个结论记录实际使用的引用编号。
        for hit in inputs.evidence:
            policy_keys = set(re.findall(r"(?<!\w)policy_key:\s*([a-zA-Z0-9_]+)(?!\w)", hit.snippet))
            if policy_keys != {order["policy_key"]}:
                continue
            for key in ("return_window_days", "exchange_window_days", "quality_only"):
                # 同时兼容换行与空格分隔的字段，不将版式变化误判为缺少规则。
                matches = re.findall(r"(?<!\w)" + key + r":\s*([0-9]+|true|false)(?=\s|$)", hit.snippet)
                for value in matches:
                    if key in rules and rules[key] != value:
                        return ToolResult(status="empty", error_code="POLICY_CONFLICT", message="政策规则冲突，需要人工核实。")
                    rules[key] = value
                    used.add(hit.citation_number)
        if not {"return_window_days", "exchange_window_days", "quality_only"} <= rules.keys():
            return ToolResult(status="empty", error_code="POLICY_INCOMPLETE", message="已检索到片段，但缺少完整退换条件，无法作出结论。")
        try:
            quality_required = {"true": True, "false": False}[rules["quality_only"]]
            return_days, exchange_days = int(rules["return_window_days"]), int(rules["exchange_window_days"])
        except (KeyError, ValueError):
            return ToolResult(status="empty", error_code="POLICY_INVALID_RULE", message="政策规则字段无效，无法作出结论。")
        quality_ok = bool(order["quality_issue"]) or not quality_required
        return ToolResult(status="success", data={
            "return_eligible": quality_ok and order["delivered_days_ago"] <= return_days,
            "exchange_eligible": quality_ok and order["delivered_days_ago"] <= exchange_days,
            "return_window_days": return_days, "exchange_window_days": exchange_days,
            "quality_required": quality_required, "citations_used": sorted(used)})

    async def inventory(inputs):
        """读取演示库存；库存不足影响换货路径，不自动创建工单。"""
        order = order_for(inputs.order_id)
        return ToolResult(status="success", data={"sku": order["sku"], "available": repository.inventory(order["sku"])}) if order else await get_order(inputs)

    # 这张表是工具注册中心。LangGraph 计划中的 step.tool 必须命中这里，
    # 否则不会进入业务函数，避免 Planner 任意调用未授权能力。
    async def analyze_discount(inputs):
        """按政策核对原价、优惠和实付金额，不执行退款或修改订单。

        该工具只做确定性的金额计算：expected_paid = original - discount。
        政策证据缺失、订单金额字段缺失或金额为负数时，返回可解释的失败码，
        由 LangGraph 停止后续结论，避免 Agent 凭空判断优惠是否正确。
        """
        order = order_for(inputs.order_id)
        if not order:
            return await get_order(inputs)

        rules = {}
        used = set()
        for hit in inputs.evidence:
            policy_keys = set(re.findall(r"(?<!\w)policy_key:\s*([a-zA-Z0-9_]+)(?!\w)", hit.snippet))
            if policy_keys != {order["policy_key"]}:
                continue
            for key in ("refund_basis", "discount_basis", "discount_consistency"):
                matches = re.findall(r"(?<!\w)" + key + r":\s*([^\s]+)(?=\s|$)", hit.snippet)
                for value in matches:
                    if key in rules and rules[key] != value:
                        return ToolResult(status="empty", error_code="POLICY_CONFLICT",
                                          message="优惠政策字段冲突，需要人工复核")
                    rules[key] = value
                    used.add(hit.citation_number)
        required = {"refund_basis", "discount_basis", "discount_consistency"}
        if not required <= rules.keys():
            return ToolResult(status="empty", error_code="POLICY_INCOMPLETE",
                              message="缺少优惠核对或退款依据，无法判断优惠争议")

        fields = ("original_cents", "discount_cents", "paid_cents")
        if any(field not in order for field in fields):
            return ToolResult(status="empty", error_code="ORDER_AMOUNT_FIELDS_MISSING",
                              message="订单缺少原价、优惠或实付金额，无法完成核对")
        try:
            original = int(order["original_cents"])
            discount = int(order["discount_cents"])
            paid = int(order["paid_cents"])
        except (TypeError, ValueError):
            return ToolResult(status="empty", error_code="ORDER_AMOUNT_INVALID",
                              message="订单金额字段格式错误，无法完成核对")
        if min(original, discount, paid) < 0 or discount > original:
            return ToolResult(status="empty", error_code="ORDER_AMOUNT_INVALID",
                              message="订单金额不满足基本约束，需要人工复核")

        expected_paid = original - discount
        difference = paid - expected_paid
        return ToolResult(status="success", data={
            "original_cents": original,
            "discount_cents": discount,
            "paid_cents": paid,
            "expected_paid_cents": expected_paid,
            "difference_cents": difference,
            "consistent": difference == 0,
            "discount_basis": rules["discount_basis"],
            "discount_consistency": rules["discount_consistency"],
            "refund_basis": rules["refund_basis"],
            "refund_amount_cents": paid,
            "citations_used": sorted(used),
        })

    async def analyze_price_adjustment(inputs):
        """比较演示购买价与当前价，并按政策期限判断是否可申请价保。

        价格数据来自项目 B 自己的本地演示凭证；政策或凭证缺失时返回 empty，
        LangGraph 会停止后续判断，避免把优惠金额、预计补差或退款说成已执行。
        """
        order = order_for(inputs.order_id)
        if not order:
            return await get_order(inputs)
        rules, used = {}, set()
        for hit in inputs.evidence:
            policy_keys = set(re.findall(r"(?<!\w)policy_key:\s*([a-zA-Z0-9_]+)(?!\w)", hit.snippet))
            if policy_keys != {order["policy_key"]}:
                continue
            for key in ("price_protection_days", "price_protection_basis"):
                matches = re.findall(r"(?<!\w)" + key + r":\s*([^\s]+)(?=\s|$)", hit.snippet)
                for value in matches:
                    if key in rules and rules[key] != value:
                        return ToolResult(status="empty", error_code="POLICY_CONFLICT", message="价保政策字段冲突，需要人工复核")
                    rules[key] = value
                    used.add(hit.citation_number)
        if not {"price_protection_days", "price_protection_basis"} <= rules.keys():
            return ToolResult(status="empty", error_code="PRICE_POLICY_INCOMPLETE", message="缺少价保期限或补差计算依据，无法判断")
        price = repository.price_adjustment(inputs.order_id, buyer_id)
        if not price or any(key not in price for key in ("purchase_price_cents", "current_price_cents", "price_drop_days_ago")):
            return ToolResult(status="empty", error_code="PRICE_FIELDS_MISSING", message="缺少购买价、当前价或降价时间凭证，无法比较")
        try:
            purchase = int(price["purchase_price_cents"])
            current = int(price["current_price_cents"])
            days = int(price["price_drop_days_ago"])
            protection_days = int(rules["price_protection_days"])
        except (TypeError, ValueError):
            return ToolResult(status="empty", error_code="PRICE_FIELDS_INVALID", message="价格凭证格式无效，需要人工复核")
        if min(purchase, current, days, protection_days) < 0:
            return ToolResult(status="empty", error_code="PRICE_FIELDS_INVALID", message="价格凭证不能为负数，需要人工复核")
        difference = purchase - current
        eligible = difference > 0 and days <= protection_days and rules["price_protection_basis"] == "purchase_price_minus_current_price"
        return ToolResult(status="success", data={
            "purchase_price_cents": purchase,
            "current_price_cents": current,
            "difference_cents": max(difference, 0),
            "price_drop_days_ago": days,
            "price_protection_days": protection_days,
            "eligible": eligible,
            "price_protection_basis": rules["price_protection_basis"],
            "citations_used": sorted(used),
        })

    definitions = [
        ("get_order", "查询服务端认证买家名下的一笔订单", OrderInput, get_order),
        ("retrieve_policy", "检索项目 B 本地该商品的政策文档", PolicyInput, retrieve),
        ("assess_after_sales", "根据实际订单和实际检索证据核对退换条件", AssessmentInput, assess),
        ("check_inventory", "查询演示商品的可用换货库存", OrderInput, inventory)]
    # B3 使用独立工具核对优惠，避免把退款或订单修改能力暴露给 Planner。
    definitions.append(("analyze_discount", "核对原价、优惠和实付金额，并返回退款依据", DiscountInput, analyze_discount))
    definitions.append(("analyze_price_adjustment", "比较购买价、当前价和降价时间，判断演示价保资格", PriceAdjustmentInput, analyze_price_adjustment))
    return {name: TypedCoreTool(name, description, schema, handler) for name, description, schema, handler in definitions}
