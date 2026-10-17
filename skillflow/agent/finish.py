"""独立完成门与可信事实渲染：依据原始任务条件和执行收据，不采用模型自由结论。"""

from dataclasses import dataclass, field

from .goals import TaskGoal


@dataclass
class Finding:
    """可展示事实与内部缺项分开；缺项最多触发一次受预算约束的补查。"""

    status: str = "completed"
    reason: str = "GOAL_SATISFIED"
    facts: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    # 同索引对应两种表达：facts 供审计与提案校验，customer_facts 供客服聊天展示。
    customer_facts: list[str] = field(default_factory=list)
    after_sales: bool = False

    def add_fact(self, fact: str, customer: str | None = None):
        """可信业务结论只生成一次，客服表达不能改变事实数量和引用关系。"""
        self.facts.append(fact)
        self.customer_facts.append(customer if customer is not None else fact)


def inspect_goal(goal: TaskGoal, records: list[dict], logistics_enabled: bool = False) -> Finding:
    """只从当前任务的真实观察取证；空政策、缺批次分别表达为未完成或待人工。"""
    result = Finding()
    result.after_sales = bool(set(goal.requirements) & {'returns', 'incident', 'discount', 'price'})
    # S4 恢复复核可将旧证据标记失效；保留审计但不再参与任务完成判断。
    valid = [r for r in records if not r.get("invalidated") and r["observation"]["status"] in {"success", "empty"}]
    by = {(r["name"], r["arguments"].get("order_id")): r for r in valid}

    def take(name, order_id=None):
        record = by.get((name, order_id))
        if record and record["observation"].get("result_ref") not in result.refs:
            result.refs.append(record["observation"]["result_ref"])
        return record["observation"] if record else None

    targets = list(goal.scope.explicit_order_ids)
    if not targets:
        pages = [r for r in valid if r["name"] == "list_my_orders"]
        if not pages or not pages[-1]["observation"]["data"].get("discovery_complete"):
            result.missing.append("完成本人订单候选分页查询")
            return result
        # 从最后一次起始页开始计算，禁止重复分页污染候选集。
        start = max(i for i, row in enumerate(pages) if not row["arguments"].get("after_order_id"))
        targets = sorted({item["order_id"] for row in pages[start:]
                          for item in row["observation"]["data"].get("items", [])})
        for row in pages[start:]:
            result.refs.append(row["observation"]["result_ref"])
    if not targets:
        result.status, result.reason = "needs_manual_review", "ORDER_NOT_FOUND"
        result.add_fact("未查询到符合当前关键词的本人订单，请核对商品名称或提供订单号。",
                        '暂时没有找到符合描述的本人订单。麻烦您核对商品名称，或提供具体订单号，我再帮您查询。')
        return result
    if len(targets) > 1 and not goal.scope.quality_batch:
        result.status, result.reason, result.candidates = "clarify", "CLARIFICATION_REQUIRED", targets
        result.add_fact("找到多笔候选订单：" + "、".join(targets) + "。请选择要调查的一笔，例如“选择 " + targets[0] + "”。",
                        '为您找到多笔相关订单：' + '、'.join(targets) + '。您想查询哪一笔？可以回复“选择 ' + targets[0] + '”。')
        return result
    for order_id in targets:
        order = take("get_order", order_id)
        if not order:
            if logistics_enabled and any(r['name'] == 'get_order' and r['arguments'].get('order_id') == order_id
                    and r['observation'].get('error_code') == 'ORDER_FORBIDDEN' for r in records):
                # S5 新入口明确呈现授权拒绝，不将权限不足误写为物流缺失或可补查事实。
                result.status, result.reason = 'needs_manual_review', 'ORDER_FORBIDDEN'
                result.add_fact(f'{order_id}：当前登录买家无权访问此订单，已停止物流调查。',
                    f'当前登录账号无权访问订单 {order_id}，我无法继续查询其物流。请使用订单所属账号登录后再查询。')
                continue
            result.missing.append(order_id + " 查询本人订单")
            continue
        if order["status"] == "empty":
            result.status, result.reason = "needs_manual_review", "ORDER_NOT_FOUND"
            result.add_fact(order_id + "：未找到本人订单，请核对单号。",
                            f'暂时没有查到您名下的订单 {order_id}。麻烦您核对一下订单号。')
            continue
        data = order["data"]["order"]
        result.add_fact(f"{order_id}：{data['product']}，签收 {data['delivered_days_ago']} 天，实付 {data['paid_cents']/100:.2f} 元。",
            f"已查到订单 {order_id}，商品是{data['product']}，实付 {data['paid_cents']/100:.2f} 元，签收 {data['delivered_days_ago']} 天。")
        for requirement, tool in (("returns", "assess_after_sales"), ("discount", "analyze_discount"),
                                  ("price", "analyze_price_adjustment"), ("incident", "match_quality_incident"),
                                  ("alternatives", "list_alternatives"), ("logistics", "get_logistics")):
            if requirement not in goal.requirements:
                continue
            observation = take(tool, order_id)
            policy = take("retrieve_policy", order_id) if requirement in {"returns", "discount", "price"} else None
            if not observation:
                if policy and policy["status"] == "empty":
                    result.status, result.reason = "needs_manual_review", policy["error_code"] or "POLICY_MISSING"
                    result.add_fact(f"{order_id}：缺少可用政策证据，无法判断{ {'returns':'退换资格','discount':'优惠金额','price':'价保资格'}[requirement]}，需要人工核实。",
                        f"订单 {order_id} 目前缺少可用政策，暂时无法确认{ {'returns':'退换资格','discount':'优惠金额','price':'价保资格'}[requirement]}。建议联系店铺人工客服核实；当前并未自动转接。")
                elif requirement == "logistics":
                    if logistics_enabled:
                        # 故障观察也参与解释失败，但绝不作为已完成物流查询的成功证据。
                        attempts = [r for r in records if not r.get('invalidated') and r['name'] == tool
                                    and r['arguments'].get('order_id') == order_id]
                        if attempts:
                            code = attempts[-1]['observation'].get('error_code') or 'MCP_QUERY_INCOMPLETE'
                            result.status, result.reason = 'needs_manual_review', code
                            from .customer_response import logistics_failure
                            result.add_fact(f'{order_id}：物流查询未完成，原因 {code}；无法确认配送进度。',
                                            logistics_failure(order_id, code))
                        else:
                            result.missing.append(order_id + ' get_logistics')
                    else:
                        result.status, result.reason = "unsupported", "MCP_NOT_READY"
                        result.add_fact("物流 MCP 尚未开放，无法查询配送进度。", '当前服务暂未开放配送物流查询，暂时无法为您核实配送进度。')
                else:
                    result.missing.append(order_id + " " + tool)
                continue
            if observation["status"] == "empty" and requirement not in {"alternatives", "logistics"}:
                result.status, result.reason = "needs_manual_review", observation["error_code"] or "EVIDENCE_INCOMPLETE"
                result.add_fact(f"{order_id}：{observation['message'] or '缺少判断所需证据，需要人工核实。'}")
                continue
            value = observation["data"]
            if requirement == 'logistics':
                from .customer_response import logistics_message
                # 事实块仅由严格校验后的字段构造，模型不能自行增加轨迹或送达承诺。
                if observation['status'] == 'empty':
                    result.add_fact(f'{order_id}：暂无配送物流记录（项目 B 模拟数据）；查询时间 {value["queried_at"]}。',
                                    logistics_message(order_id, value, empty=True))
                else:
                    label = {'DELIVERED': '已签收', 'IN_TRANSIT': '运输中', 'EXCEPTION': '配送异常', 'UNKNOWN': '状态未知'}[value['logistics_status']]
                    events = '；'.join(e['time'] + ' ' + e['description'] for e in value['events']) or '暂无运输事件'
                    result.add_fact(f'{order_id}：配送状态 {label}（项目 B 模拟数据）；来源更新时间 {value["updated_at"]}；来源版本 {value["source_version"]}；查询时间 {value["queried_at"]}。运输记录：{events}。',
                                    logistics_message(order_id, value))
            elif requirement == "returns":
                result.add_fact(f"{order_id}：退货资格{'符合' if value['return_eligible'] else '不符合'}；换货资格{'符合' if value['exchange_eligible'] else '不符合'}。政策退货期限 {value['return_window_days']} 天、换货期限 {value['exchange_window_days']} 天。",
                    f"根据查到的店铺政策，订单 {order_id} 的退货资格{'符合' if value['return_eligible'] else '不符合'}，换货资格{'符合' if value['exchange_eligible'] else '不符合'}。退货和换货期限分别为 {value['return_window_days']} 天、{value['exchange_window_days']} 天。")
                # 与退换 Skill SOP 一致：资格成功核验后记录库存，库存不改变资格。
                inventory = take("check_inventory", order_id)
                if not inventory or inventory["status"] != "success":
                    result.missing.append(order_id + " check_inventory")
                else:
                    available = inventory["data"]["available"]
                    result.add_fact(f"{order_id}：当前可用库存 {available} 件。" + ("缺货，暂时不能换货。" if available == 0 and value["exchange_eligible"] else "库存不改变上述资格结论。"),
                        f"这款商品当前可用库存 {available} 件。" + ('很抱歉，目前缺货，暂时不能换货。' if available == 0 and value['exchange_eligible'] else '库存情况与退换资格分别核验，请以以上资格结论为准。'))
            elif requirement == "incident":
                decision = {"AFFECTED": "处于已审核事件影响范围", "NOT_AFFECTED": "不在该事件已审核批次范围", "UNVERIFIED": "无法核实是否受影响"}[value["decision"]]
                result.add_fact(f"{order_id}：{decision}（{value['decision']}）；事件 {value['event_id'] or '暂无可匹配事件'}，版本 {value['event_version'] or '暂无'}；退货资格{'符合' if value['return_eligible'] else '未确认符合'}；原因 {value['reason_code']}。" + ("需要人工核实。" if value["manual_review"] else ""),
                    f"已按可信订单批次和已审核事件核对：订单 {order_id} {decision}，退货资格{'符合' if value['return_eligible'] else '尚未确认符合'}。" + ('这部分还需要人工核实，暂时不能给您肯定结论。' if value['manual_review'] else '具体事件版本及核对依据可在右侧查看。'))
                if value["manual_review"]:
                    result.status, result.reason = "needs_manual_review", value["reason_code"]
            elif requirement == "alternatives":
                items = value.get("items", [])
                result.add_fact(f"{order_id}：" + ("在售有库存替代候选：" + "、".join(item.get("name", item.get("sku", "")) for item in items) if items else "暂无在售有库存替代候选") + "；候选不代表安全认证。",
                    ('为您查到以下在售且有库存的替代候选：' + '、'.join(item.get('name', item.get('sku', '')) for item in items) + '。' if items else '暂时没有查到在售且有库存的替代候选。') + '这些是商品目录候选，不代表已获得安全认证。')
            elif requirement == "discount":
                result.add_fact(f"{order_id}：优惠金额{'一致' if value['consistent'] else '存在差异'}；按政策预计实付 {value['expected_paid_cents']/100:.2f} 元，差额 {value['difference_cents']/100:.2f} 元。",
                    f"已核对订单 {order_id} 的优惠金额，结果{'一致' if value['consistent'] else '存在差异'}。按政策预计实付 {value['expected_paid_cents']/100:.2f} 元，核对差额为 {value['difference_cents']/100:.2f} 元。")
            elif requirement == "price":
                result.add_fact(f"{order_id}：价保资格{'符合' if value['eligible'] else '不符合'}，凭证价差 {value['difference_cents']/100:.2f} 元。",
                    f"根据已查到的政策和价格凭证，订单 {order_id} 的价保资格{'符合' if value['eligible'] else '不符合'}，凭证价差为 {value['difference_cents']/100:.2f} 元。")
    result.refs = list(dict.fromkeys(result.refs))
    return result


def render_facts(finding: Finding, proposal) -> str:
    """索引及引用需完整匹配；未通过校验的模型文本永远不展示为业务事实。"""
    if sorted(proposal.fact_indexes) != list(range(len(finding.facts))) or sorted(proposal.result_refs) != sorted(finding.refs):
        raise ValueError("ANSWER_PROPOSAL_INVALID")
    # 校验通过后才进入客服表达；原始事实、版本、原因码和引用继续保留在执行证据中。
    from .customer_response import render_customer_response
    return render_customer_response(finding, proposal)
