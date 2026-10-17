# 优惠争议核对 SOP

## 步骤
明确本人目标订单；必要时 list_my_orders 并澄清。依次 get_order、retrieve_policy、analyze_discount。分析仅接受当前任务、同订单的可信 evidence_refs。

## 约束与失败
原价、优惠、实付必须来自订单；退款依据来自政策。缺少字段、规则冲突或证据过期时说明不足，不自行算出无依据结论。查询不等于申请，不执行退款。

## 完成条件
discount_assessed：工具成功返回金额一致性和政策依据。引用结果解释差额，保留分与元单位，不声称金额已支付或退回。
