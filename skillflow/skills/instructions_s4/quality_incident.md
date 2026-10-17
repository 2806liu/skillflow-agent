# 质量事件调查与退货 SOP（S4）

1. 查询具体本人订单；未给单号可 list_my_orders 分页发现候选，多候选先澄清。明确批量需求只允许只读。
2. get_order 后 match_quality_incident，保留可信批次三态、事件版本与原因。UNVERIFIED 不等于不受影响；范围外不等于安全认证。
3. 用户要求替代品时调用 list_alternatives，只描述在售有库存候选。
4. 本轮明确申请且固定目标 application 为 return 时，完成事件资格调查后调用 create_ticket_draft，再 submit_ticket。提交工具会暂停等待服务端确认，不能绕过卡片或传 confirmed。
5. 事件资格不符合或待人工时不创建草稿；不得用普通店铺政策覆盖事件结论。版本变化由运行时复核并要求重新确认。

事件退货申请的资格依据是 match_quality_incident 的 return_eligible，不依赖 retrieve_policy / assess_after_sales。工具确认受影响且退货资格符合后，即可按本轮明确申请创建草稿并提出提交；不要因普通政策未登记而忽略已经核实的事件规则。
