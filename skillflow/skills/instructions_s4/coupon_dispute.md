# 优惠争议 SOP（S4）

get_order 后 retrieve_policy，使用当前任务的 evidence.ref 调用 analyze_discount。金额由工具确定，不能重新扣除已分摊优惠。
仅在本轮明确申请、固定 application 为 discount_dispute 且结果存在差异时 create_ticket_draft，随后 submit_ticket 提出待确认动作；服务端卡片确认后才能提交。金额一致、政策缺失或结果不足时停止申请并说明原因。
