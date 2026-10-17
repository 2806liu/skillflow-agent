# 价保申请 SOP（S4）

get_order 后 retrieve_policy，将可信 evidence.ref 交给 analyze_price_adjustment。价差、时间与凭证以工具结果为准。
仅在本轮明确申请、固定 application 为 price_adjustment 且资格符合时 create_ticket_draft，再 submit_ticket。运行时暂停等待确认；价格或政策变化必须重新核验并展示新卡片。提交工单不执行转账或退款。
