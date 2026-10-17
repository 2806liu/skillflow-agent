# 订单配送物流 SOP

## 步骤
先明确本人订单，必要时查询候选并澄清，再 get_order 核验。物流能力可用后才调用 get_logistics 查询配送进度。

## 当前能力边界
S2 仅交付本规范和注册声明，get_logistics 待 S5 stdio MCP 实现。工具未就绪时明确 MCP_NOT_READY；不得编造运输轨迹，不把售后工单当成退货运单。

## 完成条件
logistics_verified：实际 MCP 返回该本人订单的配送记录，或明确无记录。当前能力未实现时不能宣称物流调查已完成。
