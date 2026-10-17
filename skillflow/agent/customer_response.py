"""客服表达层：只格式化已经核验的事实，不计算资格、不补造物流或办理结果。"""

from datetime import datetime, timedelta, timezone


def readable_time(value):
    """可信 ISO 时间转换为北京时间；无法解析时保留原值，不推断日期。"""
    if not value:
        return '未提供'
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            return value
        return parsed.astimezone(timezone(timedelta(hours=8))).strftime('%Y年%m月%d日 %H:%M')
    except (ValueError, TypeError):
        return value


def logistics_message(order_id, value, empty=False):
    """聊天呈现状态、更新时间和最近轨迹；完整轨迹、版本及查询时间留在右侧证据。"""
    if empty:
        return (f'订单 {order_id} 暂无配送物流记录，目前还不能确认配送进度。'
                '您可以核对订单号后再查询。以上是本项目的演示查询结果。')
    label = {'DELIVERED': '已签收', 'IN_TRANSIT': '运输中', 'EXCEPTION': '配送异常', 'UNKNOWN': '状态未知'}[value['logistics_status']]
    text = f'订单 {order_id} 的配送记录显示：{label}。来源更新时间为 {readable_time(value["updated_at"])}（北京时间）。'
    events = value['events'][-3:]
    if events:
        text += '\n' + ('最近三条运输记录：' if len(value['events']) > 3 else '运输记录：')
        text += ''.join('\n• ' + readable_time(event['time']) + '：' + event['description'] for event in events)
    text += '\n以上来自本项目的模拟配送记录，完整来源与本次查询时间可在右侧查看。'
    return text


def logistics_failure(order_id, code):
    """将技术故障转为明确的客服说明；不承诺自动重试、人工转接或实际查到物流。"""
    if code == 'MCP_TIMEOUT':
        reason = '物流服务暂时没有在限定时间内返回结果'
    elif code == 'MCP_UNAVAILABLE':
        reason = '物流服务暂时无法连接'
    elif code == 'MCP_ACCESS_DENIED':
        reason = '物流服务未允许访问这笔订单'
    else:
        reason = '物流服务暂时未能提供可核验的结果'
    return f'订单 {order_id} 的物流查询未完成：{reason}，因此目前无法确认配送进度。您可以稍后重新查询。'


def render_customer_response(finding, proposal):
    """仅选择通过校验的对应事实表达；失败不说已完成，物流查询不附加退款话术。"""
    facts = finding.customer_facts if finding.customer_facts else finding.facts
    if len(facts) != len(finding.facts):
        raise ValueError('ANSWER_PROPOSAL_INVALID')
    if finding.status == 'clarify':
        opening = '我需要先和您确认一下：'
    elif finding.reason == 'ORDER_FORBIDDEN':
        opening = '抱歉，这笔订单暂时无法为您查询。'
    elif finding.status == 'completed':
        opening = '已帮您核对，结果如下：'
    else:
        opening = '目前查到的情况如下，部分信息还需要进一步核实：'
    if proposal.tone == 'empathic' and finding.reason != 'ORDER_FORBIDDEN':
        opening = '理解您的担心。' + opening
    paragraphs = [opening, *(facts[i] for i in proposal.fact_indexes)]
    if finding.after_sales:
        paragraphs.append('目前为您完成的是查询核对，尚未提交售后申请。' if finding.status == 'completed'
                          else '目前尚未提交售后申请。')
    return '\n\n'.join(paragraphs)
