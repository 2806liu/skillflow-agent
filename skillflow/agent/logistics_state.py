"""物流恢复时效策略：只使旧执行段物流失效，不清除其他可信调查事实。"""

def invalidate_historical_logistics(state, context):
    """不隐式查物流；标历史、清该项收据，让模型通过新 call_id 显式补查并扣预算。"""
    if not getattr(context.tools, 'logistics_enabled', False):
        return
    changed = False
    for row in state.constraints.get('tool_records', []):
        ob = row['observation']
        source_run = ob['data'].get('execution_run_id', row.get('execution_run_id'))
        if (row['name'] == 'get_logistics' and not row.get('invalidated') and
                source_run != context.metadata.run_id):
            row['invalidated'], row['historical_reason'] = True, 'LOGISTICS_PREVIOUS_EXECUTION'
            changed = True
    # 事务回执可能领先于图；即使记录未进入图，也不能将其旧物流事实视为当前查询。
    session = context.session
    for key, value in list(session._facts.items()):
        if key[0] == 'get_logistics' and value.get('execution_run_id') != context.metadata.run_id:
            session._facts.pop(key, None)
            session._receipts.pop(key, None)
            changed = True
    if changed and state.constraints.get('_logistics_refresh_run') != context.metadata.run_id:
        state.constraints['_logistics_refresh_run'] = context.metadata.run_id
        # 可能仍处于同一响应的工具批次内，提示留到下一模型节点，不能插断消息配对。
        state.constraints['_logistics_refresh_notice'] = True
