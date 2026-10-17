"""工具重复调用策略：补齐同订单依赖属于进展；权限拒绝或成功工具不因此放行。"""

from .adapters.local_tools import fingerprint


def duplicate_blocked(call, records, requires=()):
    """只决定是否允许重新校验；真实参数、权限、证据版本和预算仍由执行器核验。

    前置条件拒绝后，只有同订单的必要工具产生了新的成功观察，且全部依赖
    已成功时才允许重新尝试。历史失效记录不授予权限；同call_id的回执重放
    继续由原执行器/持久图负责，这里仅处理新call_id但相同工具参数的请求。
    """
    key=fingerprint({'name':call.name,'args':call.arguments})
    valid=[r for r in records if not r.get('invalidated')]
    previous=[(i,r) for i,r in enumerate(valid)
        if fingerprint({'name':r['name'],'args':r['arguments']})==key]
    if not previous: return False
    last_index,last=previous[-1]
    # 重复保护自身生成的拒绝不抹掉原始拒绝原因，但仍要求最近一次尝试之后有进展。
    original=next((r['observation'] for _,r in reversed(previous)
        if r['observation'].get('error_code')!='DUPLICATE_NO_PROGRESS'),last['observation'])
    order_id=call.arguments.get('order_id')
    if requires and order_id and original.get('error_code')=='PRECONDITION_MISSING':
        successful=[(i,r) for i,r in enumerate(valid) if r['observation'].get('status')=='success'
            and r['arguments'].get('order_id')==order_id and r['name'] in requires]
        if (set(requires)<={r['name'] for _,r in successful}
                and any(i>last_index for i,_ in successful)):
            return False
    # 临时故障保持最多两次尝试；同状态无进展、权限/参数拒绝和已成功调用继续拦截。
    observation=last['observation']
    return (len(previous)>=2 or observation.get('status')!='failed'
            or not observation.get('retryable'))
