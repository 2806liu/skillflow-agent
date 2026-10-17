/* 聊天进度组件：只描述收到的真实事件；动画与计时不代表已执行工具或已取得业务结果。 */
(() => {
  'use strict';
  let active = null;
  const toolText = {
    load_skill:'正在准备所需的查询步骤…', list_my_orders:'正在查找您名下的相关订单…',
    get_order:'正在核对您的订单信息…', retrieve_policy:'正在查找适用的店铺政策…',
    assess_after_sales:'正在核对这笔订单的退换条件…', check_inventory:'正在查询商品库存…',
    match_quality_incident:'正在核对订单批次和已审核的质量事件…', list_alternatives:'正在查找在售的替代商品…',
    analyze_discount:'正在核对订单金额和优惠信息…', analyze_price_adjustment:'正在核对价保政策和价格凭证…',
    get_logistics:'正在查询这笔订单的配送物流…', create_ticket_draft:'正在准备售后申请草稿…',
    submit_ticket:'正在核对申请办理结果…'
  };
  function create(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function start(id, text = '已收到您的问题，正在为您处理，请稍等…') {
    // SSE 重连沿用原气泡及计时，避免多出“客服正在输入”或重置等待时长。
    if (active?.id === id) return;
    stop();
    const chat = document.getElementById('chat');
    const box = create('div', undefined, 'bubble assistant pending');
    box.dataset.progressId = id;
    box.setAttribute('aria-busy', 'true');
    box.append(create('span', 'SkillFlow', 'role'));
    const status = create('div', text, 'pending-status');
    status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
    const dots = create('span', undefined, 'typing-dots'); dots.setAttribute('aria-hidden', 'true');
    for (let i = 0; i < 3; i++) dots.append(create('i'));
    const meta = create('div', undefined, 'pending-meta'); meta.setAttribute('aria-hidden', 'true');
    const elapsed = create('span', '已等待 0 秒');
    const hint = create('span', '结果返回后会在这里显示。');
    meta.append(elapsed, hint); box.append(status, dots, meta); chat.append(box);
    active = {id, box, status, elapsed, hint, started: performance.now(), timer: null};
    chat.scrollTop = chat.scrollHeight;
    active.timer = setInterval(() => {
      if (!active || active.id !== id) return;
      const seconds = Math.floor((performance.now() - active.started) / 1000);
      active.elapsed.textContent = `已等待 ${seconds} 秒`;
      if (seconds >= 12) active.hint.textContent = '还在处理中，您无需重复发送。';
    }, 1000);
  }
  function setText(text) {
    if (active && active.status.textContent !== text) active.status.textContent = text;
  }
  function event(kind, value) {
    if (!active) return;
    if (kind === 'status') {
      setText(value.code === 'INCIDENT_INVESTIGATING' ? '正在核对您的订单与质量事件…' : '请求已受理，正在等待处理…');
      return;
    }
    if (kind === 'tool_start') { setText(toolText[value.tool] || '正在查询相关信息…'); return; }
    if (kind !== 'trace') return;
    const node = value.node, detail = value.details || {}, tool = value.tool || detail.tool;
    if (node === 'agent_before_model') {
      setText(detail.purpose === 'answer_proposal' ? '正在整理查询结果，马上为您回复…'
        : detail.purpose === 'task_goal' ? '正在理解您的问题，确认查询内容…' : '正在核对已有信息并安排下一步查询…');
    } else if (node === 'agent_before_tool' || node === 'execute' && value.status === 'running') {
      setText(toolText[tool] || '正在查询相关信息…');
    } else if (node === 'agent_after_tool' || node === 'agent_tool_observation') {
      const code = detail.error_code || value.details?.result?.error_code;
      setText(code === 'ORDER_FORBIDDEN' ? '这笔订单的访问权限未通过，正在整理说明…'
        : value.status === 'failed' ? '本次查询暂未成功，正在确认后续处理方式…' : '已收到查询结果，正在继续核对…');
    } else if (node === 'agent_finish_gate' || node === 'agent_answer_proposal') {
      setText('正在检查并整理本次查询结果…');
    }
  }
  function reconnect() {
    setText('连接暂时中断，正在恢复已有处理进度，请稍等…');
  }
  function stop(id) {
    // 旧请求迟到的清理动作不能删除后来请求的状态；结束时必须释放定时器。
    if (!active || id !== undefined && active.id !== id) return;
    clearInterval(active.timer);
    active.box.remove(); active = null;
  }
  window.SkillFlowChatProgress = Object.freeze({start, event, reconnect, stop});
})();
