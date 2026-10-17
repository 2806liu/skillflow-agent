"""本地恢复基础开销测量：只读加载与校验，不认领任务、不运行模型、不推进图。"""

import asyncio

from skillflow.agent.budget import BudgetTracker
from skillflow.agent.contracts import AgentState, TaskMetadata, BudgetUsage
from skillflow.agent.goals import TaskGoal
from skillflow.agent.persistent_graph import build_persistent_graph


async def load_recovery_state(service,task_id,buyer_id,session_id):
    """覆盖归属/版本、实际检查点、Skill哈希、证据与工具回执反序列化校验。"""
    owned=await asyncio.to_thread(service.ledger.owned,task_id,buyer_id,session_id)
    metadata=TaskMetadata.model_validate(owned['metadata'])
    service.check_version(metadata)
    async with service.checkpoints.open() as saver:
        snapshot=await build_persistent_graph(saver).aget_state(service.graph_config(metadata))
    state=AgentState.model_validate(snapshot.values['payload'])
    goal=TaskGoal.model_validate(state.constraints['task_goal'])
    budget=BudgetTracker(metadata.limits,BudgetUsage.model_validate(owned['budget']))
    session=service.tools.graph_session(metadata,goal.scope,budget,service.events)
    await session.restore_state(state.constraints['_tool_state'])
    return {'checkpoint_id':snapshot.config['configurable']['checkpoint_id'],
            'tool_receipts':len(session.dump_state()['receipts'])}
