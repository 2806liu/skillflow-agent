"""任务级工具运行边界：按需 Skill、固定工具面、目标约束、可信引用及只读执行。"""

import asyncio
from dataclasses import asdict, dataclass
from time import perf_counter
from uuid import uuid4, uuid5, NAMESPACE_URL

from pydantic import ConfigDict, Field, ValidationError

from skillflow.contracts import Evidence

from .adapters.local_tools import fingerprint
from .budget import BudgetExceeded, BudgetTracker
from .contracts import AgentState, Contract, RuntimeEvent, SkillDocument, TaskMetadata, ToolCall, ToolObservation, ToolSpec
from .skills import SkillError


class BoundaryError(Exception):
    """边界拒绝与底层故障分开，禁止把拒绝动作交给业务处理器。"""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


class TargetScope(Contract):
    """由服务端从用户任务构造，不能由模型参数扩充；批量模式只用于明确的质量事件查询。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    explicit_order_ids: tuple[str, ...] = Field(default=(), max_length=50)
    keyword: str = Field(default="", max_length=80)
    quality_batch: bool = False


@dataclass(frozen=True)
class ToolSurface:
    """一次模型请求对应的工具集合快照，加载 Skill 后旧快照不会随之扩权。"""

    token: str
    model_call_id: str
    tool_specs: tuple[ToolSpec, ...]


@dataclass(frozen=True)
class EvidenceRecord:
    """原始 Evidence 留在可信存储中，额外绑定任务、买家、订单及事实版本。"""

    task_id: str
    buyer_id: str
    order_id: str
    order_hash: str
    policy_hash: str
    evidence_json: str


class ToolSession:
    """每 task 独享上下文、预算、证据和调用回执，避免跨会话共享可变执行状态。"""

    def __init__(self, metadata: TaskMetadata, scope: TargetScope, registry, skills, adapter, tasks, events,
                 budget: BudgetTracker | None = None, tool_timeout: float = 20, graph_managed: bool = False):
        self.metadata, self.scope = metadata, scope
        self.registry, self.skills, self.adapter = registry, skills, adapter
        self.tasks, self.events = tasks, events
        self.budget = budget or BudgetTracker(metadata.limits)
        self.tool_timeout = tool_timeout
        # S3 消息及状态由图保存；S2 独立诊断仍保存自己的快照。
        self.graph_managed = graph_managed
        self._loaded: dict[str, SkillDocument] = {}
        self._surfaces, self._calls, self._evidence, self._receipts = {}, {}, {}, {}
        # 完成门也核验空结果的来源，不能只核验成功收据。
        self._facts = {}
        self._candidates, self._scan_complete, self._expected_cursor = set(), False, None
        self._lock = asyncio.Lock()
        self.observations: list[ToolObservation] = []
        # 仅 S4 装配可信工单命令；模型及持久状态均不能提供这个处理器。
        self.write_handler = None

    @property
    def loaded(self) -> dict[str, SkillDocument]:
        """外部只取得深拷贝，不能通过修改列表获得额外工具。"""
        return {name: doc.model_copy(deep=True) for name, doc in self._loaded.items()}

    def offered_tools(self) -> list[ToolSpec]:
        """load_skill 常驻，其余取已加载技能并集，再过滤未开放能力。"""
        names = {"load_skill"}
        for document in self._loaded.values():
            names.update(document.tool_names)
        return self.registry.model_specs(list(names))

    def begin_turn(self, model_call_id: str) -> ToolSurface:
        """由模型调度器调用一次；同一响应内全部工具使用同一份快照。"""
        surface = ToolSurface(uuid4().hex, model_call_id, tuple(self.offered_tools()))
        self._surfaces[surface.token] = (model_call_id, frozenset(tool.name for tool in surface.tool_specs))
        return surface

    def dump_state(self):
        """S4 保存可校验 JSON：技能内容锁、证据、调用去重、事实及候选分页；不保存客户端。"""
        return {"version": "tool_session_v1", "task_id": self.metadata.task_id,
            "buyer_id": self.metadata.buyer_id, "session_id": self.metadata.session_id,
            "loaded": {k:v.model_dump(mode="json") for k,v in self._loaded.items()},
            "surfaces": {k:[v[0],sorted(v[1])] for k,v in self._surfaces.items()},
            "calls": {k:[v[0],v[1].model_dump(mode="json")] for k,v in self._calls.items()},
            "evidence": {k:asdict(v) for k,v in self._evidence.items()},
            "receipts": [[*k,v] for k,v in self._receipts.items()],
            "facts": [[*k,v] for k,v in self._facts.items()],
            "candidates": sorted(self._candidates), "scan_complete": self._scan_complete,
            "expected_cursor": self._expected_cursor,
            "observations": [v.model_dump(mode="json") for v in self.observations]}

    async def restore_state(self, value):
        """只从受控 Saver/业务回执恢复；重新核对身份及每份 Skill 的版本哈希。"""
        if (value.get("version"),value.get("task_id"),value.get("buyer_id"),value.get("session_id")) != (
                "tool_session_v1",self.metadata.task_id,self.metadata.buyer_id,self.metadata.session_id):
            raise BoundaryError("CHECKPOINT_INCOMPATIBLE")
        loaded={}
        for name,raw in value["loaded"].items():
            old=SkillDocument.model_validate(raw)
            loaded[name]=await self.skills.get_pinned(name,old.version,old.content_hash)
        self._loaded=loaded
        self._surfaces={k:(v[0],frozenset(v[1])) for k,v in value["surfaces"].items()}
        self._calls={k:(v[0],ToolObservation.model_validate(v[1])) for k,v in value["calls"].items()}
        self._evidence={k:EvidenceRecord(**v) for k,v in value["evidence"].items()}
        for saved in self._evidence.values():
            if (saved.task_id,saved.buyer_id)!=(self.metadata.task_id,self.metadata.buyer_id):
                raise BoundaryError("EVIDENCE_SCOPE_MISMATCH")
            Evidence.model_validate_json(saved.evidence_json)
        self._receipts={(name,order):v for name,order,v in value["receipts"]}
        self._facts={(name,order):v for name,order,v in value["facts"]}
        self._candidates=set(value["candidates"])
        self._scan_complete=value["scan_complete"]
        self._expected_cursor=value["expected_cursor"]
        self.observations=[ToolObservation.model_validate(v) for v in value["observations"]]

    async def dispatch(self, call: ToolCall, surface: ToolSurface) -> ToolObservation:
        """串行处理同 task 调用，先校验、占预算并记账，再执行只读适配器。"""
        async with self._lock:
            digest = fingerprint({"name": call.name, "arguments": call.arguments})
            previous = self._calls.get(call.call_id)
            if previous:
                if previous[0] == digest:
                    # 上次结果已有内存回执，重试只补齐诊断持久化，不能因审计失败而绕过记录。
                    await self._save_state()
                    await self._event("tool_replay", "replayed", {"stage": "S2", "tool": call.name,
                                      "tool_call_id": call.call_id, "invocation_id": previous[1].invocation_id,
                                      "result_ref": previous[1].result_ref})
                    return previous[1].model_copy(deep=True)
                return ToolObservation(invocation_id=uuid4().hex, tool_call_id=call.call_id,
                                       status="rejected", error_code="TOOL_CALL_CONFLICT")
            # S4 从已检查点化的 call_id 派生稳定 invocation_id；结果未知的重试仍扣一次尝试。
            namespace=getattr(self,'stable_call_namespace',None)
            invocation_id = uuid5(NAMESPACE_URL,namespace+':'+call.call_id).hex if namespace else uuid4().hex
            own_timer = not self.budget.is_running
            if own_timer:
                self.budget.start()
            started = perf_counter()
            details = {"stage": "S2", "tool": call.name, "tool_call_id": call.call_id,
                       "model_call_id": surface.model_call_id, "invocation_id": invocation_id,
                       "arguments_hash": digest}
            if self.graph_managed:
                details.update(stage="S3", arguments=call.arguments)
            try:
                self.budget.reserve_tool()
                # 调用前保存已占用预算；属于诊断快照，不提供检查点恢复。
                await self._save_state()
                await self._event("before_tool", "started", details)
                # 整个校验与执行阶段受超时约束，归属和前置事实读取也不能无限等待。
                # MCP 的十秒包含本次订单授权和连接等待；累计任务期限仍有更高优先级。
                call_timeout = (self.adapter.manager.query_timeout if call.name == 'get_logistics' and
                                hasattr(self.adapter, 'manager') else self.tool_timeout)
                data, status, code, message = await asyncio.wait_for(
                    self._execute(call, surface), min(call_timeout, self.budget.remaining_seconds()))
                observation = ToolObservation(invocation_id=invocation_id, tool_call_id=call.call_id,
                                              status=status, data=data, error_code=code, message=message,
                                              retryable=status == 'failed' and code in {'MCP_TIMEOUT', 'MCP_UNAVAILABLE'},
                                              result_ref="r_" + uuid4().hex if status in {"success", "empty"} else None)
            except (BoundaryError, SkillError, BudgetExceeded) as exc:
                observation = ToolObservation(invocation_id=invocation_id, tool_call_id=call.call_id,
                                              status="rejected", error_code=exc.code)
            except ValidationError:
                observation = ToolObservation(invocation_id=invocation_id, tool_call_id=call.call_id,
                                              status="rejected", error_code="INVALID_TOOL_ARGUMENTS")
            except asyncio.TimeoutError:
                code = ("TASK_TIMEOUT" if self.budget.snapshot().active_seconds >= self.metadata.limits.active_timeout_seconds
                        else "MCP_TIMEOUT" if call.name == 'get_logistics' and hasattr(self.adapter, 'manager') else "TOOL_TIMEOUT")
                observation = ToolObservation(invocation_id=invocation_id, tool_call_id=call.call_id,
                                              status="failed", error_code=code, retryable=True)
            finally:
                if own_timer:
                    self.budget.pause()
            # 基础设施异常不在这里吞掉；事件或快照失败必须向上报告。
            self._calls[call.call_id] = (digest, observation.model_copy(deep=True))
            self.observations.append(observation)
            await self._event("after_tool", observation.status, {**details, "result_ref": observation.result_ref},
                              observation.error_code, (perf_counter() - started) * 1000)
            await self._save_state()
            return observation.model_copy(deep=True)

    def checked_arguments(self, call: ToolCall, surface: ToolSurface):
        """检查公开 Schema、Skill 和本轮工具面；确认门与普通执行共用同一入口。"""
        definition = self.registry.resolve(call.name)
        if definition is None:
            raise BoundaryError("UNKNOWN_TOOL")
        args = definition.arguments.model_validate(call.arguments).model_dump()
        actual_surface = self._surfaces.get(surface.token)
        if actual_surface is None or actual_surface[0] != surface.model_call_id:
            raise BoundaryError("TOOL_SURFACE_INVALID")
        allowed = {"load_skill"}.union(*(set(doc.tool_names) for doc in self._loaded.values()))
        if call.name not in allowed:
            raise BoundaryError("SKILL_NOT_LOADED")
        if definition.unavailable_code:
            raise BoundaryError(definition.unavailable_code)
        if call.name not in actual_surface[1]:
            raise BoundaryError("TOOL_NOT_OFFERED")
        return args

    async def _execute(self, call: ToolCall, surface: ToolSurface):
        """公开参数校验和任务权限检查独立于旧业务 handler。"""
        args = self.checked_arguments(call, surface)
        definition = self.registry.resolve(call.name)
        if definition.effect != "read_only":
            if self.write_handler is None:
                raise BoundaryError("WRITE_NOT_ENABLED")
            return await self.write_handler(call.name,args,call.call_id)
        if call.name == "load_skill":
            return await self._load_skill(args)
        if call.name == "list_my_orders":
            return await self._list_orders(args)

        order_id = args["order_id"]
        owner = await self.adapter.owner(order_id)
        if owner is not None and owner != self.metadata.buyer_id:
            raise BoundaryError("ORDER_FORBIDDEN")
        self._require_target(order_id, call.name)
        order = await self.adapter.order(order_id)
        if order is None:
            return {}, "empty", "ORDER_NOT_FOUND", "未查询到本人目标订单。"
        order_hash = fingerprint(order)
        source_version = await self.adapter.source_version(call.name, order)
        for required in definition.requires:
            receipt = self._receipts.get((required, order_id))
            if receipt is None:
                raise BoundaryError("PRECONDITION_MISSING")
            if receipt["order_hash"] != order_hash:
                raise BoundaryError("ORDER_FACTS_CHANGED")
            if required != "get_order" and receipt["source_version"] != source_version:
                raise BoundaryError("SOURCE_VERSION_CHANGED")
        if "evidence_refs" in args:
            refs = args.pop("evidence_refs")
            if len(set(refs)) != len(refs):
                raise BoundaryError("INVALID_EVIDENCE_REF")
            evidence = []
            for ref in refs:
                saved = self._evidence.get(ref)
                if saved is None:
                    raise BoundaryError("EVIDENCE_NOT_FOUND")
                if (saved.task_id, saved.buyer_id, saved.order_id) != (
                        self.metadata.task_id, self.metadata.buyer_id, order_id):
                    raise BoundaryError("EVIDENCE_SCOPE_MISMATCH")
                if saved.order_hash != order_hash or saved.policy_hash != source_version:
                    raise BoundaryError("EVIDENCE_STALE")
                evidence.append(Evidence.model_validate_json(saved.evidence_json).model_dump())
            args["evidence"] = evidence

        timeout = min(self.tool_timeout, self.budget.remaining_seconds())
        try:
            result = await asyncio.wait_for(self.adapter.invoke(call.name, args), timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            raise
        except Exception:
            # 不向模型暴露底层异常原文，也不把失败规范化为成功。
            return {}, "failed", "TOOL_FAILED", "本地业务工具执行失败。"
        self.budget.remaining_seconds()
        current = await self.adapter.order(order_id)
        if fingerprint(current) != order_hash or await self.adapter.source_version(call.name, order) != source_version:
            raise BoundaryError("FACTS_CHANGED_DURING_CALL")
        data = dict(result.data)
        if call.name == 'get_logistics' and result.status in {'success', 'empty'}:
            # 执行段标识用于恢复时将旧结果标为历史，不是访问授权。
            data['execution_run_id'] = self.metadata.run_id
        if result.status in {"success", "empty"}:
            self._facts[(call.name, order_id)] = {"order_hash": order_hash, "source_version": source_version,
                                                  "result_hash": fingerprint(data), "execution_run_id": self.metadata.run_id}
        if result.status == "success":
            self._receipts[(call.name, order_id)] = {"order_hash": order_hash, "source_version": source_version}
            if call.name == "retrieve_policy":
                public_evidence = []
                for item in data.pop("evidence", []):
                    evidence = Evidence.model_validate(item)
                    ref = "e_" + uuid4().hex
                    self._evidence[ref] = EvidenceRecord(self.metadata.task_id, self.metadata.buyer_id,
                                                       order_id, order_hash, source_version, evidence.model_dump_json())
                    public_evidence.append({"ref": ref, **evidence.model_dump()})
                data["evidence"] = public_evidence
        return data, result.status, result.error_code, result.message

    async def verify_current(self):
        """S3 完成前重新核验订单及来源版本；发生变化则停止，不沿用过期证据。"""
        for (name, order_id), saved in self._facts.items():
            order = await self.adapter.order(order_id)
            if fingerprint(order) != saved["order_hash"]:
                raise BoundaryError("ORDER_FACTS_CHANGED")
            if await self.adapter.source_version(name, order) != saved["source_version"]:
                raise BoundaryError("SOURCE_VERSION_CHANGED")
            if name in {"check_inventory", "list_alternatives"}:
                # 库存不属于订单字段，完成前单独重查；不把稍早的库存描述为当前事实。
                result = await self.adapter.invoke(name, {"order_id": order_id})
                if fingerprint(result.data) != saved["result_hash"]:
                    raise BoundaryError("INVENTORY_CHANGED")

    def _require_target(self, order_id, tool_name):
        """唯一候选必须来自完整扫描；明确批量质量任务才允许逐单调查。"""
        if self.scope.explicit_order_ids:
            targets = set(self.scope.explicit_order_ids)
        else:
            if not self._scan_complete:
                raise BoundaryError("CANDIDATE_SCAN_INCOMPLETE")
            targets = self._candidates
        if order_id not in targets:
            raise BoundaryError("TARGET_NOT_ALLOWED")
        if len(targets) > 1:
            if not self.scope.quality_batch:
                raise BoundaryError("CLARIFICATION_REQUIRED")
            if tool_name not in {"get_order", "match_quality_incident", "list_alternatives"}:
                raise BoundaryError("BATCH_SCOPE_FORBIDDEN")

    async def _load_skill(self, args):
        """技能加载属于工具尝试；重复加载不重复放入 SOP，版本必须一致。"""
        document = await self.skills.load(args["name"], args["version"])
        old = self._loaded.get(document.name)
        if old and (old.version, old.content_hash) != (document.version, document.content_hash):
            raise SkillError("SKILL_VERSION_INCOMPATIBLE")
        proposed = {**self._loaded, document.name: document}
        proposed_names = {"load_skill"}.union(*(set(doc.tool_names) for doc in proposed.values()))
        data = {"name": document.name, "version": document.version, "content_hash": document.content_hash,
                "already_loaded": old is not None,
                "opened_tools": [tool.name for tool in self.registry.model_specs(list(proposed_names))],
                "unavailable_tools": {name: definition.unavailable_code for name in document.tool_names
                                      if (definition := self.registry.resolve(name)).unavailable_code},
                "completion_conditions": document.completion_conditions}
        if old is None:
            data["instructions"] = document.instructions
        # 规范和版本记录成功后再改变授权；超时或审计失败时不能获得额外工具。
        await self._event("load_skill", "prepared", {"stage": "S2", **data, "reason": args["reason"]})
        self._loaded[document.name] = document
        return data, "success", None, "已加载技能规范；新开放业务工具从下一轮工具面生效。"

    async def _list_orders(self, args):
        """连续读取同一任务关键词的所有分页，防止跳页制造伪唯一候选。"""
        if args["keyword"].casefold() != self.scope.keyword.casefold():
            raise BoundaryError("TARGET_QUERY_MISMATCH")
        cursor = args["after_order_id"]
        if cursor is not None and (self._scan_complete or cursor != self._expected_cursor):
            raise BoundaryError("PAGINATION_CURSOR_INVALID")
        args["order_ids"] = self.scope.explicit_order_ids or None
        try:
            result = await asyncio.wait_for(self.adapter.invoke("list_my_orders", args),
                                            min(self.tool_timeout, self.budget.remaining_seconds()))
        except (asyncio.TimeoutError, asyncio.CancelledError):
            raise
        except Exception:
            return {}, "failed", "TOOL_FAILED", "订单候选查询失败。"
        self.budget.remaining_seconds()
        if cursor is None:
            self._candidates.clear()
        self._candidates.update(row["order_id"] for row in result.data["items"])
        self._scan_complete = not result.data["has_more"]
        self._expected_cursor = result.data["next_after_order_id"]
        data = {**result.data, "discovery_complete": self._scan_complete,
                "candidate_count": len(self._candidates),
                "needs_clarification": self._scan_complete and len(self._candidates) > 1 and not self.scope.quality_batch}
        return data, result.status, result.error_code, result.message

    def completion_status(self, skill_name: str, order_id: str) -> dict[str, bool]:
        """S2 只提供固定条件的收据检查，S3 完成门还须核验当前事实与业务目标。"""
        names = {"order_verified": "get_order", "policy_assessed": "assess_after_sales",
                 "inventory_checked": "check_inventory", "discount_assessed": "analyze_discount",
                 "price_assessed": "analyze_price_adjustment", "incident_assessed": "match_quality_incident",
                 "logistics_verified": "get_logistics"}
        return {condition: (names[condition], order_id) in self._receipts
                for condition in self._loaded[skill_name].completion_conditions}

    async def _event(self, node, status, details, code=None, duration_ms=0):
        """记录实际 Skill、版本、开放工具和错误；基础设施失败向上传播。"""
        if self.graph_managed:
            details = {**details, "stage": "S5" if self.metadata.engine_version.startswith("tool_loop_s5_") else "S4" if self.metadata.engine_version.startswith("tool_loop_s4_") else "S3"}
        await self.events.append(RuntimeEvent(metadata=self.metadata, node=node, status=status,
                                              budget=self.budget.snapshot(), error_code=code,
                                              duration_ms=duration_ms, details=details))

    async def _save_state(self):
        """存储诊断状态，原始证据仓和工具面不提供恢复，S4 再接实际检查点。"""
        if self.graph_managed:
            return
        await self.tasks.save_state(AgentState(metadata=self.metadata, goal="S2 技能与只读工具独立验收",
            constraints=self.scope.model_dump(mode="json"), loaded_skills={name: doc.version for name, doc in self._loaded.items()},
            observations=[ob.model_copy(deep=True) for ob in self.observations], budget=self.budget.snapshot(), status="running"))

    async def finish_diagnostic(self):
        """明确结束独立脚本验收；这是诊断记录，不是模型判断业务任务完成。"""
        async with self._lock:
            self.budget.pause()
            await self.tasks.save_state(AgentState(metadata=self.metadata, goal="S2 技能与只读工具独立验收",
                constraints=self.scope.model_dump(mode="json"), loaded_skills={name: doc.version for name, doc in self._loaded.items()},
                observations=[ob.model_copy(deep=True) for ob in self.observations], budget=self.budget.snapshot(),
                status="runtime_ready", stop_reason="S2_DIAGNOSTIC_ONLY"))
