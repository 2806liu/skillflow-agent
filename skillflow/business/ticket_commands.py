"""工单领域命令：复用业务 handler 与工单仓库；不调用模型、旧 Agent 图或 SSE。"""

import asyncio
from contextlib import closing
import json
import threading
from uuid import uuid4

from skillflow.agent.adapters.local_tools import LocalBusinessAdapter, fingerprint
from skillflow.agent.durable_store import TaskConflict
from .sqlite_tickets import SqliteTicketStore

LABELS={'return':'退货','exchange':'换货','discount_dispute':'优惠争议','price_adjustment':'价保补差'}


class TicketCommandService:
    """新旧工单共享存储事务；新 Agent 额外绑定任务、确认卡片及稳定操作回执。"""

    def __init__(self, ledger, repository, policies, confirmation_seconds=180):
        self.ledger,self.repository,self.policies=ledger,repository,policies
        self.store,self.tickets=ledger.store,SqliteTicketStore(ledger.store)
        self.confirmation_seconds=confirmation_seconds
        # 本机政策重建与命令复核使用同一锁；数据库事实由 IMMEDIATE 写锁保护。
        self.policy_lock=threading.RLock()

    def _facts(self, metadata, application):
        """持数据库写锁时执行短时本地规则复核；没有网络、模型、旧图或用户等待。"""
        order_id,kind=application['order_id'],application['request_type']
        owner=self.repository.owner_of_order(order_id)
        if owner is not None and owner!=metadata.buyer_id:
            raise TaskConflict('ORDER_FORBIDDEN')
        order=self.repository.get_order(order_id,metadata.buyer_id)
        if not order:
            raise TaskConflict('ORDER_NOT_FOUND')
        adapter=LocalBusinessAdapter(self.repository,self.policies,metadata.buyer_id)
        fact={'order':order,'request_type':kind}
        ticket_meta={}
        if order.get('category')=='power_bank':
            if kind!='return':
                raise TaskConflict('INCIDENT_RETURN_ONLY')
            assessment=self.repository.evaluate(order_id,metadata.buyer_id)
            if not assessment or not assessment['return_eligible']:
                raise TaskConflict((assessment or {}).get('reason_code','INCIDENT_EVIDENCE_MISSING'))
            fact['incident']=assessment
            ticket_meta={'incident_id':assessment['event_id'],'incident_version':assessment['event_version'],
                         'batch_code':order['batch_code'],'batch_source':order['batch_source'],
                         'reason_code':assessment['reason_code']}
        else:
            policy=adapter._invoke_typed_tool('retrieve_policy',{'order_id':order_id,'query':'退货 换货 优惠 退款 价保 政策 期限'})
            evidence=policy.data.get('evidence',[])
            if policy.status!='success' or not evidence:
                raise TaskConflict(policy.error_code or 'POLICY_MISSING')
            tool={'return':'assess_after_sales','exchange':'assess_after_sales',
                  'discount_dispute':'analyze_discount','price_adjustment':'analyze_price_adjustment'}[kind]
            analysis=adapter._invoke_typed_tool(tool,{'order_id':order_id,'evidence':evidence})
            if analysis.status!='success':
                raise TaskConflict(analysis.error_code or 'EVIDENCE_INCOMPLETE')
            data=analysis.data
            fact.update(evidence=evidence,analysis=data)
            if kind in {'return','exchange'}:
                stock=self.repository.inventory(order['sku'])
                fact['stock']=stock
                eligible=data['return_eligible'] if kind=='return' else data['exchange_eligible'] and stock>0
            elif kind=='discount_dispute':
                eligible=not data['consistent']
                ticket_meta=dict(data)
            else:
                eligible=data['eligible'] and data['difference_cents']>0 and bool(data.get('citations_used'))
                ticket_meta=dict(data)
                ticket_meta['policy_document_ids']=sorted({v['document_id'] for v in evidence})
            if not eligible:
                raise TaskConflict('APPLICATION_NOT_ELIGIBLE')
        return fact,ticket_meta

    @staticmethod
    def action_hash(application, ticket_id):
        """确认绑定单号、类型与具体草稿，理由或模型表述不能改变该授权对象。"""
        return fingerprint({'application':application,'ticket_id':ticket_id})

    def _receipt(self, db, metadata, operation_id, action_hash):
        """业务操作回执优先于最新事实，用于已提交但尚未保存图结果的崩溃窗口。"""
        row=db.execute('SELECT * FROM agent_operation_receipts WHERE operation_id=?',(operation_id,)).fetchone()
        if row:
            if (row['task_id'],row['buyer_id'],row['action_hash'])!=(metadata.task_id,metadata.buyer_id,action_hash):
                raise TaskConflict('OPERATION_CONFLICT')
            return json.loads(row['result_json'])

    def _save_receipt(self, db, metadata, operation_id, action_hash, result):
        """与工单状态及确认消费在同一事务提交；故障会全部回滚。"""
        db.execute('INSERT INTO agent_operation_receipts VALUES(?,?,?,?,?,?)',
            (operation_id,metadata.task_id,metadata.buyer_id,action_hash,json.dumps(result,ensure_ascii=False),self.ledger.clock()))

    def _draft(self, metadata, application, reason, operation_id):
        digest=fingerprint(application)
        with self.policy_lock, self.store.transaction() as db:
            self.ledger._lease(db,metadata)
            prior=self._receipt(db,metadata,operation_id,digest)
            if prior is not None:
                return prior
            facts,meta=self._facts(metadata,application)
            record,created,conflict=self.tickets.create_draft(buyer_id=metadata.buyer_id,session_id=metadata.session_id,
                order_id=application['order_id'],sku=facts['order']['sku'],request_type=application['request_type'],
                reason=reason,source_run_id=metadata.run_id,metadata=meta,connection=db)
            if conflict:
                raise TaskConflict('TICKET_CONFLICT')
            if not record:
                raise TaskConflict('INCIDENT_CHANGED_OR_UNVERIFIED')
            # 各任务的卡片独立；查询或另办一单不能误撤销其他任务的合法草稿。
            result={'ticket':self.tickets.public(record).model_dump(mode='json'),'created':created}
            self._save_receipt(db,metadata,operation_id,digest,result)
            return result

    async def draft(self, metadata, application, reason, operation_id):
        """同步规则与 SQL 在工作线程中执行，不阻塞其他会话的 Agent 循环。"""
        return await asyncio.to_thread(self._draft,metadata,application,reason,operation_id)

    def _card(self, db, metadata, application, ticket_id, facts, ticket_meta):
        """重新核验后生成下一代卡片，旧卡片不可复用；刷新普通页面不走此函数。"""
        record=self.tickets._load(db,ticket_id)
        if not record or record['buyer_id']!=metadata.buyer_id or record['status']!='draft':
            raise TaskConflict('TICKET_NOT_CONFIRMABLE')
        if (record['order_id'],record['request_type'])!=(application['order_id'],application['request_type']):
            raise TaskConflict('ACTION_MISMATCH')
        # 草稿复用时更新已重新核验的事实元数据，工单身份和类型仍保持固定。
        db.execute('UPDATE tickets SET metadata_json=? WHERE ticket_id=?',(json.dumps(ticket_meta,ensure_ascii=False),ticket_id))
        generation=db.execute('SELECT coalesce(max(generation),0)+1 FROM agent_confirmations WHERE task_id=?',(metadata.task_id,)).fetchone()[0]
        db.execute("UPDATE agent_confirmations SET status='invalidated' WHERE task_id=? AND status='pending'",(metadata.task_id,))
        card={'confirmation_id':str(uuid4()),'ticket_id':ticket_id,**application,'generation':generation,
              'action_hash':self.action_hash(application,ticket_id),'facts_hash':fingerprint(facts),
              'expires_at':self.ledger.clock()+self.confirmation_seconds,
              'summary':f"订单 {application['order_id']}：提交{LABELS[application['request_type']]}工单；未执行退款。"}
        order=facts['order']
        # 用户可审阅具体商品、金额和本次规则版本，不能只展示一个不可读哈希让用户确认。
        card['facts_summary']=[f"商品：{order['product']}；签收 {order['delivered_days_ago']} 天；实付 {order['paid_cents']/100:.2f} 元。"]
        if 'incident' in facts:
            value=facts['incident']
            card['facts_summary'].append(f"已审核事件 {value['event_id']} / 版本 {value['event_version']}；{value['decision']}；退货资格符合；{value['reason_code']}。")
        elif 'stock' in facts:
            card['facts_summary'].append(f"当前申请资格符合；同款库存 {facts['stock']} 件。")
        elif 'analysis' in facts:
            card['facts_summary'].append(f"当前申请资格符合；核验差额 {facts['analysis'].get('difference_cents',0)/100:.2f} 元。")
        db.execute('INSERT INTO agent_confirmations VALUES(?,?,?,?,?,?,?,?,?,?,?)',
            (card['confirmation_id'],metadata.task_id,metadata.buyer_id,metadata.session_id,ticket_id,card['action_hash'],
             card['facts_hash'],generation,card['expires_at'],'pending',json.dumps(card,ensure_ascii=False)))
        self.tickets._event(db,record,metadata.session_id,'agent_confirmation_created','pending')
        return card

    def _prepare(self, metadata, application, ticket_id):
        with self.policy_lock,self.store.transaction() as db:
            self.ledger._lease(db,metadata)
            facts,meta=self._facts(metadata,application)
            return self._card(db,metadata,application,ticket_id,facts,meta)

    async def prepare(self, metadata, application, ticket_id):
        """创建卡片前核验必要事实；卡片不等于已确认或已提交。"""
        return await asyncio.to_thread(self._prepare,metadata,application,ticket_id)

    def _submit(self, metadata, application, card, operation_id, decision):
        digest=self.action_hash(application,card['ticket_id'])
        with self.policy_lock,self.store.transaction() as db:
            self.ledger._lease(db,metadata)
            prior=self._receipt(db,metadata,operation_id,digest)
            if prior is not None:
                return prior
            row=db.execute('SELECT * FROM agent_confirmations WHERE confirmation_id=?',(card['confirmation_id'],)).fetchone()
            if not row or (row['task_id'],row['buyer_id'],row['session_id'],row['action_hash'])!=(
                    metadata.task_id,metadata.buyer_id,metadata.session_id,digest):
                raise TaskConflict('CONFIRMATION_FORBIDDEN')
            if row['status']!='pending':
                raise TaskConflict('APPROVAL_STALE')
            try:
                facts,meta=self._facts(metadata,application)
            except TaskConflict as exc:
                # 资格不再符合时仍提交“确认已失效”的审计，不因抛错回滚成 pending。
                db.execute("UPDATE agent_confirmations SET status='invalidated' WHERE confirmation_id=?",(card['confirmation_id'],))
                record=self.tickets._load(db,card['ticket_id'])
                if record:
                    self.tickets._event(db,record,metadata.session_id,'confirm_revalidation_failed','rejected',exc.code)
                return {'rejected':True,'error_code':exc.code}
            expired=row['expires_at']<=self.ledger.clock()
            if expired or row['facts_hash']!=fingerprint(facts) or decision=='refresh':
                new=self._card(db,metadata,application,card['ticket_id'],facts,meta)
                return {'renewed':True,'pending_action':new,'reason':'APPROVAL_EXPIRED' if expired else 'APPROVAL_STALE'}
            if decision!='confirm':
                raise TaskConflict('CONFIRMATION_REQUIRED')
            record,changed=self.tickets.confirm(card['ticket_id'],buyer_id=metadata.buyer_id,
                session_id=metadata.session_id,connection=db)
            if not record or record['status']!='submitted':
                raise TaskConflict('TICKET_NOT_CONFIRMABLE')
            db.execute("UPDATE agent_confirmations SET status='consumed' WHERE confirmation_id=?",(card['confirmation_id'],))
            result={'ticket':self.tickets.public(record).model_dump(mode='json'),'changed':changed,'renewed':False}
            self._save_receipt(db,metadata,operation_id,digest,result)
            return result

    async def submit(self, metadata, application, card, operation_id, decision):
        """复核、确认消费、工单提交、审计及操作回执同事务；不会执行支付或退款。"""
        return await asyncio.to_thread(self._submit,metadata,application,card,operation_id,decision)
