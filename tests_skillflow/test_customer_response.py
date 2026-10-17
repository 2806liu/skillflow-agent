"""客服表达回归：核验事实与展示分离，未知物流和越权不能被润色为已查到。"""

import unittest

from skillflow.agent.finish import inspect_goal, render_facts
from skillflow.agent.goals import AnswerProposal, TaskGoal
from skillflow.agent.tool_executor import TargetScope


class CustomerResponseTests(unittest.TestCase):
    """只检查业务含义和证据约束，不固定礼貌用词或标点。"""
    def finding(self, logistics=None, forbidden=False):
        """按实际工具结果结构构造完成门输入，原始事实继续供右侧审计。"""
        goal = TaskGoal(requirements=['order', 'logistics'], scope=TargetScope(explicit_order_ids=['A123']),
                        original_request='查询 A123 配送物流')
        order = {'name':'get_order', 'arguments':{'order_id':'A123'}, 'observation':{
            'status':'rejected' if forbidden else 'success', 'error_code':'ORDER_FORBIDDEN' if forbidden else None,
            'result_ref':None if forbidden else 'order-ref', 'data':{} if forbidden else {'order':{
                'product':'演示耳机', 'paid_cents':18900, 'delivered_days_ago':2}}}}
        rows = [order]
        if logistics:
            rows.append({'name':'get_logistics', 'arguments':{'order_id':'A123'}, 'observation':logistics})
        return inspect_goal(goal, rows, logistics_enabled=True)

    def reply(self, finding):
        """仍须选择全部可信事实及精确结果引用，客服表达层无权跳过校验。"""
        return render_facts(finding, AnswerProposal(fact_indexes=list(range(len(finding.facts))), result_refs=finding.refs))

    def test_empty_record_remains_unknown_and_does_not_claim_completion_or_refund(self):
        finding = self.finding({'status':'empty', 'error_code':None, 'result_ref':'logistics-ref',
            'message':'暂无记录', 'data':{'queried_at':'2026-10-16T13:00:00Z'}})
        raw = list(finding.facts)
        answer = self.reply(finding)
        self.assertIn('暂无配送物流记录', answer)
        self.assertIn('不能确认配送进度', answer)
        self.assertNotIn('已签收', answer)
        self.assertNotIn('退款', answer)
        self.assertEqual(raw, finding.facts)
        self.assertEqual(finding.refs, ['order-ref', 'logistics-ref'])

    def test_permission_denial_is_not_polished_into_a_success(self):
        finding = self.finding(forbidden=True)
        answer = self.reply(finding)
        self.assertIn('无权访问', answer)
        self.assertNotIn('已帮您核对', answer)
        self.assertNotIn('已完成', answer)
        self.assertEqual(finding.reason, 'ORDER_FORBIDDEN')
        self.assertEqual(finding.refs, [])

    def test_display_does_not_relax_reference_or_fact_index_validation(self):
        finding = self.finding(forbidden=True)
        with self.assertRaisesRegex(ValueError, 'ANSWER_PROPOSAL_INVALID'):
            render_facts(finding, AnswerProposal(fact_indexes=[0], result_refs=['invented-ref']))
        with self.assertRaisesRegex(ValueError, 'ANSWER_PROPOSAL_INVALID'):
            render_facts(finding, AnswerProposal(fact_indexes=[], result_refs=[]))


if __name__ == '__main__':
    unittest.main()
