"""Checks for gold isolation, paid explanations, stopped successes and resume."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import dataset, methods, run

r=dataset.r


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.task=dict(id='case',dataset='spider',group='x',split='dev',kind='sql',question='Count people.',
                       evidence='',schema='CREATE TABLE people(id INTEGER);',db_path='/not_used.sqlite')
        self.full={'schema_text':self.task['schema']}
        self.state=dict(task_id='case',method='SELF_DEBUG',round=1,sql='SELECT 1',
            observation=dict(status='ok',columns=['x'],rows=[[1],[2],[3]],sql='SELECT 1'),
            rejected_sqls=['SELECT 1'],critiques=[],feedback=r.methods.NEGATIVE)

    def test_no_reference_fields(self):
        with self.assertRaises(ValueError):
            methods.selfdebug_explanation(dict(self.task,gold_sql='SECRET'),self.state,self.full)
        with self.assertRaises(ValueError):
            methods.selfdebug_explanation(self.task,dict(self.state,gold_result='SECRET'),self.full)

    def test_selfdebug_order_and_preview(self):
        msg=methods.selfdebug_explanation(self.task,self.state,self.full)
        self.assertNotIn('Count people.',msg[-1]['content'])
        self.assertIn('[[1], [2]]',msg[-1]['content'])
        self.assertNotIn('[3]',msg[-1]['content'])
        messages,_=methods.build(self.task,self.state,self.full,'answer','QUESTION EXPLANATION',
                                {'case':{'text':'SQL EXPLANATION'}})
        self.assertEqual(messages[2]['content'],'SQL EXPLANATION')
        self.assertIn('Count people.',messages[3]['content'])
        self.assertEqual(messages[4]['content'],'QUESTION EXPLANATION')
        self.assertIn(r.methods.NEGATIVE,messages[5]['content'])

    def test_reflexion_remembers_prior_failure(self):
        state=dict(self.state,method='REFLEXION',critiques=['Earlier reflection: count distinct people.'])
        for stage in ('feedback','answer'):
            msg,_=methods.build(self.task,state,self.full,stage,'New reflection')
            self.assertIn('Earlier reflection: count distinct people.',msg[0]['content'])
            self.assertIn(r.methods.NEGATIVE,msg[0]['content'])

    def test_completed_round_does_not_invoke_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);r.save(root/'r1/paths.jsonl',[{'saved':True}]);r.save(root/'r1/complete.json',{})
            class NoCalls:
                def invoke(self,*args):raise AssertionError('Unexpected inference')
            self.assertEqual(methods.round_run(root,1,[],{}, {},{},NoCalls(),None),[{'saved':True}])

    def test_success_cannot_accumulate_cost(self):
        bank=type('Bank',(),{'raw':{'a':dict(prompt_tokens=3,generation_tokens=2),
                                      'b':dict(prompt_tokens=7,generation_tokens=1)},
                             'tokens':lambda self,ids:sum(self.raw[i]['prompt_tokens']+
                                                         self.raw[i]['generation_tokens'] for i in ids)})()
        a=r.account(dict(task_id='case',method='CURRENT',accepted=True,sql='SELECT 1',call_ids=['a']),bank)
        run.validate_transition([a],[dict(a)],bank)
        b=r.account(dict(a,call_ids=['a','b']),bank)
        with self.assertRaises(AssertionError):run.validate_transition([a],[b],bank)

    def test_official_feedback_after_commit(self):
        bank=type('Bank',(),{'raw':{'a':dict(prompt_tokens=3,generation_tokens=2)},'tokens':lambda s,ids:5*len(ids)})()
        before=r.account(dict(task_id='case',method='CURRENT',accepted=False,feedback=r.methods.NEGATIVE,
                              sql='SELECT 1',call_ids=['a']),bank)
        after={k:v for k,v in before.items() if k not in ('accepted','feedback')};after.update(carried=False,round=4)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def signal(evaluator,task,observation,gate):
                self.assertTrue((root/'deliveries_committed.json').exists());self.assertEqual(gate,'official')
                return dict(accepted=True,feedback='')
            after['observation']=self.state['observation']
            with patch.object(r,'signal',signal):
                result=run.commit(root,[before],[after],{'case':self.task},None,bank)
            self.assertTrue(result[0]['accepted'])

    def test_three_selfdebug_calls_are_all_billed(self):
        class Bank:
            def __init__(self):
                self.raw={'initial':dict(prompt_tokens=3,generation_tokens=2)}
                self.stages=[]
            def tokens(self,ids):
                return sum(self.raw[i]['prompt_tokens']+self.raw[i]['generation_tokens'] for i in ids)
            def invoke(self,directory,specs,stopped=False):
                rows=[]
                for sp in specs:
                    rid=sp['request']['id'];self.stages.append(sp['stage'])
                    self.raw[rid]=dict(prompt_tokens=5,generation_tokens=2)
                    rows.append(dict(task_id=sp['task_id'],method=sp['method'],stage=sp['stage'],
                                     text='SELECT 2' if sp['stage']=='answer' else 'Generated explanation',
                                     raw_text='SELECT 2' if sp['stage']=='answer' else 'Generated explanation',
                                     request_ids=[rid],request_id=rid))
                return rows
        bank=Bank()
        before=r.account(dict(task_id='case',method='SELF_DEBUG',round=0,accepted=False,
            initial_accepted=False,feedback=r.methods.NEGATIVE,stopped=False,sql='SELECT 1',
            observation=self.state['observation'],call_ids=['initial'],rejected_sqls=[],critiques=[]),bank)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(r,'execute',return_value=self.state['observation']),patch.object(
                r,'signal',return_value=dict(accepted=False,feedback=r.methods.NEGATIVE)):
                after=methods.round_run(Path(folder),1,[before],{'case':self.task},
                    {'case':self.full},{},bank,None)
            self.assertEqual(bank.stages,['sql_explanation','feedback','answer'])
            self.assertEqual(len(after[0]['call_ids']),4)
            self.assertEqual(after[0]['total_tokens'],26)

    def test_planning_is_billed_even_without_deliverable_program_patch(self):
        class Bank:
            raw={'initial':dict(prompt_tokens=3,generation_tokens=2),
                 'plan':dict(prompt_tokens=7,generation_tokens=3)}
            def tokens(self,ids):
                return sum(self.raw[x]['prompt_tokens']+self.raw[x]['generation_tokens'] for x in ids)
            def invoke(self,directory,specs):
                self.asserted_count=len(specs)
                return [dict(task_id='case',text='Requirements',request_ids=['plan'])]
        bank=Bank()
        before=r.account(dict(task_id='case',method='CURRENT',round=3,accepted=False,
            initial_accepted=False,feedback=r.methods.NEGATIVE,stopped=False,sql='SELECT 1',
            observation=self.state['observation'],call_ids=['initial'],rejected_sqls=[],critiques=[]),bank)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            r.save(root/'initial_paths.jsonl',[dict(before,round=0)])
            for stage in (1,2,3):r.save(root/f'r{stage}/paths.jsonl',[dict(before,round=stage)])
            with patch.object(run.contract,'plan_messages',return_value=([dict(role='user',content='Requirements')],{})),\
                 patch.object(run.multiplicity_safe,'propose',return_value=(None,{})),\
                 patch.object(r,'signal',side_effect=AssertionError('No new candidate to evaluate')):
                after=run.current_tail(root,4,[before],{'case':self.task},bank,None)
            self.assertEqual(bank.asserted_count,1)
            self.assertTrue(after[0]['carried'])
            self.assertFalse(after[0]['accepted'])
            self.assertEqual(after[0]['total_tokens'],15)
            self.assertEqual(after[0]['call_ids'],['initial','plan'])


if __name__=='__main__':unittest.main()
