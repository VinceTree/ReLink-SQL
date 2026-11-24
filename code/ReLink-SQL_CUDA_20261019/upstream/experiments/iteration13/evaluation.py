"""Evaluation boundary: labels never enter prompt construction."""
from collections import defaultdict
from pathlib import Path
import json
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iteration12 import run as prior
from iteration12.feedback import UserSimulator
from iteration8.check_official_matching import official_spider, official_bird

HERE = Path(__file__).resolve().parent


class Evaluator(UserSimulator):
    def __init__(self, reference_path):
        super().__init__(reference_path)
        self.spider = official_spider()
        self.bird = official_bird()
        self.cache = {}

    def official(self, task, observation):
        key = (task['id'], prior.batch.digest(observation))
        if key not in self.cache:
            ref = self._references[task['id']]
            if ref['gold_result']['status'] != 'ok':
                value = None
            elif observation['status'] != 'ok':
                value = False
            else:
                predicted = [tuple(x) for x in observation['rows']]
                gold = [tuple(x) for x in ref['gold_result']['rows']]
                random.seed(7)
                value = (self.spider(gold, predicted, order_matters='order by' in ref['gold_sql'].lower())
                         if task['dataset'] == 'spider' else self.bird(predicted, gold))
            self.cache[key] = value
        return self.cache[key]


def summarize(paths, tasks, evaluator):
    result = []
    groups = defaultdict(list)
    for row in paths:
        for ds in ('ALL', tasks[row['task_id']]['dataset']):
            groups[row['method'], row['round'], ds].append(row)
    for (method, rnd, ds), rows in sorted(groups.items()):
        errors = [r for r in rows if not r['initial_accepted']]
        official = [evaluator.official(tasks[r['task_id']], r['observation']) for r in rows]
        result.append(dict(method=method, round=rnd, dataset=ds, n=len(rows),
            initial_correct=sum(r['initial_accepted'] for r in rows),
            rescued=sum(r['accepted'] for r in errors), correct=sum(r['accepted'] for r in rows),
            mean_all_tokens=sum(r['total_tokens'] for r in rows)/len(rows),
            mean_repair_tokens=sum(r['repair_tokens'] for r in rows)/len(errors) if errors else 0,
            official_endpoint_correct=sum(v is True for v in official),
            official_scorable=sum(v is not None for v in official),
            endpoint_disagreements=sum(v != r['accepted'] for v,r in zip(official,rows)),
            feedback_protocol='internal result equality; official matching is endpoint sensitivity only'))
    return result


def sensitivity():
    prior.verify()
    target = HERE / 'results/scoring_audit'
    tasks = {r['id']: r for r in prior.read(prior.OUT/'public.jsonl')}
    evaluator = Evaluator(prior.REFERENCE)
    sources = [prior.OUT/'initial_paths.jsonl', prior.OUT/'r1/paths.jsonl', prior.OUT/'r2/paths.jsonl',
               prior.OUT/'share_author_prompts_r1/paths.jsonl']
    records = []
    for source in sources:
        for r in prior.read(source):
            method = r.get('method', 'SHARE_AUTHOR_PROMPTS')
            rnd = r.get('round', 1)
            if rnd == 0 and method != 'RELINK14':
                continue
            records.append(dict(task_id=r['task_id'], method=method if rnd else 'INITIAL', round=rnd,
                                accepted=r['accepted'], observation=r['observation']))
    old_observations = prior.read(prior.OLD/'candidate_observations.json')
    initial = {r['task_id']:r for r in prior.read(prior.OUT/'initial.jsonl')}
    for name, method in [('fresh14','FRESH14'),('relink_fresh14','RELINK')]:
        source = prior.OUT/(name+'_paths.jsonl'); sources.append(source)
        for r in prior.read(source):
            obs = (initial[r['task_id']]['observation'] if r['initial_accepted'] else
                   old_observations[r['repair_request_id']]['observation'])
            records.append(dict(task_id=r['task_id'],method=method,round=1,
                                accepted=r['accepted'],observation=obs))
    outcomes = []
    for r in records:
        tid = r['task_id']
        assert r['accepted'] == evaluator.assess(tid,r['observation'])['accepted']
        official = evaluator.official(tasks[tid],r['observation'])
        outcomes.append(dict(task_id=tid,dataset=tasks[tid]['dataset'],method=r['method'],round=r['round'],
            internal_correct=r['accepted'],official_match_correct=official,
            observation_sha256=prior.batch.digest(r['observation'])))
    grouped = defaultdict(list)
    for r in outcomes:
        for ds in ('ALL',r['dataset']):grouped[r['method'],r['round'],ds].append(r)
    summary = [dict(method=m,round=rnd,dataset=ds,n=len(rows),
        internal_correct=sum(r['internal_correct'] for r in rows),
        official_correct=sum(r['official_match_correct'] is True for r in rows),
        disagreements=sum(r['internal_correct'] != r['official_match_correct'] for r in rows))
        for (m,rnd,ds),rows in sorted(grouped.items())]
    prior.save(target/'outcomes.jsonl',outcomes)
    prior.save(target/'summary.json',summary)
    prior.save(target/'complete.json',dict(new_model_calls=0,endpoint_rescore_only=True,
        test_suite_accuracy=False,acceptance_and_cost_unchanged=True,
        initial_disagreements=sum(r['internal_correct']!=r['official_match_correct'] for r in outcomes if r['round']==0),
        sources={str(p):prior.batch.sha(p) for p in sources+[prior.REFERENCE,prior.OLD/'candidate_observations.json']},
        analysis_code_sha256=prior.batch.sha(__file__),
        matchers={str(p):prior.batch.sha(p) for p in [Path(__file__).resolve().parents[1]/'iteration8/check_official_matching.py',
            Path(__file__).resolve().parents[1]/'iteration8/sources/sql_eval/exec_eval.py',
            Path(__file__).resolve().parents[1]/'iteration8/sources/sql_eval/bird_evaluation.py']}))
    print(json.dumps([r for r in summary if r['dataset']=='ALL'],ensure_ascii=False,indent=2))


if __name__ == '__main__':
    sensitivity()
