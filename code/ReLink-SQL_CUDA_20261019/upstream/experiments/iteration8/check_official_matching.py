"""Supplementary comparison with pinned official result matching functions.

Uses committed observations, no model calls. This is NOT test-suite accuracy.
"""
from pathlib import Path
from collections import defaultdict
from itertools import product
from typing import Tuple,Any,List,Set
import ast,random,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from iteration8.cohort import HERE,OUT,DATA,read,write,sha

def official_spider():
    source=HERE/'sources/sql_eval/exec_eval.py'
    provenance=json.loads((source.parent/'source.json').read_text())
    assert sha(source)==provenance['files'][source.name]
    names={'permute_tuple','unorder_row','quick_rej','multiset_eq','get_constraint_permutation','result_eq'}
    tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==names
    ns=dict(Tuple=Tuple,Any=Any,List=List,Set=Set,defaultdict=defaultdict,product=product,random=random)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    return ns['result_eq']

def official_bird():
    source=HERE/'sources/sql_eval/bird_evaluation.py';tree=ast.parse(source.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute_sql')
    cond=next(n.test for n in ast.walk(fn) if isinstance(n,ast.If))
    assert ast.unparse(cond)=='set(predicted_res) == set(ground_truth_res)'
    code=compile(ast.Expression(body=cond),str(source),'eval')
    return lambda p,g:bool(eval(code,{'set':set},{'predicted_res':p,'ground_truth_res':g}))

def run():
    marker=json.loads((OUT/'score_complete.json').read_text());assert sha(OUT/'scored.jsonl')==marker['scored_sha256']
    assert sha(DATA/'private/references.json')==marker['reference_sha256']
    refs=json.loads((DATA/'private/references.json').read_text());rows=read(OUT/'scored.jsonl')
    sp=official_spider();bird=official_bird();random.seed(7);cache={};results=[]
    for r in rows:
        key=(r['task_id'],r['observation'].get('sql',''))
        if key not in cache:
            g=refs[r['task_id']];p=r['observation']
            if not r['reference_ok']:value=None
            elif p['status']!='ok':value=False
            else:
                a=[tuple(x) for x in p['rows']];b=[tuple(x) for x in g['gold_result']['rows']]
                value=bool(sp(b,a,order_matters='order by' in g['gold_sql'].lower())) if r['dataset']=='spider' else bird(a,b)
            cache[key]=value
        results.append(dict(task_id=r['task_id'],dataset=r['dataset'],flow=r['flow'],arm=r['arm'],internal_correct=r['correct'],official_match_correct=cache[key]))
    write(OUT/'official_match_scored.jsonl',results,True)
    summary=[]
    for ds in ['spider','bird']:
        for flow in ['single','fallback']:
            for arm in sorted({r['arm'] for r in results}):
                rr=[r for r in results if r['dataset']==ds and r['flow']==flow and r['arm']==arm];valid=[r for r in rr if r['official_match_correct'] is not None]
                summary.append(dict(dataset=ds,flow=flow,arm=arm,n=len(rr),scorable=len(valid),internal_correct=sum(r['internal_correct'] for r in valid),official_match_correct=sum(r['official_match_correct'] for r in valid),changed=sum(r['internal_correct']!=r['official_match_correct'] for r in valid)))
    write(OUT/'official_match_summary.json',summary)
    write(OUT/'official_match_audit.json',dict(scope='Pinned official result matchers applied to the same committed single-database observations; no test suites, SQL preprocessing, substituted gold values, or reference-based routing',spider_commit=json.loads((HERE/'sources/sql_eval/source.json').read_text())['commit'],bird_commit='483554eae102996f5ec1f4feab4e78ef29c2a394',bird_evaluation_sha256=sha(HERE/'sources/sql_eval/bird_evaluation.py'),new_model_calls=0,scored_sha256=sha(OUT/'official_match_scored.jsonl'),original_results_unchanged=True))
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':run()
