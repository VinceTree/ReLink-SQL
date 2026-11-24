"""Two-phase public W/F/V collection using one shared batch backend.

Preparation/finalization never loads a model or opens references. GPU execution
is the separate batch_public_generate run command. All policies replay the same
three-action pool; uncalled views are explicit zero-token weak aliases.
"""
from pathlib import Path
import argparse,ast,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recoverylab.sql import execute
from iteration3.context_variants import build_repair_context
from sqlglot.errors import SqlglotError
from iteration4 import batch_public_generate as batch

ROOT=Path(__file__).resolve().parents[1]
COLLECT=ROOT/'iteration2/collect.py'
tree=ast.parse(COLLECT.read_text())
MODELS=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MODELS' for t in n.targets)))
SYSTEM=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYSTEM' for t in n.targets)))
CONFIDENCE=('logprob_mean','logprob_min','logprob_q10','low_confidence_fraction')

def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,rows):
    p=Path(p);text=''.join(json.dumps(r,ensure_ascii=False,default=str)+'\n' for r in rows)
    if p.exists() and p.read_text()!=text:raise ValueError('Refusing changed existing artifact '+str(p))
    p.write_text(text)

def initial_messages(t):
    user='Database schema:\n'+t['schema']+'\nRequest: '+t['question']
    if t.get('evidence'):user+='\nProvided domain knowledge: '+t['evidence']
    return [{'role':'system','content':SYSTEM},{'role':'user','content':user}]

def request(t,action,schema=None):
    model,rev=MODELS['weak' if action=='weak_initial' else 'strong14']
    return dict(id=t['id']+'|'+action,messages=initial_messages(dict(t,schema=schema or t['schema'])),model=model,revision=rev,max_tokens=384)

def result_pool(tasks,results,action,pool_id,contexts=None):
    records=[]
    for t in tasks:
        ident=t['id']+'|'+action
        if ident not in results:raise ValueError('Missing required completion '+ident)
        r=results[ident];text=r['raw_text'];obs=execute(t['db_path'],text)
        record=dict(task_id=t['id'],dataset=t['dataset'],group=t['group'],split=t['split'],action=action,
            text=text,raw_text=text,observation=obs,model=r['model'],revision=r['revision'],
            prompt_tokens=r['prompt_tokens'],generation_tokens=r['generation_tokens'],llm_calls=1,
            finish_reason=r['finish_reason'],prompt_sha256=r['rendered_prompt_sha256'],
            backend='mlx_lm.BatchGenerator/batch4',pool_id=pool_id,batch_id=r['batch_id'],batch_members=r['batch_members'],
            batch_attempt_id=r['attempt_id'],batch_manifest_sha256=r['manifest_sha256'],max_tokens=384,
            service_seconds=None,service_seconds_note='Shared batch elapsed is not additive per-item latency; token-only evaluation',
            feedback_seconds=None)
        if action.endswith('_initial'):
            assert all(k in r for k in CONFIDENCE)
            record.update({k:r[k] for k in CONFIDENCE})
        else:
            c=contexts[t['id']]
            record.update(schema_mode='closure_columns',selected_tables=c.get('selected_tables',[]),selected_columns=c.get('selected_columns',{}),
                index_diagnostics=c.get('diagnostics',{}),deanchored_generation=True,entry='initial_execution_error',
                confidence_not_exposed='Matches frozen training view records; per-token values remain in batch raw archive')
        records.append(record)
    return records

def completed(p):
    m=json.loads((p/'complete.json').read_text());assert sha(p/'results.jsonl')==m['results_sha256']
    rows=read(p/'results.jsonl');assert len(rows)==m['items']
    assert len({r['id'] for r in rows})==len(rows)
    return {r['id']:r for r in rows}

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare_weak','finish_weak','prepare_strong','finish_strong'])
    p.add_argument('--public',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    tasks=read(a.public);assert len({t['id'] for t in tasks})==len(tasks)
    assert all(not any(k=='correct' or k.startswith('gold') for k in t) for t in tasks)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    protocol=dict(public_sha256=sha(a.public),code_sha256=sha(__file__),source_formatter_sha256=sha(COLLECT),
        batch_runner_sha256=sha(batch.__file__),structure_sources={n:sha(ROOT/'iteration3'/n) for n in ['context_variants.py','repair_index.py','column_context.py']},
        actions=['weak_initial','strong14_initial','fresh14_columns'],models=MODELS,temperature=0,max_tokens=384,batch_size=4,
        prompt_encoding='stream_string',token_metric='Complete model input plus output; output includes terminal EOS',
        backend='mlx_lm.BatchGenerator shared by every compared policy',
        conditional_view='Only initial execution errors cause a real view generation; otherwise exact weak alias with zero extra tokens',
        selector_guard='Any SQLGlot exception retains full original schema; no per-question or gold condition',
        legacy_backend_difference='Earlier training used scalar stream_generate; batch numerical differences are possible and are shared across every confirmation policy',
        labels_accessed=False)
    protocol.update(pool_id='confirmation_'+hashlib.sha256((sha(a.public)+sha(__file__)).encode()).hexdigest()[:20],
        backend_row_name='mlx_lm.BatchGenerator/batch4',backend_version=batch.backend_metadata()['mlx_lm_version'])
    mp=out/'pool_provenance.json'
    if mp.exists() and json.loads(mp.read_text())!=json.loads(json.dumps(protocol)):raise ValueError('Frozen protocol mismatch')
    mp.write_text(json.dumps(protocol,indent=2))
    if a.phase=='prepare_weak':
        rp=out/'weak_requests.jsonl';dump(rp,[request(t,'weak_initial') for t in tasks])
        print(batch.prepare(rp,out/'weak_batch'))
    elif a.phase=='finish_weak':
        results=completed(out/'weak_batch');assert len(results)==len(tasks)
        dump(out/'weak_rollouts.jsonl',result_pool(tasks,results,'weak_initial',protocol['pool_id']))
        print(json.dumps(dict(weak_complete=True,rows=len(tasks),labels_accessed=False)))
    elif a.phase=='prepare_strong':
        weak={r['task_id']:r for r in read(out/'weak_rollouts.jsonl')};assert set(weak)=={t['id'] for t in tasks}
        req=[request(t,'strong14_initial') for t in tasks];contexts={}
        for t in tasks:
            r=weak[t['id']]
            if r['observation']['status']=='ok':continue
            try:c=build_repair_context(t,r,mode='closure_columns',token_budget=768)
            except SqlglotError as e:
                c=dict(schema_text=t['schema'],selected_tables=[],selected_columns={},diagnostics=dict(fallback='full_schema_after_sqlglot_exception',actual_mode='full',exception_type=type(e).__name__))
            contexts[t['id']]=c;req.append(request(t,'fresh14_columns',c['schema_text']))
        # Graph acquisition times are irrelevant to this token/accuracy study;
        # keep original compile artifact, never insert times into model features.
        cp=out/'view_contexts.json'
        if not cp.exists():cp.write_text(json.dumps(contexts,ensure_ascii=False,default=str))
        rp=out/'strong_requests.jsonl';dump(rp,req)
        print(batch.prepare(rp,out/'strong_batch'))
    else:
        weak={r['task_id']:r for r in read(out/'weak_rollouts.jsonl')};results=completed(out/'strong_batch')
        errors=[t for t in tasks if weak[t['id']]['observation']['status']!='ok'];contexts=json.loads((out/'view_contexts.json').read_text())
        assert len(results)==len(tasks)+len(errors)
        records=list(weak.values())+result_pool(tasks,results,'strong14_initial',protocol['pool_id'])+result_pool(errors,results,'fresh14_columns',protocol['pool_id'],contexts)
        for t in tasks:
            w=weak[t['id']]
            if w['observation']['status']=='ok':
                records.append(dict(w,action='fresh14_columns',model=MODELS['strong14'][0],revision=MODELS['strong14'][1],
                    prompt_tokens=0,generation_tokens=0,llm_calls=0,finish_reason='not_called',observation=dict(w['observation'],seconds=0.),
                    schema_mode='closure_columns',deanchored_generation=True,entry='initial_execution_error'))
        dump(out/'rollouts.jsonl',records)
        (out/'complete.json').write_text(json.dumps(dict(tasks=len(tasks),records=len(records),view_calls=len(errors),
            rollouts_sha256=sha(out/'rollouts.jsonl'),public_sha256=sha(a.public),provenance_sha256=sha(mp),labels_accessed=False),indent=2))
        print(json.dumps(dict(complete=True,records=len(records),view_calls=len(errors),labels_accessed=False)))

if __name__=='__main__':main()
