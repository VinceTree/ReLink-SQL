"""Public-only, resumable paper-component comparisons on identical initial SQL.

DIN/CHESS use pinned author templates. Self-Refine and CRITIC are explicitly
SQL-domain adaptations, not replications of their original benchmark results.
No evaluation labels are accessed. Every generation is durably logged.
"""
from pathlib import Path
import argparse, hashlib, json, re, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recoverylab.sql import connect, execute, extract_sql
from iteration2.collect import MODELS, seed

ROOT = Path(__file__).resolve().parent
METHODS = ['din_gentle', 'din_generic', 'chess_refine', 'self_refine', 'critic']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def render(name, values):
    text = (ROOT/'prompts'/name).read_text()
    # Literal replacement preserves JSON/braces inside CHESS's original examples.
    for k, v in values.items(): text = text.replace('{'+k+'}', str(v))
    return text

def din_schema(task):
    tables, fks, pks = [], [], []
    with connect(task['db_path']) as con:
        names = [x[0] for x in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        for n in names:
            q = '"'+n.replace('"', '""')+'"'
            cols = list(con.execute(f'PRAGMA table_info({q})'))
            tables.append('Table '+n+', columns = [*,'+','.join(x[1] for x in cols)+']')
            pks += [n+'.'+x[1] for x in cols if x[5]]
            fks += [n+'.'+x[3]+' = '+x[2]+'.'+str(x[4]) for x in con.execute(f'PRAGMA foreign_key_list({q})')]
    return dict(schema='\n'.join(tables), foreign_keys='['+','.join(fks)+']', primary_keys='['+','.join(pks)+']')

def public_observation(obs):
    out = {k: v for k, v in obs.items() if k in ['status', 'error', 'columns']}
    if 'rows' in obs:
        out.update(rows=obs['rows'][:10], row_count=len(obs['rows']), displayed_rows=min(10, len(obs['rows'])))
    return json.dumps(out, ensure_ascii=False, default=str)

def decode_sql(raw, method):
    if method == 'chess_refine':
        found = re.findall(r'<FINAL_ANSWER>(.*?)</FINAL_ANSWER>', raw, re.S|re.I)
        if found: return extract_sql(found[-1])
        if '<FINAL_ANSWER>' in raw: return extract_sql(raw.split('<FINAL_ANSWER>')[-1])
    # DIN's completion backend expected SELECT continuation; chat models can
    # return a complete SQL. Accept either without doubling its SELECT prefix.
    if method == 'din_gentle':
        blocks = re.findall(r'```(?:sql|sqlite)?\s*(.*?)```', raw, re.S|re.I)
        candidate = (blocks[-1] if blocks else raw).strip()
        if re.match(r'^(?:SELECT|WITH)\b', candidate, re.I): return candidate
        return 'SELECT '+candidate
    return extract_sql(raw)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--public', required=True); p.add_argument('--initial', required=True)
    p.add_argument('--output', required=True); p.add_argument('--entry', choices=['error','all'], default='error')
    p.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    p.add_argument('--reuse-stages', help='Completed public stage ledger; model and reconstructed prompts verified before reuse')
    p.add_argument('--rounds', type=int, default=2); p.add_argument('--max-tokens', type=int, default=2048)
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True, exist_ok=True)
    tasks=read(a.public); assert len({t['id'] for t in tasks})==len(tasks)
    assert all(not any(k.startswith('gold') or k=='correct' for k in t) for t in tasks)
    initial={r['task_id']:r for r in read(a.initial) if r['action']=='weak_initial'}
    assert all(t['id'] in initial for t in tasks)
    manifest=dict(arguments=vars(a), public_sha256=sha(a.public), initial_sha256=sha(a.initial),
        code_sha256=sha(__file__), templates={p.name:sha(p) for p in sorted((ROOT/'prompts').glob('*.txt'))},
        model=MODELS['strong14'], quality_labels_accessed=False,
        adaptations=['Common paid weak initial and configurable public execution-error entry.',
        'DIN Spider generic/gentle correction templates on both datasets; BIRD domain evidence appended to question.',
        'DIN completion-to-chat adapter accepts SELECT continuation or full query; greedy EOS/2048 cap replaces author API stop strings.',
        'CHESS candidate refinement component only; common schema, execution report with first 10 rows, fixed few-shot template; source uppercase SELECT validation retains prior SQL on missing SELECT.',
        'Self-Refine SQL adaptation: feedback then refinement, complete previous feedback history; no external execution supplied.',
        'CRITIC program-to-SQL adaptation: actual readonly execution before critique and revision; fixed rounds, latest executable candidate (including empty result) retained as next-round context.',
        'All paid stages counted, including feedback and repeated history; fixed round prefixes available without gold stopping.'])
    mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=json.loads(json.dumps(manifest)):
        raise ValueError('Frozen run mismatch; use a new output directory')
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    dest=out/'stages.jsonl'; saved=read(dest) if dest.exists() else []
    records={(r['task_id'],r['method'],r['round'],r['stage']):r for r in saved}
    assert len(records)==len(saved), 'Duplicate paid stage key'
    if a.reuse_stages and not saved:
        reused=read(a.reuse_stages); valid_ids={t['id'] for t in tasks}
        for r in reused:
            if r['task_id'] in valid_ids and r['method'] in a.methods and r['round']<=a.rounds:
                assert (r['model'],r['revision'])==tuple(MODELS['strong14'])
                key=(r['task_id'],r['method'],r['round'],r['stage']); assert key not in records
                records[key]=r
        dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records.values()))
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    repo,rev=MODELS['strong14']; model,tokenizer=load(str(Path.home()/'.cache/huggingface/hub'/('models--'+repo.replace('/','--'))/'snapshots'/rev))
    def generate(t, method, rnd, stage, prompt):
        key=(t['id'],method,rnd,stage)
        messages=[{'role':'user','content':prompt}]
        chat=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if key in records:
            assert records[key]['prompt_sha256']==hashlib.sha256(chat.encode()).hexdigest(), 'Resumed prompt changed'
            return records[key]
        if len(tokenizer.encode(chat))>24000: raise ValueError('Context exceeds 24000; no silent truncation')
        mx.random.seed(seed(t['id'], method+str(rnd)+stage)); chunks=[]
        for r in stream_generate(model,tokenizer,chat,max_tokens=a.max_tokens,sampler=make_sampler(temp=0)): chunks.append(r.text)
        raw=''.join(chunks); text=decode_sql(raw,method) if stage=='answer' else raw
        rec=dict(task_id=t['id'],dataset=t['dataset'],group=t['group'],split=t['split'],method=method,round=rnd,stage=stage,
            text=text,raw_text=raw,prompt_tokens=r.prompt_tokens,generation_tokens=r.generation_tokens,finish_reason=r.finish_reason,
            model=repo,revision=rev,prompt_sha256=hashlib.sha256(chat.encode()).hexdigest(),messages=messages)
        if method=='chess_refine': rec['closed_final_answer_tag']=bool(re.search(r'<FINAL_ANSWER>.*?</FINAL_ANSWER>',raw,re.S|re.I))
        if stage=='answer':rec['observation']=execute(t['db_path'],text)
        with dest.open('a') as f: f.write(json.dumps(rec,ensure_ascii=False,default=str)+'\n'); f.flush()
        records[key]=rec
        print(json.dumps(dict(method=method,round=rnd,stage=stage,task_id=t['id'],input_tokens=r.prompt_tokens,output_tokens=r.generation_tokens)),flush=True)
        return rec
    decisions=[]
    for method in a.methods:
        for t in tasks:
            init=initial[t['id']]; enabled=a.entry=='all' or init['observation']['status']!='ok'
            sql=init['text']; obs=init['observation']; history=[]; paid=[]
            rounds=a.rounds if method in ['self_refine','critic'] else 1
            for rnd in range(1,rounds+1):
                if enabled:
                    values=dict(schema=t['schema'],question=t['question'],evidence=t.get('evidence',''),sql=sql,
                        observation=public_observation(obs),history='\n\n'.join(history))
                    if method.startswith('din_'):
                        values.update(din_schema(t))
                        values['question']=t['question']+('\nEvidence: '+t['evidence'] if t.get('evidence') else '')
                        prompt=render('din_sql_gentle.txt' if method=='din_gentle' else 'din_sql_generic_figure6.txt',values)
                    elif method=='chess_refine':
                        prompt=render('chess_revise_one.txt',dict(DATABASE_SCHEMA=t['schema'],QUESTION=t['question'],HINT=t.get('evidence',''),QUERY=sql,RESULT=public_observation(obs)))
                    else:
                        feedback=generate(t,method,rnd,'feedback',render(method+'_sql_feedback_adapted.txt',values))
                        paid.append([t['id'],method,rnd,'feedback']);values['feedback']=feedback['text']
                        prompt=render(method+'_sql_refine_adapted.txt',values)
                    answer=generate(t,method,rnd,'answer',prompt);paid.append([t['id'],method,rnd,'answer'])
                    if method in ['self_refine','critic']:
                        history.append('Previous SQL:\n'+sql+'\nFeedback:\n'+values['feedback'])
                    # Empty SQL result is a non-None program output as well.
                    if method!='critic' or answer['observation']['status']=='ok':
                        sql=answer['text'];obs=answer['observation']
                    delivered=answer['text']
                    if method=='chess_refine' and 'SELECT' not in delivered:
                        delivered=init['text']
                else: delivered=init['text']
                decisions.append(dict(task_id=t['id'],dataset=t['dataset'],group=t['group'],policy=method+'_'+a.entry+'_r'+str(rnd),
                    sql=delivered,paid_stages=list(paid),input_tokens=init['prompt_tokens']+sum(records[tuple(k)]['prompt_tokens'] for k in paid),
                    output_tokens=init['generation_tokens']+sum(records[tuple(k)]['generation_tokens'] for k in paid)))
    (out/'public_decisions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in decisions))
    (out/'complete.json').write_text(json.dumps(dict(stages=len(records),decisions=len(decisions),decisions_sha256=sha(out/'public_decisions.jsonl'),stages_sha256=sha(dest),manifest_sha256=sha(mp),labels_accessed=False),indent=2))
    print(json.dumps(dict(complete=True,stages=len(records),decisions=len(decisions))),flush=True)

if __name__=='__main__':main()
