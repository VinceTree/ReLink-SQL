"""Feedback experiments using the existing generation backend and SQL methods.

No model training or provider implementation is duplicated here. Cached requests
must match exactly, and every logical invocation is charged in full.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iteration12 import run as prior, methods
from iteration11 import contexts
from iteration11.collect import CACHE_PAIRS
from iteration13.evaluation import Evaluator, summarize, HERE
from recoverylab.sql import execute

read, save = prior.read, prior.save
DIRECT = {
    'FRESH14': ('F', False, False, False),
    'RELINK': ('V', False, False, False),
    'NO_SCOPE': ('NO_SCOPE', False, False, False),
    'NO_BRIDGE': ('NO_BRIDGE', False, False, False),
    'LEX_MATCH': ('LEX_MATCH', False, False, False),
    'FULL_REPORT': ('F', False, True, True),
    'VIEW_REPORT': ('V', False, True, True),
}
ALIASES = {'FULL_SQL_REPORT': 'EXEC14', 'VIEW_SQL_REPORT': 'RELINK14'}


def stable(value):
    if isinstance(value, dict):
        return {k: stable(v) for k,v in value.items() if 'seconds' not in k and k!='schema_cache_hit'}
    if isinstance(value, list): return [stable(v) for v in value]
    return value


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer
    info = prior.batch.model_metadata(*methods.model('RELINK14'))
    return AutoTokenizer.from_pretrained(info['local_path'],local_files_only=True,trust_remote_code=False)


def length_matched(task, record):
    """Lexical column ordering; only the target length comes from ReLink."""
    selector = methods.selectors
    tables = selector.ri._schema_index(task['db_path'])[0]['tables']
    query = selector.ri._words(task['question']+' '+task.get('evidence',''))
    ranked = sorted(((-3*len(selector.ri._words(t)&query)-len(selector.ri._words(c)&query),t,c)
                     for t,meta in tables.items() for c in meta['columns']))
    target = len(tokenizer().encode(contexts.recovery_context(task,record)['schema_text'],add_special_tokens=False))
    options = {}
    def candidate(n):
        if n not in options:
            chosen = defaultdict(list)
            for _,t,c in ranked[:n]:chosen[t].append(c)
            ctx = selector.render(task,chosen,preserve_keys=False)
            options[n] = (len(tokenizer().encode(ctx['schema_text'],add_special_tokens=False)),ctx)
        return options[n][0]
    low, high = 1, len(ranked)
    while low < high:
        mid = (low+high)//2
        if candidate(mid) < target: low = mid+1
        else: high = mid
    for n in range(max(1,low-4), min(len(ranked),low+4)+1):candidate(n)
    n = min(options,key=lambda k:(abs(options[k][0]-target),options[k][0],k))
    actual,ctx = options[n]
    return dict(ctx,length_match=dict(target=target,actual=actual,ratio=actual/max(1,target),
                                     prefix_columns=n,public_question_terms_only=True))


def public_state(row, rnd, method=None):
    assert not row['accepted'] and not row.get('stopped')
    return dict(task_id=row['task_id'],method=method or row['method'],round=rnd,sql=row['sql'],
        observation=row['observation'],rejected_sqls=row['rejected_sqls']+[row['sql']],
        critiques=row['critiques'],feedback=methods.NEGATIVE)


def build(task, state, full, stage='answer', auxiliary=None):
    method = state['method']
    checked = dict(state,method=ALIASES.get(method,method if method in methods.METHODS else 'RELINK14'))
    methods.validate(task,checked)
    if method not in DIRECT:
        return methods.build(task,checked,full,stage,auxiliary)
    selection, with_sql, with_exec, with_feedback = DIRECT[method]
    record = dict(raw_text=state['sql'],text=state['sql'],observation=state['observation'])
    if selection == 'F': ctx = full
    elif selection == 'V': ctx = contexts.recovery_context(task,record)
    elif selection == 'LEX_MATCH': ctx = length_matched(task,record)
    else:
        try:ctx=methods.selectors.current(task,record,selection)
        except RecursionError:ctx=dict(full,selection_fallback='parser recursion')
    messages = methods.initial_messages(dict(task,schema=ctx['schema_text']))
    if with_sql:messages[-1]['content']+='\n\nCurrent SQL to correct:\n'+state['sql']
    if with_exec:messages[-1]['content']+='\n\nExecution report:\n'+methods.paper.public_observation(state['observation'])
    if with_feedback:messages=methods.attach(messages,checked)
    return messages, stable(ctx)


def spec(state, stage, messages, metadata=None, model_key='strong14', role=None, attempt=0, tag=''):
    if state['method']=='RETRY7':model_key='strong'
    model, revision = methods.MODELS[model_key] if isinstance(model_key,str) else model_key
    cap = 2048 if stage in ('feedback','selection') else 1024
    if role and role != 'generator':
        model,revision=str(prior.EXP/'models'/('share-'+role+'-4bit')),methods.share.CHECKPOINTS[role][1]
        cap=3000
    return dict(task_id=state['task_id'],method=state['method'],round=state['round'],stage=stage,
        request=dict(id=f"{state['task_id']}|I13_{tag}_{state['method']}_R{state['round']}_{stage}_A{attempt}",
                     messages=messages,model=model,revision=revision,max_tokens=cap),
        metadata=stable({k:v for k,v in (metadata or {}).items() if k!='schema_text'}))


class CallBank:
    def __init__(self, folders):
        self.raw, self.index, self.sources = {}, {}, {}
        for folder in dict.fromkeys(map(Path,folders)):
            if (folder/'complete.json').exists(): self.load(folder)

    @staticmethod
    def key(task_id, request, stopped):
        return prior.batch.digest([task_id,request['model'],request['revision'],request['messages'],request['max_tokens'],stopped])

    def load(self, folder):
        manifest=read(folder/'manifest.json')
        encoding=manifest['prompt_encoding']
        assert encoding in ('stream_string','explicit_no_special_tokens'), encoding
        stopped=encoding=='explicit_no_special_tokens'
        if stopped:assert manifest['stop_policy'].startswith('Original SHARE')
        requests={r['id']:r for r in read(folder/'requests.jsonl')}
        for rid,row in prior.completed(folder).items():
            req=requests[rid]
            assert row['messages_sha256']==prior.batch.digest(req['messages'])
            assert row['model']==req['model'] and row['revision']==req['revision']
            if rid in self.raw:assert prior.batch.digest(row)==prior.batch.digest(self.raw[rid])
            self.raw[rid]=row
            key=self.key(rid.split('|')[0],req,stopped)
            self.index.setdefault(key,rid)
            self.sources[rid]=str(folder)

    def invoke(self, directory, specs, stopped=False):
        record_path=directory/'records.jsonl'
        save(directory/'logical_requests.jsonl',specs)
        if record_path.exists():return read(record_path)
        missing=[];cached=[]
        for sp in specs:
            key=self.key(sp['task_id'],sp['request'],stopped)
            if key not in self.index:missing.append(sp)
            else:
                rid=self.index[key];raw=self.raw[rid]
                cached.append(dict(task_id=sp['task_id'],method=sp['method'],round=sp['round'],stage=sp['stage'],
                    metadata=sp['metadata'],request_id=rid,request_ids=[rid],text=raw['text'],raw_text=raw['raw_text'],
                    finish_reason=raw['finish_reason'],reused_from=self.sources[rid],logical_request_id=sp['request']['id']))
        generated=prior.execute_batch(directory/'generated',missing,stopped) if missing else []
        if missing:self.load(directory/'generated/batch')
        rows=cached+generated
        order={(s['task_id'],s['method']):i for i,s in enumerate(specs)}
        rows.sort(key=lambda r:order[r['task_id'],r['method']])
        assert len(rows)==len(specs)
        save(record_path,rows)
        return rows

    def tokens(self, ids):
        return sum(self.raw[i]['prompt_tokens']+self.raw[i]['generation_tokens'] for i in ids)


def history_batches():
    roots=[directory for _,directory in CACHE_PAIRS]
    roots += [prior.OLD/'roots/batch',prior.OLD/'recovery/batch']
    roots += [p.parent for p in sorted(prior.OUT.glob('r*/**/batch/complete.json'))]
    roots += [p.parent for p in sorted((prior.OUT/'share_author_prompts_r1').glob('**/batch/complete.json'))]
    return roots


def signal(evaluator, task, observation, gate):
    if gate=='internal':return evaluator.assess(task['id'],observation)
    accepted=evaluator.official(task,observation)
    if accepted is None:raise ValueError('Unscorable reference')
    return dict(accepted=bool(accepted),feedback='' if accepted else methods.NEGATIVE)


def account(row, bank):
    row['total_tokens']=bank.tokens(row['call_ids'])
    row['repair_tokens']=bank.tokens(row['call_ids'][1:])
    row['input_tokens']=sum(bank.raw[i]['prompt_tokens'] for i in row['call_ids'])
    row['output_tokens']=sum(bank.raw[i]['generation_tokens'] for i in row['call_ids'])
    assert row['total_tokens']==row['input_tokens']+row['output_tokens']
    return row


def initial_paths(tasks, initial, arms, evaluator, bank, gate='internal'):
    rows=[]
    for method in arms:
        for task in tasks:
            r=initial[task['id']];value=signal(evaluator,task,r['observation'],gate)
            rows.append(account(dict(task_id=task['id'],method=method,round=0,sql=r['sql'],observation=r['observation'],
                call_ids=[r['request_id']],critiques=[],rejected_sqls=[],initial_accepted=value['accepted'],
                stopped=False,**value),bank))
    return rows


def round_run(dest, rnd, previous, tasks, full, tables, bank, evaluator, gate='internal', model_key='strong14'):
    folder=dest/f'r{rnd}'
    if (folder/'complete.json').exists():return read(folder/'paths.jsonl')
    states=[]
    for row in previous:
        if row['accepted'] or row.get('stopped'):continue
        if row['method']=='FRESH14' and rnd>1:continue
        states.append(public_state(row,rnd))
    save(folder/'states.jsonl',states)
    print('ROUND',dest.name,rnd,'pending',dict((m,sum(s['method']==m for s in states)) for m in sorted({s['method'] for s in states})),flush=True)
    generic=[s for s in states if s['method']!='SHARE']
    first=[]
    for s in generic:
        method=ALIASES.get(s['method'],s['method'])
        stage=methods.initial_stage(method)
        messages,extra=build(tasks[s['task_id']],s,full[s['task_id']],stage)
        first.append(spec(s,stage,messages,extra,model_key,tag=dest.name))
    records=bank.invoke(folder/'generic1',first)
    by={(r['task_id'],r['method']):r for r in records};second=[]
    for s in generic:
        if methods.initial_stage(ALIASES.get(s['method'],s['method']))=='answer':continue
        messages,extra=build(tasks[s['task_id']],s,full[s['task_id']],'answer',by[s['task_id'],s['method']]['text'])
        second.append(spec(s,'answer',messages,extra,model_key,tag=dest.name))
    records+=bank.invoke(folder/'generic2',second)
    work=[s for s in states if s['method']=='SHARE'];outputs={s['task_id']:{} for s in work}
    for stage in methods.share.STAGES if work else []:
        assistant=stage!='generator_sql';calls={s['task_id']:[] for s in work};last={}
        for attempt in range(11 if assistant else 1):
            pending=[s for s in work if not calls[s['task_id']] or len(last[s['task_id']]['text'].strip())<=1]
            if not pending:break
            specs=[]
            for s in pending:
                t=tasks[s['task_id']]
                methods.validate(t,dict(s,method='SHARE_META'))
                msg,extra=methods.share.stage_messages(stage,dict(t,db_id=t['group']),s['sql'],
                    tables[t['dataset']+':'+t['group']],outputs[t['id']],{},methods.share.read_templates())
                specs.append(spec(s,stage,msg,extra,model_key,methods.share.ROLE[stage],attempt,tag=dest.name))
            for r in bank.invoke(folder/'share'/stage/f'attempt{attempt}',specs,assistant):
                calls[r['task_id']]+=r['request_ids'];last[r['task_id']]=r
        for s in work:
            tid=s['task_id'];r=dict(last[tid],request_ids=calls[tid])
            if assistant and len(r['text'].strip())<=1:r['text']='[INVALID]'
            outputs[tid][stage]=r['text'];records.append(r)
        print('STAGE',dest.name,rnd,stage,'complete',flush=True)
    grouped=defaultdict(list)
    for r in records:grouped[r['task_id'],r['method']].append(r)
    state_by={(s['task_id'],s['method']):s for s in states}
    def deliver(before):
        row={k:v for k,v in before.items() if k not in ('accepted','feedback')}
        row.update(round=rnd,carried=before['accepted'] or before.get('stopped') or (before['method']=='FRESH14' and rnd>1))
        if row['carried']:
            if not before['accepted']:row['stopped']=True;row['stop_reason']='unchanged fresh-generation prompt'
            return row
        key=before['task_id'],before['method'];s=state_by[key];steps=grouped[key]
        finals=[r for r in steps if r['stage'] in ('answer','generator_sql')]
        assert len(finals)==1
        sql=(methods.share.extract_final_sql(finals[0]['text']) if before['method']=='SHARE' else
             methods.decode(finals[0]['raw_text'],ALIASES.get(before['method'],before['method'])))
        row.update(sql=sql,observation=execute(tasks[key[0]]['db_path'],sql,timeout=10,max_rows=50000),
            call_ids=before['call_ids']+[rid for r in steps for rid in r['request_ids']],
            rejected_sqls=s['rejected_sqls'],critiques=before['critiques']+[
                'SQL:\n'+s['sql']+'\nCritique:\n'+r['text'] for r in steps if r['stage']=='feedback'])
        return account(row,bank)
    if (folder/'deliveries.jsonl').exists():deliveries=read(folder/'deliveries.jsonl')
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:deliveries=list(pool.map(deliver,previous))
        save(folder/'deliveries.jsonl',deliveries)
    # Every delivered SQL is committed before the evaluator returns a bit.
    save(folder/'deliveries_committed.json',dict(sha256=prior.batch.sha(folder/'deliveries.jsonl'),before_feedback=True))
    paths=[dict(r,**signal(evaluator,tasks[r['task_id']],r['observation'],gate)) for r in deliveries]
    for before,after in zip(previous,paths):
        assert before['task_id']==after['task_id'] and before['method']==after['method']
        if before['accepted']:assert after['accepted'] and before['call_ids']==after['call_ids']
        assert after['total_tokens']>=before['total_tokens']
    save(folder/'paths.jsonl',paths)
    save(folder/'complete.json',dict(paths_sha256=prior.batch.sha(folder/'paths.jsonl'),gate=gate))
    print('ROUND COMPLETE',dest.name,rnd,dict((m,sum(r['accepted'] for r in paths if r['method']==m)) for m in sorted({r['method'] for r in paths})),flush=True)
    return paths


def protect(dest, config, inputs):
    files=prior.implementation_files()+[Path(__file__),HERE/'evaluation.py',HERE/'PROTOCOL.md']
    files += [prior.EXP/n for n in ('iteration11/collect.py','iteration8/check_official_matching.py',
        'iteration8/sources/sql_eval/exec_eval.py','iteration8/sources/sql_eval/bird_evaluation.py')]
    inputs=list(map(Path,inputs))
    value=dict(config=config,files={str(p):prior.batch.sha(p) for p in files+inputs})
    save(dest/'prepared.json',value)
    return value


def verify(dest):
    for f,h in read(dest/'prepared.json')['files'].items():assert prior.batch.sha(f)==h,f


def report(dest, paths, tasks, evaluator, gate):
    result=summarize(paths,tasks,evaluator)
    for r in result:r['feedback_protocol']=gate
    save(dest/'summary.json',result,immutable=False)
    batches=[p.parent for p in sorted(dest.glob('**/batch/complete.json'))]
    raw=[r for folder in batches for r in prior.completed(folder).values()]
    save(dest/'accounting.json',dict(new_actual_calls=len(raw),new_input_tokens=sum(r['prompt_tokens'] for r in raw),
        new_output_tokens=sum(r['generation_tokens'] for r in raw),truncated_calls=sum(r['finish_reason']=='length' for r in raw)),immutable=False)
    return result


if __name__=='__main__':
    raise SystemExit('Use the development or confirmation entry point; no default experiment is launched.')
