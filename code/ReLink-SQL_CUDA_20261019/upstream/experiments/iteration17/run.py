"""Full-dev experiment orchestration; reuse generation, prompts and SQL repair."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import argparse
import fcntl
import os
import time
import traceback
from . import dataset, initial, methods
from .dataset import HERE, DATA, r
from ..iteration15 import contract, multiplicity_safe, multiplicity, compile_repair, repair
from iteration14 import branches, demand

MAIN=('CURRENT','SHARE','CRITIC','SELF_REFINE','SELF_DEBUG','REFLEXION')
EXTRA=('DIN_GENTLE','CHESS_C')
GATE='official'


def inputs():
    return [Path(__file__),Path(methods.__file__),HERE/'METHODS.md',HERE/'PROTOCOL.md',
            DATA/'prepared.json',DATA/'public.jsonl',DATA/'full_contexts.json',DATA/'share_tables.json',
            DATA/'private/references.json',initial.OUT/'initial.jsonl',
            *[Path(module.__file__) for module in (contract,multiplicity_safe,multiplicity,compile_repair,repair,branches,demand)]]


def validate_transition(before,after,bank):
    assert len(before)==len(after)
    for a,b in zip(before,after):
        assert a['task_id']==b['task_id'] and a['method']==b['method']
        assert b['call_ids'][:len(a['call_ids'])]==a['call_ids']
        assert r.account(dict(b),bank)['total_tokens']==b['total_tokens']
        if a['accepted']:
            assert b['accepted'] and b['sql']==a['sql'] and b['call_ids']==a['call_ids']
        assert b['total_tokens']>=a['total_tokens']


def commit(folder,before,deliveries,tasks,evaluator,bank):
    r.save(folder/'deliveries.jsonl',deliveries)
    r.save(folder/'deliveries_committed.json',dict(
        sha256=r.prior.batch.sha(folder/'deliveries.jsonl'),before_feedback=True))
    paths=[]
    for a,b in zip(before,deliveries):
        bit=({k:a[k] for k in ('accepted','feedback')} if b['carried'] else
             r.signal(evaluator,tasks[b['task_id']],b['observation'],GATE))
        paths.append(dict(b,**bit))
    validate_transition(before,paths,bank)
    r.save(folder/'paths.jsonl',paths)
    r.save(folder/'complete.json',dict(paths_sha256=r.prior.batch.sha(folder/'paths.jsonl'),gate=GATE))
    return paths


def history(dest):
    result={}
    for path in [dest/'initial_paths.jsonl',*[dest/f'r{i}/paths.jsonl' for i in range(1,4)]]:
        for row in r.read(path):
            if not row['accepted'] and not row.get('carried'):
                result.setdefault(row['task_id'],[]).append(dict(
                    event=str(path)+'#'+row['task_id'],sql=row['sql'],observation=r.stable(row['observation'])))
    return result


def current_tail(dest,rnd,previous,tasks,bank,evaluator):
    folder=dest/f'r{rnd}'
    if (folder/'complete.json').exists():return r.read(folder/'paths.jsonl')
    original={x['task_id']:x for x in r.read(dest/'r3/paths.jsonl')}
    histories=history(dest)
    if rnd==4:
        specs=[]
        for row in previous:
            if row['accepted']:continue
            state=r.public_state(row,rnd,method='EXEC14');tid=row['task_id']
            msg,meta=contract.plan_messages(tasks[tid],state)
            sp=r.spec(dict(state,method='CURRENT'),'contract',msg,meta,tag='full_dev_17')
            sp['request']['max_tokens']=contract.PLAN_CAP;specs.append(sp)
        plans={x['task_id']:x for x in bank.invoke(folder/'planning',specs)}
        deliveries=[];proposals=[]
        for row in previous:
            tid=row['task_id'];out={k:v for k,v in row.items() if k not in ('accepted','feedback')}
            out.update(round=rnd,carried=True)
            if not row['accepted']:
                out['call_ids']=row['call_ids']+plans[tid]['request_ids']
                sql,meta=multiplicity_safe.propose(tasks[tid],histories[tid],plans[tid]['text'])
                observation=r.execute(tasks[tid]['db_path'],sql,timeout=10,max_rows=50000) if sql else None
                key=repair.result_key(observation) if observation else None
                novel=key is not None and key not in {repair.result_key(x['observation']) for x in histories[tid]}
                proposals.append(dict(task_id=tid,sql=sql,observation=r.stable(observation),metadata=meta,deliver=novel))
                if novel:out.update(sql=sql,observation=observation,carried=False,
                                    rejected_sqls=row['rejected_sqls']+[row['sql']])
                out=r.account(out,bank)
            deliveries.append(out)
        r.save(folder/'proposals.jsonl',proposals)
    elif rnd==5:
        plans={x['task_id']:x for x in r.read(dest/'r4/planning/records.jsonl')}
        specs=[]
        for row in previous:
            if row['accepted']:continue
            tid=row['task_id'];state=r.public_state(original[tid],rnd,method='EXEC14')
            msg,meta=contract.answer_messages(tasks[tid],state,histories[tid],plans[tid]['text'])
            specs.append(r.spec(dict(state,method='CURRENT'),'answer',msg,meta,tag='full_dev_17'))
        answers={x['task_id']:x for x in bank.invoke(folder/'answer',specs)}
        def deliver(row):
            out={k:v for k,v in row.items() if k not in ('accepted','feedback')}
            out.update(round=rnd,carried=row['accepted'])
            if not row['accepted']:
                tid=row['task_id'];answer=answers[tid];sql=r.methods.decode(answer['raw_text'],'EXEC14')
                out.update(sql=sql,observation=r.execute(tasks[tid]['db_path'],sql,timeout=10,max_rows=50000),
                           call_ids=row['call_ids']+answer['request_ids'],
                           rejected_sqls=row['rejected_sqls']+[row['sql']])
                out=r.account(out,bank)
            return out
        with ThreadPoolExecutor(max_workers=4) as pool:deliveries=list(pool.map(deliver,previous))
    else:raise ValueError(rnd)
    return commit(folder,previous,deliveries,tasks,evaluator,bank)


def run_round(dest,rnd,previous,tasks,full,tables,bank,evaluator):
    if previous[0]['method']!='CURRENT':
        after=methods.round_run(dest,rnd,previous,tasks,full,tables,bank,evaluator,GATE)
    elif rnd>=4:
        after=current_tail(dest,rnd,previous,tasks,bank,evaluator)
    elif rnd in (2,3):
        # Preserve the original compact direct prompt AND its paid-request memo.
        # Only the evaluation boundary changes from internal to official matching.
        runtime=SimpleNamespace(**vars(r))
        runtime.signal=lambda evaluator,task,observation,gate:r.signal(evaluator,task,observation,GATE)
        def save(path,value,**kwargs):
            if path.name=='complete.json' and isinstance(value,dict) and 'gate' in value:
                value=dict(value,gate=GATE)
            return r.save(path,value,**kwargs)
        runtime.save=save
        execute=r.methods.selectors.clone(branches.step,r=runtime)
        if rnd==2:
            local={'direct':{x['task_id']:{k:x[k] for k in branches.LOCAL_FIELDS} for x in previous}}
            seen={}
        else:
            local=r.read(dest/'r2/local_contexts.json');seen=r.read(dest/'r2/seen.json')
        after,_,_=execute(dest,rnd,'direct',rnd-1,previous,local,seen,tasks,full,bank,evaluator,True)
    else:
        def build(task,state,context,stage='answer',auxiliary=None):
            return r.build(task,dict(state,method='RELINK'),context,stage,auxiliary)
        proxy=SimpleNamespace(**vars(r.methods))
        proxy.initial_stage=lambda method:'answer' if method=='CURRENT' else r.methods.initial_stage(method)
        execute=r.methods.selectors.clone(r.round_run,methods=proxy,build=build)
        after=execute(dest,rnd,previous,tasks,full,tables,bank,evaluator,GATE)
    validate_transition(previous,after,bank)
    return after


def prepare_group(cohort,method,tasks,evaluator,bank,initial_answers):
    dest=HERE/'results'/cohort/method
    limit=5 if method=='CURRENT' else (2 if cohort=='smoke32' else 8)
    r.protect(dest,dict(cohort=cohort,method=method,gate=GATE,max_rounds=limit,
                       development_benchmark=True),inputs())
    r.verify(dest)
    start=r.initial_paths(list(tasks.values()),initial_answers,[method],evaluator,bank,GATE)
    r.save(dest/'initial_paths.jsonl',start)
    return dest,start


def main(mode,wait=False):
    while not (initial.OUT/'complete.json').exists():
        if not wait:raise RuntimeError('Shared initial generation is not complete')
        time.sleep(30)
    with (HERE/'inference.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock.write(str(os.getpid()));lock.flush()
        dataset.verify()
        assert r.prior.batch.sha(initial.OUT/'initial.jsonl')==r.read(initial.OUT/'complete.json')['initial_sha256']
        ref=DATA/'private/references.json';ref_marker=r.read(DATA/'private/complete.json')
        assert r.prior.batch.sha(ref)==ref_marker['sha256']
        public=r.read(DATA/'public.jsonl');valid=[t for t in public if t['id'] not in ref_marker['unscorable']]
        full=r.read(DATA/'full_contexts.json');tables=r.read(DATA/'share_tables.json')
        evaluator=r.Evaluator(ref);bank=initial.bank()
        initial_answers={x['task_id']:x for x in r.read(initial.OUT/'initial.jsonl')}
        if mode=='smoke':
            # Interface smoke uses previously developed questions and no outcome filtering.
            old={t['id'] for t in r.read(r.HERE/'data/confirmation400/public.jsonl')}
            valid=[t for ds in ('spider','bird') for t in sorted(
                [x for x in valid if x['dataset']==ds and x['id'] in old],
                key=lambda x:r.prior.batch.digest(['full_dev_17_smoke',x['id']]))[:16]]
            cohorts={'smoke32':{t['id']:t for t in valid}}
            arms=('SELF_DEBUG','REFLEXION','CURRENT')
        else:
            cohorts={ds:{t['id']:t for t in valid if t['dataset']==ds} for ds in ('spider','bird')}
            arms=MAIN+EXTRA
        state={}
        for cohort,tasks in cohorts.items():
            for method in arms:
                dest,start=prepare_group(cohort,method,tasks,evaluator,bank,initial_answers)
                state[cohort,method]=(dest,start)
        failed=set()
        for rnd in range(1,9):
            for cohort,tasks in cohorts.items():
                for method in arms:
                    if method=='CURRENT' and rnd>5:continue
                    dest,previous=state[cohort,method]
                    if (cohort,method) in failed or (dest/'complete.json').exists():continue
                    try:
                        r.verify(dest)
                        after=run_round(dest,rnd,previous,tasks,full,tables,bank,evaluator)
                        state[cohort,method]=(dest,after)
                        paths=[x for i in range(1,rnd+1) for x in r.read(dest/f'r{i}/paths.jsonl')]
                        r.report(dest,paths,tasks,evaluator,GATE)
                        if (dest/'error.json').exists():(dest/'error.json').unlink()
                        limit=5 if method=='CURRENT' else (2 if mode=='smoke' else 8)
                        if rnd==limit or all(x['accepted'] for x in after):
                            r.verify(dest)
                            r.save(dest/'complete.json',dict(round=rnd,summary_sha256=r.prior.batch.sha(dest/'summary.json'),
                                stop_reason='all_accepted' if all(x['accepted'] for x in after) else 'fixed_round_budget'))
                    except Exception as exc:
                        r.save(dest/'error.json',dict(round=rnd,error=type(exc).__name__,message=str(exc),
                               traceback=traceback.format_exc()),immutable=False)
                        failed.add((cohort,method));traceback.print_exc()
                    from .report import generate
                    generate()
        if failed:raise RuntimeError('Incomplete method/cohort pairs: '+str(sorted(failed)))
        r.save(HERE/(mode+'_complete.json'),dict(method_cohorts=len(state),
            results={str(dest.relative_to(HERE)):r.prior.batch.sha(dest/'complete.json') for dest,_ in state.values()},
            cyclesql='pending unavailable author NLI checkpoint; not substituted'))
        print('COMPLETED',mode,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','full'])
    parser.add_argument('--wait-initial',action='store_true');args=parser.parse_args()
    main(args.mode,args.wait_initial)
