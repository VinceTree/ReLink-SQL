"""Evaluate one compiled proposal followed by an unchanged native fallback."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
from . import run,contract,compile_repair as compiler

r=run.r


def phase(folder,before,deliveries,tasks,evaluator,states):
    r.save(folder/'states.jsonl',states)
    r.save(folder/'deliveries.jsonl',deliveries)
    r.save(folder/'deliveries_committed.json',dict(sha256=r.prior.batch.sha(folder/'deliveries.jsonl'),before_feedback=True))
    previous={x['task_id']:x for x in before}
    paths=[]
    for row in deliveries:
        prior=previous[row['task_id']]
        bit=({k:prior[k] for k in ('accepted','feedback')} if row['carried'] else
             r.signal(evaluator,tasks[row['task_id']],row['observation'],'internal'))
        paths.append(dict(row,**bit))
    r.save(folder/'paths.jsonl',paths)
    r.save(folder/'complete.json',dict(paths_sha256=r.prior.batch.sha(folder/'paths.jsonl'),gate='internal'))
    return paths


def main(population,policy='requirements'):
    files=run.sources(population);source=run.HERE/'results'/population/'contract'
    assert (source/'complete.json').exists(),'Complete the common requirements/native generation first'
    r.verify(source)
    name='compiled' if policy=='requirements' else 'compiled_'+policy
    dest=run.HERE/'results'/population/name
    r.protect(dest,dict(version=compiler.VERSION,population=population,policy=policy,
        cache_only=True,critic_in_candidate=False,max_tail_model_calls=2,max_tail_sql_deliveries=2,
        native_request_unchanged=True,initial_paths_include_requirements_fee=True),
        [Path(__file__),Path(compiler.__file__),Path(contract.__file__),Path(run.repair.__file__),Path(run.__file__),
         run.HERE/'COMPILE_PROTOCOL.md',files['public'],files['reference'],*files['history'],
         source/'complete.json',source/'r1/contract/records.jsonl',source/'r1/answer/logical_requests.jsonl'])
    if (dest/'complete.json').exists():return
    tasks={t['id']:t for t in r.read(files['public'])};histories=run.history_inputs(files['history'])
    evaluator=r.Evaluator(files['reference']);bank=run.previous.CacheOnlyBank(run.banks().sources.values())
    before=[dict(x,method='EXEC14',round=0) for x in r.read(files['prefix']/'r2/paths.jsonl')]
    r.save(dest/'before_planning_paths.jsonl',before)
    original={x['task_id']:r.public_state(x,1) for x in before if not x['accepted']}
    specs=[]
    for tid,state in original.items():
        messages,metadata=contract.plan_messages(tasks[tid],state)
        sp=r.spec(state,'contract',messages,metadata,tag='i15_'+name)
        sp['request']['max_tokens']=contract.PLAN_CAP;specs.append(sp)
    plans={x['task_id']:x for x in bank.invoke(dest/'planning',specs)}
    paid=[r.account(dict(x,call_ids=x['call_ids']+(plans[x['task_id']]['request_ids'] if not x['accepted'] else [])),bank) for x in before]
    r.save(dest/'initial_paths.jsonl',paid)
    folder=dest/'r1'
    if (folder/'complete.json').exists():first=r.read(folder/'paths.jsonl')
    else:
        proposals=[];by={}
        for tid in original:
            sql,meta=compiler.propose(tasks[tid],histories[tid],plans[tid]['text'],policy)
            observation=r.execute(tasks[tid]['db_path'],sql,timeout=10,max_rows=50000) if sql else None
            key=run.repair.result_key(observation) if observation else None
            repeated=key is not None and key in {run.repair.result_key(x['observation']) for x in histories[tid]}
            deliver=observation is not None and observation.get('status')=='ok' and not repeated
            proposal=dict(task_id=tid,sql=sql,observation=observation,metadata=meta,
                deliver=deliver,known_rejected_result=repeated)
            proposals.append(proposal);by[tid]=proposal
        r.save(folder/'proposals.jsonl',proposals)
        deliveries=[];states=[]
        for row in paid:
            tid=row['task_id'];p=by.get(tid)
            out={k:v for k,v in row.items() if k not in ('accepted','feedback')}
            out.update(round=1,carried=not p or not p['deliver'])
            if not out['carried']:
                states.append(r.public_state(row,1))
                out.update(sql=p['sql'],observation=p['observation'],rejected_sqls=row['rejected_sqls']+[row['sql']])
            deliveries.append(out)
        first=phase(folder,paid,deliveries,tasks,evaluator,states)
    folder=dest/'r2'
    if (folder/'complete.json').exists():second=r.read(folder/'paths.jsonl')
    else:
        states=[r.public_state(x,2) for x in first if not x['accepted']]
        requests=[];base_requests={x['task_id']:x for x in r.read(source/'r1/answer/logical_requests.jsonl')}
        for state in states:
            tid=state['task_id']
            messages,metadata=contract.answer_messages(tasks[tid],original[tid],histories[tid],plans[tid]['text'])
            assert messages==base_requests[tid]['request']['messages'],'Native fallback input changed'
            metadata=dict(metadata,compiler_fallback=True,local_input_source='original requirements/native branch')
            requests.append(r.spec(state,'answer',messages,metadata,tag='i15_'+name))
        r.save(folder/'generation_states.jsonl',[original[x['task_id']] for x in states])
        answers={x['task_id']:x for x in bank.invoke(folder/'answer',requests)}
        def deliver(before):
            out={k:v for k,v in before.items() if k not in ('accepted','feedback')}
            out.update(round=2,carried=before['accepted'])
            if out['carried']:return out
            tid=before['task_id'];sql=r.methods.decode(answers[tid]['raw_text'],'EXEC14')
            out.update(sql=sql,observation=r.execute(tasks[tid]['db_path'],sql,timeout=10,max_rows=50000),
                call_ids=before['call_ids']+answers[tid]['request_ids'],rejected_sqls=before['rejected_sqls']+[before['sql']])
            return r.account(out,bank)
        with ThreadPoolExecutor(max_workers=4) as pool:deliveries=list(pool.map(deliver,first))
        second=phase(folder,first,deliveries,tasks,evaluator,states)
    r.report(dest,first+second,tasks,evaluator,'internal');r.verify(dest)
    r.save(dest/'complete.json',dict(summary_sha256=r.prior.batch.sha(dest/'summary.json'),accounting=r.read(dest/'accounting.json')))
    for a,b in zip(before,paid):
        assert b['accepted']==a['accepted'] and b['sql']==a['sql']
        assert b['call_ids']==a['call_ids']+(plans[a['task_id']]['request_ids'] if not a['accepted'] else [])
    r.save(dest/'audit.json',dict(ledger=run.old.audit(dest,bank),planning_fees_checked=True,
                                 native_messages_checked=len(r.read(dest/'r2/states.jsonl'))))
    print('COMPILED COMPLETE',population,policy,sum(x['accepted'] for x in second),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('population',choices=['development','exploration400'])
    parser.add_argument('--policy',choices=['requirements','toggle'],default='requirements')
    args=parser.parse_args();main(args.population,args.policy)
