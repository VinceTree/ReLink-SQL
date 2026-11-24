"""One shared initial answer for all methods, using the existing paid backend."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fcntl
import os
from .dataset import HERE, DATA, r, prepare, verify

OUT=HERE/'initial'


def bank():
    folders=[r.prior.OLD/'roots/batch',r.HERE/'results/confirmation400/initial/generated/batch']
    folders += [p.parent for p in sorted(HERE.glob('**/batch/complete.json'))]
    return r.CallBank(folders)


def main():
    prepare();verify()
    with (HERE/'inference.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock.write(str(os.getpid()));lock.flush()
        r.protect(OUT,dict(initial='7_F',n=2568,shared_across_methods=True),
                  [Path(__file__),HERE/'PROTOCOL.md',DATA/'public.jsonl',DATA/'full_contexts.json'])
        r.verify(OUT)
        if (OUT/'complete.json').exists():return
        tasks=r.read(DATA/'public.jsonl');by={t['id']:t for t in tasks}
        full=r.read(DATA/'full_contexts.json');calls=bank()
        specs=[r.spec(dict(task_id=t['id'],method='INITIAL',round=0),'answer',
                      r.methods.initial_messages(dict(t,schema=full[t['id']]['schema_text'])),
                      model_key='strong',tag='full_dev_17') for t in tasks]
        print('FULL DEV INITIAL',len(specs),flush=True)
        records=calls.invoke(OUT/'generation',specs)
        def observe(record):
            tid=record['task_id'];sql=r.methods.decode(record['raw_text'],'FRESH14')
            return dict(task_id=tid,request_id=record['request_id'],sql=sql,
                        observation=r.stable(r.execute(by[tid]['db_path'],sql,timeout=10,max_rows=50000)))
        if not (OUT/'initial.jsonl').exists():
            with ThreadPoolExecutor(max_workers=4) as pool:observed=list(pool.map(observe,records))
            r.save(OUT/'initial.jsonl',observed)
        r.verify(OUT)
        r.save(OUT/'complete.json',dict(n=len(tasks),before_feedback=True,
            initial_sha256=r.prior.batch.sha(OUT/'initial.jsonl'),
            mean_tokens=sum(calls.tokens([x['request_id']]) for x in records)/len(tasks),
            reused=sum('reused_from' in x for x in records)))
        print('FULL DEV INITIAL COMPLETE',flush=True)


if __name__=='__main__':main()
