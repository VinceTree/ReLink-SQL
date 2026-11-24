"""Run an independently implemented repair tail with the common evaluator."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from iteration13 import runner as r, analyze as old, cohort
from iteration14 import branches as previous
from iteration15 import repair

HERE=Path(__file__).resolve().parent


def sources(population):
    dev=population=='development'
    prefix=previous.HERE/'results'/population/'direct_first'
    original=r.HERE/('results/development/continuation' if dev else 'results/confirmation400/internal')
    return dict(prefix=prefix,public=(r.prior.OUT if dev else cohort.DATA)/'public.jsonl',
        reference=r.prior.REFERENCE if dev else cohort.DATA/'private/references.json',
        full=r.prior.OUT/'full_contexts.json' if dev else r.HERE/'results/confirmation400/full_contexts.json',
        history=[original/'initial_paths.jsonl',prefix/'initial_paths.jsonl',prefix/'r1/paths.jsonl',prefix/'r2/paths.jsonl'])


def failed_event(row,source):
    return dict(event=str(source)+'#'+row['task_id'],sql=row['sql'],observation=r.stable(row['observation']))


def history_inputs(files):
    histories={}
    for index,file in enumerate(files):
        rows=r.read(file)
        if index==0:
            rows=[x for x in rows if x['method']=='RELINK']
        for row in rows:
            if row['accepted'] or row.get('carried'):
                continue
            histories.setdefault(row['task_id'],[]).append(failed_event(row,file))
    return histories


def banks():
    return r.CallBank(previous.history()+[p.parent for root in (HERE,previous.HERE)
                     for p in sorted(root.glob('**/batch/complete.json'))])


def main(population,variants,rounds=2):
    files=sources(population);r.verify(files['prefix'])
    tasks={t['id']:t for t in r.read(files['public'])};full=r.read(files['full'])
    seed=[dict(x,method='EXEC14',round=0) for x in r.read(files['prefix']/'r2/paths.jsonl')]
    assert not any(x['critiques'] for x in seed)
    evaluator=r.Evaluator(files['reference']);bank=banks()
    for variant in variants:
        dest=HERE/'results'/population/variant
        r.protect(dest,dict(version=repair.VERSION,variant=variant,population=population,
            max_additional_sql_calls=rounds,critic_in_candidate=False,seed='common ReLink and two direct modifications'),
            [Path(__file__),Path(repair.__file__),HERE/'PROTOCOL.md',files['public'],files['reference'],files['full'],*files['history']])
        r.verify(dest)
        if (dest/'complete.json').exists():
            continue
        histories=history_inputs(files['history'])
        r.save(dest/'initial_paths.jsonl',seed)
        previous_rows=seed;paths=[]
        def build(task,state,full,stage='answer',auxiliary=None):
            assert stage=='answer' and auxiliary is None
            return repair.build(task,state,full,histories[task['id']],variant)
        round_run=r.methods.selectors.clone(r.round_run,build=build)
        for rnd in range(1,rounds+1):
            r.verify(dest)
            previous_rows=round_run(dest,rnd,previous_rows,tasks,full,{},bank,evaluator)
            for row in previous_rows:
                if not row['accepted'] and not row.get('carried'):
                    histories.setdefault(row['task_id'],[]).append(failed_event(row,dest/f'r{rnd}/paths.jsonl'))
            paths+=previous_rows;r.report(dest,paths,tasks,evaluator,'internal')
        r.verify(dest)
        r.save(dest/'complete.json',dict(summary_sha256=r.prior.batch.sha(dest/'summary.json'),accounting=r.read(dest/'accounting.json')))
        r.save(dest/'audit.json',old.audit(dest,bank))
        print('VARIANT COMPLETE',population,variant,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('population',choices=['development','exploration400'])
    parser.add_argument('--variants',nargs='+',choices=['effects','no_checks','direct'],default=['effects','direct'])
    args=parser.parse_args();main(args.population,args.variants)
