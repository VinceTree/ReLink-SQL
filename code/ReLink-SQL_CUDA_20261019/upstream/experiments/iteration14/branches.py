"""Execute repair branches with separate input histories and one global bill."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iteration13 import runner as r, analyze as old, cohort
from iteration14 import demand as d, analyze

HERE = Path(__file__).resolve().parent/'branches_v6'
VERSION = 'anchored_repair_branches_v6'
LOCAL_FIELDS = ('task_id','method','round','sql','observation','accepted','stopped','rejected_sqls','critiques')


class CacheOnlyBank(r.CallBank):
    def invoke(self, directory, specs, stopped=False):
        for sp in specs:
            assert self.key(sp['task_id'], sp['request'], stopped) in self.index, (
                'Missing exact response; no inference will be started by branch replay', sp['task_id'], sp['stage'])
        return super().invoke(directory, specs, stopped)


def history():
    return r.history_batches() + [p.parent for root in (r.HERE, Path(__file__).resolve().parent)
                                 for p in sorted((root/'results').glob('**/batch/complete.json'))]


def step(dest, rnd, macro, depth, previous, local, seen, tasks, full, bank, evaluator, memo):
    folder = dest/f'r{rnd}'
    if (folder/'complete.json').exists():
        return r.read(folder/'paths.jsonl'), r.read(folder/'local_contexts.json'), r.read(folder/'seen.json')
    method = 'EXEC14' if macro == 'direct' else 'CRITIC'
    stage = 'answer' if macro == 'direct' else 'feedback'
    build = d.make_build('direct') if macro == 'direct' else r.build
    specs, contexts, skipped = [], {}, []
    r.save(folder/'seen_before.json', seen)
    for before in previous:
        tid = before['task_id']
        if before['accepted'] or before.get('stopped'):
            continue
        s = r.public_state(local[macro][tid], rnd, method)
        messages, metadata = build(tasks[tid], s, full[tid], stage)
        sp = r.spec(s, stage, messages, dict(metadata, macro=macro, depth=depth,
                    context_origin='ReLink root' if depth==1 else 'previous answer in this branch'), tag=dest.name)
        key = bank.key(tid, sp['request'], False)
        if memo and macro == 'direct' and key in seen.get(tid, {}):
            skipped.append(dict(task_id=tid, request_key=key, previous_request_id=seen[tid][key],
                                reason='Identical direct-generation request already delivered and rejected in this path'))
            continue
        specs.append(sp);contexts[tid] = s
    states = [r.public_state(x, rnd) for x in previous if x['task_id'] in contexts]
    r.save(folder/'states.jsonl', states)
    r.save(folder/'generation_states.jsonl', list(contexts.values()))
    r.save(folder/'skipped.jsonl', skipped)
    print('BRANCH', dest.name, rnd, macro, depth, 'pending', len(specs), 'memo skips', len(skipped), flush=True)
    first = {x['task_id']:x for x in bank.invoke(folder/'first', specs)}
    first_specs = {x['task_id']:x for x in specs}
    finals, final_specs = first, first_specs
    if macro == 'critic':
        specs = []
        for tid, s in contexts.items():
            messages, metadata = build(tasks[tid], s, full[tid], 'answer', first[tid]['text'])
            specs.append(r.spec(s, 'answer', messages, dict(metadata, macro=macro, depth=depth,
                                context_origin='ReLink root'), tag=dest.name))
        finals = {x['task_id']:x for x in bank.invoke(folder/'second', specs)}
        final_specs = {x['task_id']:x for x in specs}

    def delivered(before):
        tid = before['task_id']
        row = {k:v for k,v in before.items() if k not in ('accepted','feedback')}
        row.update(round=rnd, carried=tid not in finals)
        if row['carried']:
            return row
        sql = r.methods.decode(finals[tid]['raw_text'], method)
        ids = first[tid]['request_ids'] + (finals[tid]['request_ids'] if macro=='critic' else [])
        critique = (['SQL:\n'+contexts[tid]['sql']+'\nCritique:\n'+first[tid]['text']] if macro=='critic' else [])
        row.update(sql=sql, observation=r.execute(tasks[tid]['db_path'],sql,timeout=10,max_rows=50000),
            call_ids=before['call_ids']+ids, rejected_sqls=before['rejected_sqls']+[before['sql']],
            critiques=before['critiques']+critique)
        return r.account(row, bank)

    if (folder/'deliveries.jsonl').exists():
        deliveries = r.read(folder/'deliveries.jsonl')
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            deliveries = list(pool.map(delivered, previous))
        r.save(folder/'deliveries.jsonl', deliveries)
    r.save(folder/'deliveries_committed.json', dict(sha256=r.prior.batch.sha(folder/'deliveries.jsonl'), before_feedback=True))
    by = {x['task_id']:x for x in previous}
    paths = []
    for row in deliveries:
        tid = row['task_id']
        value = ({k:by[tid][k] for k in ('accepted','feedback')} if row['carried'] else
                 r.signal(evaluator,tasks[tid],row['observation'],'internal'))
        after = dict(row, **value);paths.append(after)
        if row['carried']:
            assert after['call_ids'] == by[tid]['call_ids']
            continue
        before_local = local[macro][tid]
        critique = (['SQL:\n'+contexts[tid]['sql']+'\nCritique:\n'+first[tid]['text']] if macro=='critic' else [])
        local[macro][tid] = dict(before_local, round=depth, sql=after['sql'], observation=after['observation'],
            accepted=after['accepted'], rejected_sqls=before_local['rejected_sqls']+[before_local['sql']],
            critiques=before_local['critiques']+critique)
        if not after['accepted']:
            key = bank.key(tid, final_specs[tid]['request'], False)
            seen.setdefault(tid, {})[key] = finals[tid]['request_id']
    r.save(folder/'paths.jsonl', paths)
    r.save(folder/'local_contexts.json', local)
    r.save(folder/'seen.json', seen)
    r.save(folder/'complete.json', dict(paths_sha256=r.prior.batch.sha(folder/'paths.jsonl'), gate='internal'))
    print('BRANCH COMPLETE', dest.name, rnd, sum(x['accepted'] for x in paths), flush=True)
    return paths, local, seen


def audit_memo(dest, bank):
    previous = {x['task_id']:x for x in r.read(dest/'initial_paths.jsonl')}
    known, count = {}, 0
    for file in sorted(dest.glob('r*/paths.jsonl')):
        folder = file.parent
        assert r.read(folder/'seen_before.json') == known
        states = {x['task_id']:x for x in r.read(folder/'states.jsonl')}
        after = {x['task_id']:x for x in r.read(file)}
        for skip in r.read(folder/'skipped.jsonl'):
            tid = skip['task_id']
            assert known[tid][skip['request_key']] == skip['previous_request_id']
            assert tid not in states and not previous[tid]['accepted']
            assert after[tid]['call_ids'] == previous[tid]['call_ids']
            count += 1
        for directory in ('first','second'):
            path = folder/directory/'logical_requests.jsonl'
            if not path.exists():
                continue
            records = {x['task_id']:x for x in r.read(path.parent/'records.jsonl')}
            for sp in r.read(path):
                tid = sp['task_id']
                if sp['stage']=='answer' and not after[tid]['accepted']:
                    known.setdefault(tid,{})[bank.key(tid,sp['request'],False)] = records[tid]['request_id']
        assert r.read(folder/'seen.json') == known
        previous = after
    return dict(skipped_previously_failed_requests=count, only_prior_path_results_used=True, passed=True)


def main(population, variants):
    dev = population=='development'
    public = r.prior.OUT/'public.jsonl' if dev else cohort.DATA/'public.jsonl'
    reference = r.prior.REFERENCE if dev else cohort.DATA/'private/references.json'
    source = r.HERE/('results/development/continuation' if dev else 'results/confirmation400/internal')
    full_path = r.prior.OUT/'full_contexts.json' if dev else r.HERE/'results/confirmation400/full_contexts.json'
    r.verify(source)
    seed = [dict(x,method='CRITIC',round=0) for x in r.read(source/'r1/paths.jsonl') if x['method']=='RELINK']
    tasks = {x['id']:x for x in r.read(public)};full = r.read(full_path)
    bank = CacheOnlyBank(history());evaluator = r.Evaluator(reference)
    for variant in variants:
        dest = HERE/'results'/population/variant
        order = [('critic',1),('direct',1),('direct',2)] if variant=='critic_first' else [('direct',1),('direct',2),('critic',1)]
        memo = variant!='no_memo'
        r.protect(dest,dict(version=VERSION,variant=variant,order=order,internal_memo=memo,cache_only=True),
            [Path(__file__),Path(d.__file__),HERE/'PROTOCOL.md',public,reference,full_path,source/'r1/paths.jsonl'])
        r.verify(dest)
        if (dest/'complete.json').exists():
            continue
        r.save(dest/'initial_paths.jsonl', seed)
        previous = seed
        local = {macro:{x['task_id']:{k:x[k] for k in LOCAL_FIELDS} for x in seed} for macro in ('direct','critic')}
        seen = {};paths = []
        for rnd,(macro,depth) in enumerate(order,1):
            r.verify(dest)
            previous,local,seen = step(dest,rnd,macro,depth,previous,local,seen,tasks,full,bank,evaluator,memo)
            paths += previous;r.report(dest,paths,tasks,evaluator,'internal')
        r.verify(dest)
        r.save(dest/'complete.json',dict(summary_sha256=r.prior.batch.sha(dest/'summary.json'),accounting=r.read(dest/'accounting.json')))
        r.save(dest/'audit.json',old.audit(dest,bank));r.save(dest/'memo_audit.json',audit_memo(dest,bank))
        print('VARIANT COMPLETE',population,variant,flush=True)
    r.methods.selectors.clone(analyze.main,HERE=HERE)(population)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('population',choices=['development','exploration400'])
    parser.add_argument('--variants',nargs='+',choices=['direct_first','no_memo','critic_first'],
                        default=['direct_first','no_memo','critic_first'])
    args=parser.parse_args();main(args.population,args.variants)
