"""Collect real six-way initial choices and their conditional recovery branches.

Only score() opens references. Preparation never loads language-model weights.
Existing generation is allowed to finish before this experiment uses the GPU.
"""
from pathlib import Path
from functools import lru_cache
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iteration8.cohort import read, write, sha
from iteration4 import batch_public_generate as batch
from iteration4.collect_confirmation import MODELS, initial_messages, completed
from iteration8 import input_selectors as selectors
from iteration11 import contexts
from recoverylab.sql import execute, result_equal

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OUT = HERE / 'results/pilot128_v1'
OLD = EXP / 'iteration8/results/main1000'
EXT = EXP / 'iteration10/results/main1000'
MODEL = {'3': MODELS['weak'], '7': MODELS['strong'], '14': MODELS['strong14']}
ROOTS = [m + '_' + c for m in MODEL for c in ('F', 'S')]
RECOVERY = ['STOP', 'SAME_V', '14_V', '14_F']
CACHE_PAIRS = [
    (OLD / 'weak_requests.jsonl', OLD / 'weak_batch'),
    (OLD / 'stage1_requests.jsonl', OLD / 'stage1'),
    (EXT / 'generic/p1/requests.jsonl', EXT / 'generic/p1/batch'),
]


def signature(request):
    return batch.digest([request['id'].split('|')[0], request['model'], request['revision'],
                         request['messages'], request['max_tokens']])


@lru_cache(maxsize=3)
def tokenizer(model_key):
    from transformers import AutoTokenizer
    metadata = batch.model_metadata(*MODEL[model_key])
    return AutoTokenizer.from_pretrained(metadata['local_path'], local_files_only=True,
                                        trust_remote_code=False)


def request(task, name, model_key, context):
    model, revision = MODEL[model_key]
    return dict(id=task['id'] + '|I11_' + name, model=model, revision=revision,
                messages=initial_messages(dict(task, schema=context['schema_text'])), max_tokens=1024)


def input_tokens(req, model_key):
    prepared = batch.prepare_prompt(req, tokenizer(model_key), 'stream_string', 24000)
    return prepared['prompt_tokens']


def store_phase(phase, requests):
    write(OUT / phase / 'requests.jsonl', requests, True)
    if requests:
        marker = OUT / phase / 'batch/manifest.json'
        if marker.exists():
            assert json.loads(marker.read_text())['requests_sha256'] == batch.digest(requests)
        else:
            print(batch.prepare(OUT / phase / 'requests.jsonl', OUT / phase / 'batch'), flush=True)
    else:
        write(OUT / phase / 'batch/results.jsonl', [], True)
        write(OUT / phase / 'batch/complete.json', dict(items=0))


def available():
    """Exact public request cache; never reuse a different first branch's evidence."""
    seen, raw, sources = {}, {}, {}
    for req_file, directory in CACHE_PAIRS:
        assert (directory / 'complete.json').exists(), directory
        rs = completed(directory)
        for r in read(req_file):
            if r['id'] in rs:
                sig = signature(r)
                seen[sig] = r['id']
                raw[r['id']] = rs[r['id']]
                sources[r['id']] = dict(requests=str(req_file), results=str(directory / 'results.jsonl'))
    for phase in ('roots', 'recovery'):
        f = OUT / phase / 'requests.jsonl'
        if f.exists():
            for r in read(f):
                seen[signature(r)] = r['id']
            d = OUT / phase / 'batch'
            if (d / 'complete.json').exists():
                raw.update(completed(d))
    return seen, raw, sources


def prepare():
    if (OUT / 'prepared.json').exists():
        verify()
        return
    ids = json.loads((EXT / 'all128_ids.json').read_text())
    public = {t['id']: t for t in read(EXT / 'public.jsonl')}
    ts = [public[i] for i in ids]
    assert len(ts) == len(set(ids)) == 128
    assert all(t['dataset'] in ('spider', 'bird') for t in ts)
    write(OUT / 'public.jsonl', ts, True)
    seen, raw, sources = available()
    requests, mapping, ctxs, features, reuse = [], [], {}, {}, {}
    for t in ts:
        tid = t['id']
        cs = dict(F=selectors.full(t), S=contexts.first_context(t))
        ctxs[tid] = cs
        for arm in ROOTS:
            m, mode = arm.split('_')
            req = request(t, 'ROOT_' + arm, m, cs[mode])
            sig = signature(req)
            if sig not in seen:
                seen[sig] = req['id']
                requests.append(req)
            rid = seen[sig]
            mapping.append(dict(task_id=tid, root=arm, request_id=rid, signature=sig))
            prompt_n = input_tokens(req, m)
            features[tid + '|' + arm] = contexts.root_features(t, cs[mode], cs['F'], m, prompt_n)
            if rid in raw:
                assert raw[rid]['prompt_tokens'] == prompt_n, (rid, prompt_n)
                reuse[rid] = dict(sources[rid], raw_sha256=batch.digest(raw[rid]))
    write(OUT / 'root_contexts.json', ctxs)
    write(OUT / 'root_mapping.jsonl', mapping, True)
    write(OUT / 'root_features.json', features)
    write(OUT / 'reused_sources.json', reuse)
    dependencies = [HERE / n for n in ('PROTOCOL.md', 'contexts.py', 'collect.py')]
    dependencies += [EXP / n for n in ('iteration3/repair_index.py', 'iteration3/column_context.py',
                                      'iteration8/input_selectors.py', 'iteration4/batch_public_generate.py',
                                      'iteration4/collect_confirmation.py', 'iteration2/collect.py',
                                      'recoverylab/sql.py')]
    source_files = [EXT / 'all128_ids.json', EXT / 'public.jsonl']
    source_files += [p for req, directory in CACHE_PAIRS for p in (req, directory / 'results.jsonl', directory / 'complete.json')]
    store_phase('roots', requests)
    own = ['public.jsonl', 'root_contexts.json', 'root_mapping.jsonl', 'root_features.json',
           'reused_sources.json', 'roots/requests.jsonl']
    write(OUT / 'prepared.json', dict(
        scope='Historical development cohort; no labels used to choose or prepare', n=128,
        initial_actions=ROOTS, model_versions=MODEL, new_root_calls=len(requests),
        cached_initial_requests=len(reuse), references_read=False, reserved400_read=False,
        files={str(p): sha(p) for p in dependencies + source_files + [OUT / n for n in own]},
    ))
    print('PREPARED', len(requests), 'new first calls;', len(reuse), 'cached requests', flush=True)


def verify():
    marker = json.loads((OUT / 'prepared.json').read_text())
    for p, expected in marker['files'].items():
        assert sha(p) == expected, p


def observe(task, raw):
    return dict(raw_text=raw['raw_text'], text=raw['raw_text'],
                observation=execute(task['db_path'], raw['raw_text'], timeout=10, max_rows=50000),
                prompt_tokens=raw['prompt_tokens'], generation_tokens=raw['generation_tokens'],
                finish_reason=raw['finish_reason'],
                **{k: raw.get(k) for k in ('logprob_mean', 'logprob_min', 'logprob_q10', 'low_confidence_fraction')})


def delivered_path(task, state, action, observed, raw):
    rid = state['request_id']
    crid = state['candidates'][action]['request_id']
    initial, candidate = observed[rid], observed[crid]
    calls = [rid] if crid == rid else [rid, crid]
    rollback = initial['observation']['status'] == 'ok' and candidate['observation']['status'] != 'ok'
    selected = rid if rollback else crid
    inp = sum(raw[r]['prompt_tokens'] for r in calls)
    out = sum(raw[r]['generation_tokens'] for r in calls)
    return dict(task_id=state['task_id'], dataset=task['dataset'], group=task['group'],
                root=state['root'], action=action, initial_request_id=rid,
                candidate_request_id=crid, selected_request_id=selected,
                call_ids=calls, rollback=rollback, prompt_tokens=inp,
                generation_tokens=out, total_tokens=inp + out,
                initial_tokens=initial['prompt_tokens'] + initial['generation_tokens'],
                extra_tokens=0 if crid == rid else candidate['prompt_tokens'] + candidate['generation_tokens'])


def roots_done():
    verify()
    if (OUT / 'roots_committed.json').exists():
        return
    ts = {t['id']: t for t in read(OUT / 'public.jsonl')}
    cs = json.loads((OUT / 'root_contexts.json').read_text())
    root_features = json.loads((OUT / 'root_features.json').read_text())
    mapping = read(OUT / 'root_mapping.jsonl')
    seen, raw, sources = available()
    jobs = {(m['task_id'], m['request_id']) for m in mapping}
    def run(job):
        tid, rid = job
        return rid, observe(ts[tid], raw[rid])
    observation_path = OUT / 'root_observations.json'
    if observation_path.exists():
        # This checkpoint can precede a failed recovery-context compilation.
        # Reuse it without changing execution outcomes or timing metadata.
        observed = json.loads(observation_path.read_text())
        assert set(observed) == {rid for _, rid in jobs}
        for rid, result in observed.items():
            for key in ('raw_text', 'prompt_tokens', 'generation_tokens', 'finish_reason',
                        'logprob_mean', 'logprob_min', 'logprob_q10', 'low_confidence_fraction'):
                assert result.get(key) == raw[rid].get(key), (rid, key)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            observed = dict(ex.map(run, sorted(jobs)))
        write(observation_path, observed)
    new, states = [], []
    for mp in mapping:
        tid, arm, rid = mp['task_id'], mp['root'], mp['request_id']
        t, initial = ts[tid], observed[rid]
        model_key, mode = arm.split('_')
        view = contexts.recovery_context(t, initial)
        state = dict(mp, base_features=root_features[tid + '|' + arm],
                     observed_features=contexts.observed_features(initial, cs[tid][mode], view),
                     recovery_context=view, candidates={})
        for action in RECOVERY:
            if action == 'STOP':
                candidate_rid, prompt_n, sig = rid, 0, mp['signature']
            else:
                m = model_key if action == 'SAME_V' else '14'
                context = cs[tid]['F'] if action == '14_F' else view
                req = request(t, arm + '_' + action, m, context)
                sig = signature(req)
                if sig not in seen:
                    seen[sig] = req['id']
                    new.append(req)
                candidate_rid = seen[sig]
                prompt_n = 0 if candidate_rid == rid else input_tokens(req, m)
            state['candidates'][action] = dict(request_id=candidate_rid, input_tokens=prompt_n,
                                              same_as_initial=candidate_rid == rid, signature=sig)
        states.append(state)
    write(OUT / 'states.jsonl', states, True)
    store_phase('recovery', new)
    write(OUT / 'roots_committed.json', dict(initial_states=len(states), new_recovery_calls=len(new),
        references_read=False, files={n: sha(OUT / n) for n in
        ('root_observations.json', 'states.jsonl', 'recovery/requests.jsonl', 'roots/batch/results.jsonl')}))
    print('ROOTS COMMITTED', len(states), 'states;', len(new), 'new conditional calls', flush=True)


def commit():
    verify()
    if (OUT / 'committed.json').exists():
        return
    m = json.loads((OUT / 'roots_committed.json').read_text())
    for n, h in m['files'].items():
        assert sha(OUT / n) == h, n
    ts = {t['id']: t for t in read(OUT / 'public.jsonl')}
    states = read(OUT / 'states.jsonl')
    _, raw, _ = available()
    observed = json.loads((OUT / 'root_observations.json').read_text())
    jobs = {(s['task_id'], c['request_id']) for s in states for c in s['candidates'].values()
            if c['request_id'] not in observed}
    def run(job):
        tid, rid = job
        return rid, observe(ts[tid], raw[rid])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        observed.update(dict(ex.map(run, sorted(jobs))))
    paths = []
    for s in states:
        t = ts[s['task_id']]
        for action in s['candidates']:
            paths.append(delivered_path(t, s, action, observed, raw))
    write(OUT / 'candidate_observations.json', observed)
    write(OUT / 'public_paths.jsonl', paths, True)
    generated = [r for phase in ('roots', 'recovery') for r in completed(OUT / phase / 'batch').values()]
    write(OUT / 'committed.json', dict(references_read=False, path_rows=len(paths),
        new_llm_calls=len(generated), new_input_tokens=sum(r['prompt_tokens'] for r in generated),
        new_output_tokens=sum(r['generation_tokens'] for r in generated),
        truncated_new_calls=sum(r['finish_reason'] == 'length' for r in generated),
        files={n: sha(OUT / n) for n in ('candidate_observations.json', 'public_paths.jsonl',
                                       'states.jsonl', 'roots/batch/results.jsonl', 'recovery/batch/results.jsonl')}))
    print('PUBLIC PATHS COMMITTED', len(paths), flush=True)


def score():
    verify()
    if (OUT / 'scored_complete.json').exists():
        return
    m = json.loads((OUT / 'committed.json').read_text())
    for n, h in m['files'].items():
        assert sha(OUT / n) == h, n
    path = EXP / 'iteration8/data/main1000/private/references.json'
    assert sha(path) == json.loads((OLD / 'score_complete.json').read_text())['reference_sha256']
    refs = json.loads(path.read_text())
    observed = json.loads((OUT / 'candidate_observations.json').read_text())
    rows = read(OUT / 'public_paths.jsonl')
    for row in rows:
        g = refs[row['task_id']]
        row['reference_ok'] = g['gold_result']['status'] == 'ok'
        for source, target in [('selected_request_id', 'correct'), ('initial_request_id', 'initial_correct'),
                               ('candidate_request_id', 'candidate_correct')]:
            row[target] = bool(result_equal(observed[row[source]]['observation'], g['gold_result'],
                                            g['order_matters'])) if row['reference_ok'] else None
    write(OUT / 'scored_paths.jsonl', rows, True)
    write(OUT / 'scored_complete.json', dict(scope='Development labels, not held-out confirmation',
        files={'scored_paths.jsonl': sha(OUT / 'scored_paths.jsonl')}, reference_sha256=sha(path),
        unscorable=sorted({r['task_id'] for r in rows if not r['reference_ok']})))
    print('SCORED', len(rows), flush=True)


def run():
    prepare()
    # These markers are written only after all old generation stages finish.
    while not all((EXT / n).exists() for n in ('generic_complete.json', 'share_complete.json')):
        print('Waiting for existing iteration10 generation to finish', flush=True)
        time.sleep(30)
    previous = {n: sha(EXT / n) for n in ('generic_complete.json', 'share_complete.json')}
    start_marker = OUT / 'generation_start.json'
    if start_marker.exists():
        assert json.loads(start_marker.read_text())['previous_generation'] == previous
    else:
        write(start_marker, dict(previous_generation=previous, pid=os.getpid(), gpu_overlap_with_iteration10=False))
    for phase, after in [('roots', roots_done), ('recovery', commit)]:
        if not (OUT / phase / 'batch/complete.json').exists():
            subprocess.run([sys.executable, str(EXP / 'iteration4/batch_public_generate.py'),
                            'run', '--output', str(OUT / phase / 'batch')], check=True)
        after()
    score()
    subprocess.run([sys.executable, str(HERE / 'learn.py')], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'roots_done', 'commit', 'score', 'run'])
    args = parser.parse_args()
    globals()[args.phase]()
