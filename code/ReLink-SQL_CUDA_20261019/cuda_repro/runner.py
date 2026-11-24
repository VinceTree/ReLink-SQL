"""Fixed Spider subset, unchanged repair stages, and independent CUDA provenance."""
import argparse
from contextlib import contextmanager
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import uuid

from .common import ROOT, digest, output_path, read, save, sha, verify_upstream
from .upstream_bridge import install

# Keep the package layout and relative imports of the original entry point.
sys.path.insert(0, str(ROOT / 'upstream'))
install()
from experiments.iteration17 import run as engine
from recoverylab.sql import connect

r = engine.r
MISSING_PROTOCOLS = ('experiments/iteration12/PROTOCOL.md',
                     'experiments/iteration13/PROTOCOL.md')
PACKAGES = ('numpy', 'sqlglot', 'torch', 'transformers', 'accelerate',
            'bitsandbytes', 'huggingface-hub', 'tokenizers', 'safetensors')


def environment():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, platform=platform.platform(), packages=packages)


@contextmanager
def run_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def source_files():
    files = [ROOT / 'upstream_manifest.json', ROOT / 'PROTOCOL_CUDA.md',
             ROOT / 'requirements.cuda.txt']
    files += sorted((ROOT / 'cuda_repro').glob('*.py'))
    files += [ROOT / 'upstream' / name for name in read(ROOT / 'upstream_manifest.json')['files']]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def seal(directory):
    """Freeze completed stage artifacts, including resume-only local contexts."""
    target = directory / 'cuda_artifacts.json'
    files = {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p != target}
    save(target, files)


def verify_seals(output):
    for target in sorted(output.glob('**/cuda_artifacts.json')):
        files = {str(p.relative_to(target.parent)): sha(p) for p in sorted(target.parent.rglob('*'))
                 if p.is_file() and p != target}
        if files != read(target):
            raise ValueError('Changed completed stage artifact: ' + str(target.parent))


def select_rows(rows, size):
    if type(size) is not int or not 1 <= size <= len(rows):
        raise ValueError('Sample size must be between 1 and the number of dev rows')
    # Selection is frozen before gold execution or any model output is observed.
    order = sorted(range(len(rows)), key=lambda i: digest(
        ['relink-cuda-spider-subset-v1', i, rows[i]['db_id'], rows[i]['question']]))
    return order[:size]


def prepare(config_path, dev_path, database_dir, output, size=10):
    from .backend import validate_config
    output = output_path(output)
    config_path, dev_path = Path(config_path).resolve(), Path(dev_path).resolve()
    database_dir = Path(database_dir).resolve()
    with run_lock(output):
        if (output / 'prepared.json').exists():
            marker = verify(output)
            desired = (str(config_path), str(dev_path), str(database_dir), size)
            actual = tuple(marker[k] for k in ('config_path', 'dev_path', 'database_dir', 'sample_size'))
            if actual != desired:
                raise ValueError('Existing run uses different input paths/sample size; choose a new output')
            return marker
        verify_upstream()
        config = read(config_path)
        if 'run_fingerprint' in config:
            raise ValueError('Input config must not supply a run_fingerprint')
        model_inventory = validate_config(config)
        rows = read(dev_path)
        if not isinstance(rows, list) or len(rows) != 1034:
            raise ValueError('Expected Spider 1.0 dev.json with 1034 rows, not train/test/Spider 2.0')
        if any(not isinstance(row, dict) or not {'db_id', 'question', 'query'} <= set(row)
               or not all(isinstance(row[k], str) for k in ('db_id', 'question', 'query')) for row in rows):
            raise ValueError('Invalid Spider dev row')
        indices = select_rows(rows, size)
        tasks, selections, db_hashes, contexts, private_sql = [], [], {}, {}, {}
        for index in indices:
            row = rows[index]
            db_id = row['db_id']
            if not db_id or Path(db_id).name != db_id or db_id in ('.', '..'):
                raise ValueError('Invalid database identity')
            db = database_dir / db_id / (db_id + '.sqlite')
            if not db.is_file():
                raise FileNotFoundError(db)
            if str(db) not in db_hashes:
                db_hashes[str(db)] = sha(db)
            with connect(db) as con:
                if con.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                    raise ValueError('SQLite quick_check failed: ' + str(db))
                ddl = [x[0] for x in con.execute(
                    "SELECT sql FROM sqlite_master WHERE type IN ('table','view') "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name") if x[0]]
            tid = f'spider_cuda_dev:{index}:{digest(row["question"])[:12]}'
            task = dict(id=tid, dataset='spider', group=db_id, split='dev', kind='sql',
                        question=row['question'], evidence=row.get('evidence', ''),
                        db_path=str(db), schema='\n\n'.join(x.rstrip(';') + ';' for x in ddl))
            if set(task) != r.methods.TASK_FIELDS:
                raise ValueError('Unexpected public task fields')
            tasks.append(task)
            contexts[tid] = r.contexts.old.full(task)
            selections.append(dict(id=tid, source_row=index, db_id=db_id,
                                   question_sha256=digest(row['question'])))
            private_sql[tid] = row['query']
        save(output / 'public.jsonl', tasks)
        save(output / 'selection.json', dict(seed='relink-cuda-spider-subset-v1',
             no_outcome_selection=True, no_replacement=True, total_dev_rows=len(rows), rows=selections))
        save(output / 'full_contexts.json', contexts)
        # Gold SQL is stored separately and is never placed in task/spec dictionaries.
        references = {}
        for task in tasks:
            sql = private_sql[task['id']]
            observed = r.stable(r.execute(task['db_path'], sql, timeout=10, max_rows=50000))
            references[task['id']] = dict(gold_sql=sql, gold_result=observed,
                                          order_matters='order by' in sql.lower())
        save(output / 'private/references.json', references)
        unscorable = {tid: ref['gold_result'].get('error', 'unknown')
                      for tid, ref in references.items() if ref['gold_result']['status'] != 'ok'}
        save(output / 'private/reference_audit.json', dict(n=size, unscorable=unscorable, no_replacement=True))
        marker = dict(protocol='isolated-cuda-subset-v1', config=config,
                      config_path=str(config_path), dev_path=str(dev_path), database_dir=str(database_dir),
                      sample_size=size, selected_ids=[t['id'] for t in tasks],
                      unscorable=unscorable, missing_historical_inputs=list(MISSING_PROTOCOLS),
                      source_files=source_files(), environment=environment(), model_inventory=model_inventory,
                      external_files={str(config_path): sha(config_path), str(dev_path): sha(dev_path), **db_hashes},
                      artifacts={name: sha(output / name) for name in (
                          'public.jsonl', 'selection.json', 'full_contexts.json',
                          'private/references.json', 'private/reference_audit.json')})
        marker['run_fingerprint'] = digest(marker)
        save(output / 'prepared.json', marker)
        return marker


def verify(output):
    from .backend import validate_config
    output = output_path(output)
    marker = read(output / 'prepared.json')
    if digest({k: v for k, v in marker.items() if k != 'run_fingerprint'}) != marker['run_fingerprint']:
        raise ValueError('Run manifest hash mismatch')
    verify_upstream()
    if source_files() != marker['source_files']:
        raise ValueError('Code/protocol changed; create a new run directory')
    if environment() != marker['environment']:
        raise ValueError('Python/dependencies changed; create a new run directory')
    if validate_config(marker['config']) != marker['model_inventory']:
        raise ValueError('Model files changed; create a new run directory')
    for name, expected in marker['external_files'].items():
        if sha(name) != expected:
            raise ValueError('Changed external input: ' + name)
    for name, expected in marker['artifacts'].items():
        if sha(output / name) != expected:
            raise ValueError('Changed prepared artifact: ' + name)
    verify_seals(output)
    return marker


def configure_models(config):
    for key in ('strong', 'strong14'):
        model = config['models'][key]
        r.methods.MODELS[key] = (model['repo'], model['revision'])


def execute_run(output, bank_factory=None, check_gpu=True):
    from .backend import CudaCallBank
    output = output_path(output)
    with run_lock(output):
        marker = verify(output)
        config = dict(marker['config'], run_fingerprint=marker['run_fingerprint'])
        configure_models(config)
        if check_gpu:
            import torch
            if sys.version_info[:2] != (3, 11) or not torch.cuda.is_available():
                raise RuntimeError('Use the accepted Python 3.11 CUDA environment')
            i = config['device']
            runtime = dict(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                           gpu=torch.cuda.get_device_name(i), capability=list(torch.cuda.get_device_capability(i)),
                           cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                           python=sys.version, packages=environment()['packages'])
            save(output / 'runtime.json', runtime)
        bank = (bank_factory or CudaCallBank)(config, output)
        tasks = read(output / 'public.jsonl')
        valid = {t['id']: t for t in tasks if t['id'] not in marker['unscorable']}
        if not valid:
            raise ValueError('No scorable selected reference; no model calls will be made')
        if (output / 'complete.json').exists():
            return audit(output)
        evaluator = r.Evaluator(output / 'private/references.json')
        full = read(output / 'full_contexts.json')
        specs = [r.spec(dict(task_id=t['id'], method='INITIAL', round=0), 'answer',
                        r.methods.initial_messages(dict(t, schema=full[t['id']]['schema_text'])),
                        model_key='strong', tag='cuda_subset') for t in tasks]
        records = bank.invoke(output / 'initial/generation', specs)
        by_task = {t['id']: t for t in tasks}
        observations = output / 'initial/observations.jsonl'
        if observations.exists():
            observed = read(observations)
            if [(x['task_id'], x['request_id']) for x in observed] != [
                    (x['task_id'], x['request_id']) for x in records]:
                raise ValueError('Initial observation/request identities differ')
        else:
            observed = [dict(task_id=rec['task_id'], request_id=rec['request_id'],
                         sql=r.methods.decode(rec['raw_text'], 'FRESH14'),
                         observation=r.stable(r.execute(by_task[rec['task_id']]['db_path'],
                              r.methods.decode(rec['raw_text'], 'FRESH14'), timeout=10, max_rows=50000)))
                        for rec in records]
            save(observations, observed)
        seal(output / 'initial')
        initial = {row['task_id']: row for row in observed}
        dest = output / 'CURRENT'
        previous = r.initial_paths(list(valid.values()), initial, ['CURRENT'], evaluator, bank, 'official')
        save(dest / 'initial_paths.jsonl', previous)
        final_round = 0
        try:
            for rnd in range(1, 6):
                if all(row['accepted'] for row in previous):
                    break
                verify(output)
                previous = engine.run_round(dest, rnd, previous, valid, full, {}, bank, evaluator)
                seal(dest / f'r{rnd}')
                final_round = rnd
            save(output / 'final_paths.jsonl', previous)
            summary = dict(selected_n=len(tasks), scorable_n=len(valid),
                           excluded_n=len(marker['unscorable']),
                           initial_correct=sum(row['initial_accepted'] for row in previous),
                           final_correct=sum(row['accepted'] for row in previous), final_round=final_round,
                           path_tokens=sum(row['total_tokens'] for row in previous),
                           all_initial_tokens=bank.tokens([row['request_id'] for row in observed]),
                           feedback='pinned official single-database execution matching',
                           independent_test=False, small_sample_accuracy_is_not_paper_reproduction=True)
            save(output / 'summary.json', summary)
            save(output / 'complete.json', dict(run_fingerprint=marker['run_fingerprint'],
                 final_paths_sha256=sha(output / 'final_paths.jsonl'), summary_sha256=sha(output / 'summary.json'),
                 initial_paths_sha256=sha(dest / 'initial_paths.jsonl'),
                 ledger_events_sha256=sha(output / 'ledger/events.jsonl'),
                 final_round=final_round, stop_reason='all_accepted' if all(x['accepted'] for x in previous)
                 else 'fixed_five_stage_budget'))
        except BaseException as exc:
            save(output / 'errors' / (uuid.uuid4().hex + '.json'), dict(
                 type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
            raise
        return summary


def audit(output):
    from .backend import CudaCallBank
    output = output_path(output)
    marker = verify(output)
    complete = read(output / 'complete.json')
    if complete['run_fingerprint'] != marker['run_fingerprint']:
        raise ValueError('Completion marker belongs to another run')
    for name in ('final_paths', 'summary'):
        if sha(output / (name + ('.jsonl' if name == 'final_paths' else '.json'))) != complete[name + '_sha256']:
            raise ValueError('Completion hash mismatch: ' + name)
    if sha(output / 'CURRENT/initial_paths.jsonl') != complete['initial_paths_sha256']:
        raise ValueError('Initial paths hash mismatch')
    if sha(output / 'ledger/events.jsonl') != complete['ledger_events_sha256']:
        raise ValueError('Attempt events hash mismatch')
    bank = CudaCallBank(dict(marker['config'], run_fingerprint=marker['run_fingerprint']), output, readonly=True)
    raw = bank.raw
    paths = read(output / 'CURRENT/initial_paths.jsonl')
    for rnd in range(1, complete['final_round'] + 1):
        folder = output / f'CURRENT/r{rnd}'
        stage = read(folder / 'complete.json')
        committed = read(folder / 'deliveries_committed.json')
        if stage['paths_sha256'] != sha(folder / 'paths.jsonl'):
            raise ValueError('Stage paths hash mismatch')
        if committed['sha256'] != sha(folder / 'deliveries.jsonl') or not committed['before_feedback']:
            raise ValueError('Stage delivery commit mismatch')
        next_paths = read(folder / 'paths.jsonl')
        delivered = read(folder / 'deliveries.jsonl')
        if len(delivered) != len(next_paths) or any(
                dict(a, accepted=b['accepted'], feedback=b['feedback']) != b
                for a, b in zip(delivered, next_paths)):
            raise ValueError('Feedback changed a committed delivery')
        engine.validate_transition(paths, next_paths, bank)
        paths = next_paths
    if paths != read(output / 'final_paths.jsonl'):
        raise ValueError('Final path is not the committed last stage')
    for row in paths:
        if r.account(dict(row), bank) != row:
            raise ValueError('Path cost mismatch')
    summary = read(output / 'summary.json')
    if (summary['final_correct'] != sum(row['accepted'] for row in paths) or
            summary['initial_correct'] != sum(row['initial_accepted'] for row in paths) or
            summary['path_tokens'] != sum(row['total_tokens'] for row in paths)):
        raise ValueError('Summary differs from saved paths')
    events = read(output / 'ledger/events.jsonl')
    completed_attempts = {row['attempt_id'] for row in raw.values()}
    started = {row['attempt_id'] for row in events if row['event'] == 'attempt_started'}
    failed = {row['attempt_id']: row for row in events
              if row['event'] == 'attempt_failed' and row['attempt_id'] not in completed_attempts}
    return dict(passed=True, scorable_n=len(paths), completed_calls=len(raw),
                completed_call_tokens=sum(x['prompt_tokens'] + x['generation_tokens'] for x in raw.values()),
                failed_attempts=len(failed),
                attempts_without_completion_event=len(started - completed_attempts - set(failed)),
                failed_generation_tokens_observed=sum(row['generation_tokens_observed'] for row in failed.values()),
                scope='Saved token IDs, hashes, transitions and stopped successes; no new inference or SQL rescoring',
                failed_attempt_usage='See ledger/events.jsonl; incomplete observed usage is a lower bound')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Freeze subset and execute only its reference SQL; no model calls')
    prep.add_argument('--config', type=Path, required=True)
    prep.add_argument('--dev-json', type=Path, required=True)
    prep.add_argument('--database-dir', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--sample-size', type=int, default=10)
    for name in ('run', 'verify', 'audit'):
        child = sub.add_parser(name)
        child.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.config, args.dev_json, args.database_dir, args.output, args.sample_size)
        result = dict(prepared=True, output=str(args.output.resolve()), n=result['sample_size'],
                      unscorable=result['unscorable'], run_fingerprint=result['run_fingerprint'])
    elif args.command == 'run':
        result = execute_run(args.output)
    elif args.command == 'verify':
        result = dict(verified=True, run_fingerprint=verify(args.output)['run_fingerprint'])
    else:
        result = audit(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
