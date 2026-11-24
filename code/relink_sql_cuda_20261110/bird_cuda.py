"""Prepare and run an isolated BIRD dev CUDA experiment using the pinned engine."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(os.environ.get('RELINK_ROOT', 'relink_sql_cuda_20261018')).resolve()
CODE = ROOT / 'ReLink-SQL_CUDA_20261019'
sys.path.insert(0, str(CODE))
from cuda_repro import runner
from cuda_repro.backend import validate_config
from cuda_repro.common import digest, read, save, sha

QUESTION_SHA = 'ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb'
DATABASES = {
    'california_schools': 'c0903eec662e63068fd1d14403d3d6c1d473287fc10c4356333ea58f878db983',
    'card_games': 'c98bdb57fe7474da798b407785544b9af0daaad5d61fd21e2a73309493bc1227',
    'codebase_community': '92101be6d2a9f6adceea59d38f6d1c556087f9eea1432432a21268cb1348036d',
    'debit_card_specializing': 'b3d149ad05746dbbe5116e229e17e18f09c39db43cf117d9ef3441753608b691',
    'european_football_2': 'f72f5c9b990371417288a67b27a02e601ef904d45525c2c9f7931fb01001c536',
    'financial': 'd15d89cdb068a202b6f2b99342af44dffc1d52545b39ceaf62efdc0ba570101e',
    'formula_1': '79770caf966707e35516fa566e24b40ae515c74ec1ec4631235245645b87b24d',
    'student_club': 'eb89bcfe97eefa386a27904ec5aa15159811a7eac894ec659a36e48fa9f76b77',
    'superhero': '75e94a2c3236ee3bb2c01fb97a1c4b4c1c269bcefd4eab1d04be323d2d0825b1',
    'thrombosis_prediction': '87583183c3dc472fba04de702965560a9d7c0a548836613f242317e0eeb83f00',
    'toxicology': '35ef27ae6bdfda530e125ed369666ac0abb8bc8c0bcc0ad09407f547ecb61a93',
}
SEED = 'relink-cuda-bird-subset-v1'


def prepare(config_path, dev_path, database_dir, output, size, expected_question=QUESTION_SHA,
            expected_databases=None, expected_rows=1534):
    expected_databases = DATABASES if expected_databases is None else expected_databases
    config_path, dev_path, database_dir, output = map(
        lambda p: Path(p).resolve(), (config_path, dev_path, database_dir, output))
    if output.exists():
        raise ValueError('Output already exists; verify or choose another directory: ' + str(output))
    staging = output.with_name(output.name + '.staging')
    if staging.exists():
        raise ValueError('Staging exists; preserve for inspection: ' + str(staging))
    runner.verify_upstream()
    if sha(dev_path) != expected_question:
        raise ValueError('Question file hash mismatch')
    rows = read(dev_path)
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise ValueError('Unexpected number of questions')
    if any(not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in
           ('db_id', 'question', 'SQL')) or not isinstance(row.get('evidence', ''), str)
           for row in rows):
        raise ValueError('Invalid BIRD row')
    if {row['db_id'] for row in rows} != set(expected_databases):
        raise ValueError('Unexpected database identities')
    if not 1 <= size <= len(rows):
        raise ValueError('Invalid sample size')
    indices = (list(range(len(rows))) if size == len(rows) else
               sorted(range(len(rows)), key=lambda i: digest([SEED, i, rows[i]['db_id'],
                                                              rows[i]['question']]))[:size])
    config = read(config_path)
    if 'run_fingerprint' in config or config.get('device') != 0 or config.get('batch_size') != 1:
        raise ValueError('Expected fixed GPU 0, batch 1 configuration without run fingerprint')
    inventory = validate_config(config)
    schemas, external = {}, {str(config_path): sha(config_path), str(dev_path): sha(dev_path),
                              str(Path(__file__).resolve()): sha(__file__)}
    for db_id, expected in expected_databases.items():
        if Path(db_id).name != db_id or db_id in ('.', '..'):
            raise ValueError('Invalid database name')
        db = (database_dir / db_id / (db_id + '.sqlite')).resolve()
        if sha(db) != expected:
            raise ValueError('Database hash mismatch: ' + db_id)
        external[str(db)] = expected
        with runner.connect(db) as con:
            if con.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise ValueError('SQLite quick_check failed: ' + db_id)
            ddl = [x[0] for x in con.execute("SELECT sql FROM sqlite_master WHERE type IN ('table','view') "
                   "AND name NOT LIKE 'sqlite_%' ORDER BY name") if x[0]]
        schemas[db_id] = (str(db), '\n\n'.join(x.rstrip(';') + ';' for x in ddl))
        print('DATABASE VERIFIED', db_id, flush=True)
    tasks, selection, contexts, sqls = [], [], {}, {}
    for index in indices:
        row = rows[index]
        db_id = row['db_id']
        tid = f'bird_cuda_dev:{index}:{digest(row["question"])[:12]}'
        db, schema = schemas[db_id]
        task = dict(id=tid, dataset='bird', group=db_id, split='full_dev', kind='sql',
                    question=row['question'], evidence=row.get('evidence', ''), db_path=db, schema=schema)
        if set(task) != runner.r.methods.TASK_FIELDS:
            raise ValueError('Unexpected task fields')
        tasks.append(task)
        contexts[tid] = runner.r.contexts.old.full(task)
        selection.append(dict(id=tid, source_row=index, db_id=db_id,
                              question_sha256=digest(row['question'])))
        sqls[tid] = row['SQL']
    def reference(task):
        sql = sqls[task['id']]
        result = runner.r.stable(runner.r.execute(task['db_path'], sql, timeout=10, max_rows=50000))
        return task['id'], dict(gold_sql=sql, gold_result=result,
                                order_matters='order by' in sql.lower())
    references = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n, (tid, ref) in enumerate(pool.map(reference, tasks), 1):
            references[tid] = ref
            if n % 100 == 0 or n == len(tasks):
                print('REFERENCE EXECUTION', n, '/', len(tasks), flush=True)
    unscorable = {tid: ref['gold_result'].get('error', 'unknown') for tid, ref in references.items()
                  if ref['gold_result']['status'] != 'ok'}
    # mkdir is exclusive; preserve any incomplete directory on failure.
    staging.mkdir(parents=True, mode=0o700)
    save(staging / 'public.jsonl', tasks)
    save(staging / 'selection.json', dict(seed=SEED, no_outcome_selection=True,
         no_replacement=True, total_dev_rows=len(rows), rows=selection))
    save(staging / 'full_contexts.json', contexts)
    save(staging / 'private/references.json', references)
    save(staging / 'private/reference_audit.json', dict(n=size, unscorable=unscorable,
                                                        no_replacement=True))
    names = ('public.jsonl', 'selection.json', 'full_contexts.json',
             'private/references.json', 'private/reference_audit.json')
    marker = dict(protocol='isolated-cuda-bird-dev-v1', config=config,
                  config_path=str(config_path), dev_path=str(dev_path), database_dir=str(database_dir),
                  sample_size=size, selected_ids=[task['id'] for task in tasks], unscorable=unscorable,
                  missing_historical_inputs=list(runner.MISSING_PROTOCOLS), source_files=runner.source_files(),
                  environment=runner.environment(), model_inventory=inventory, external_files=external,
                  artifacts={name: sha(staging / name) for name in names})
    marker['run_fingerprint'] = digest(marker)
    save(staging / 'prepared.json', marker)
    runner.verify(staging)
    os.rename(staging, output)
    return marker


def run(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        raise ValueError('CUDA_VISIBLE_DEVICES must be unset')
    output = Path(output).resolve()
    marker = runner.verify(output)
    if marker['protocol'] != 'isolated-cuda-bird-dev-v1':
        raise ValueError('Not a BIRD run')
    lock_path = ROOT / 'runs/ablations/gpu0_ablation.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.is_symlink():
        raise ValueError('GPU lock is a symlink')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=uuid',
                                       '--format=csv,noheader'], text=True).strip()
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader'], text=True)
        if any(line.split(',')[0].strip() == gpu for line in apps.splitlines()):
            raise ValueError('GPU 0 is occupied')
        result = runner.execute_run(output)
        return dict(summary=read(output / 'summary.json'), audit=runner.audit(output), result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for option in ('config', 'dev-json', 'database-dir', 'output'):
        prep.add_argument('--' + option, type=Path, required=True)
    prep.add_argument('--sample-size', type=int, required=True)
    for name in ('run', 'verify', 'audit'):
        sub.add_parser(name).add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        marker = prepare(args.config, args.dev_json, args.database_dir, args.output, args.sample_size)
        result = dict(prepared=True, n=marker['sample_size'], unscorable=marker['unscorable'],
                      fingerprint=marker['run_fingerprint'])
    elif args.command == 'run':
        result = run(args.output)
    elif args.command == 'verify':
        marker = runner.verify(args.output)
        result = dict(verified=True, fingerprint=marker['run_fingerprint'],
                      n=marker['sample_size'], unscorable=marker['unscorable'])
    else:
        result = runner.audit(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
