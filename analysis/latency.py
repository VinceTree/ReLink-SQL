"""Per-request calls, model time and delivered-SQL execution time from saved CUDA runs.

Model time = attempt_completed - attempt_started for the completed attempt of each call.
DB time = 'seconds' of every delivered (non-carried) SQL observation, r0..r5.
Candidate executions inside r4 are stored without timing and are not included.
"""
import json, sys, statistics
from pathlib import Path

ROOT = Path(sys.argv[1])
RUNS = {
    'BIRD ReLink-SQL': 'bird_dev_CURRENT_20261110',
    'BIRD full-schema retry': 'ablations/bird_full_schema_retry_20261112',
    'Spider ReLink-SQL': 'spider_dev_CURRENT_20261020',
    'Spider full-schema retry': 'ablations/spider_full_schema_retry_20261110',
}


def jl(p):
    with open(p) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def call_seconds(run):
    start, done = {}, {}
    for e in jl(run / 'ledger/events.jsonl'):
        if e['event'] == 'attempt_started':
            start[e['attempt_id']] = e['time_ns']
        elif e['event'] == 'attempt_completed':
            done[e['request_id']] = (e['time_ns'] - start[e['attempt_id']]) / 1e9
    return done


def main():
    out = {}
    for name, rel in RUNS.items():
        run = ROOT / rel
        secs = call_seconds(run)
        final = list(jl(run / 'final_paths.jsonl'))
        rounds = [list(jl(run / 'CURRENT/initial_paths.jsonl'))]
        rounds += [list(jl(run / f'CURRENT/r{i}/paths.jsonl')) for i in range(1, 6) if (run / f'CURRENT/r{i}/paths.jsonl').exists()]
        db = {}
        for rows in rounds:
            for r in rows:
                if r.get('carried'):
                    continue
                db[r['task_id']] = db.get(r['task_id'], 0.0) + float((r.get('observation') or {}).get('seconds') or 0.0)
        repaired = [r for r in final if not r['initial_accepted']]
        def per(rows):
            calls = [len(r['call_ids']) - 1 for r in rows]
            llm = [sum(secs.get(c, 0.0) for c in r['call_ids'][1:]) for r in rows]
            missing = sum(c not in secs for r in rows for c in r['call_ids'][1:])
            dbt = [db.get(r['task_id'], 0.0) for r in rows]
            return dict(n=len(rows), mean_repair_calls=statistics.mean(calls), mean_model_seconds=statistics.mean(llm),
                        median_model_seconds=statistics.median(llm), mean_db_seconds=statistics.mean(dbt),
                        mean_repair_tokens=statistics.mean(r['repair_tokens'] for r in rows), calls_without_timing=missing)
        out[name] = dict(all_requests=per(final), initially_rejected=per(repaired),
                         accepted=sum(r['accepted'] for r in final))
        print(name, json.dumps(out[name], indent=1))
    Path(sys.argv[2]).write_text(json.dumps(out, indent=2))


main()
