"""Aggregate completed variants and expose the actual demand/probe overhead."""
from collections import Counter
from pathlib import Path
import argparse
import csv
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iteration13 import runner as r, analyze as old, cohort

HERE = Path(__file__).resolve().parent


def main(population):
    base = HERE/'results'/population
    tasks = {x['id']: x for x in r.read((r.prior.OUT if population == 'development' else cohort.DATA)/'public.jsonl')}
    full_path = (r.prior.OUT/'full_contexts.json' if population == 'development' else
                 r.HERE/'results/confirmation400/full_contexts.json')
    full = r.read(full_path)
    bank = r.CallBank(r.history_batches() + [p.parent for root in (r.HERE, HERE)
                     for p in sorted((root/'results').glob('**/batch/complete.json'))]
                     + [p.parent for p in sorted(Path(__file__).resolve().parent.glob('**/batch/complete.json'))])
    rows, comparisons, mechanisms, audits = [], [], [], {}
    for marker in sorted(base.glob('*/complete.json')):
        dest = marker.parent
        audits[dest.name] = old.audit(dest, bank)
        seed = r.read(dest/'initial_paths.jsonl')
        seed_stats = {}
        for ds in ['ALL'] + sorted({tasks[x['task_id']]['dataset'] for x in seed}):
            subset = [x for x in seed if ds == 'ALL' or tasks[x['task_id']]['dataset'] == ds]
            seed_stats[ds] = (sum(x['accepted'] for x in subset),
                              sum(x['total_tokens'] for x in subset)/len(subset))
        rows += [dict(variant=dest.name, total_repair_rounds=x['round']+1,
                      seed_correct=seed_stats[x['dataset']][0],
                      seed_mean_tokens=seed_stats[x['dataset']][1], **x)
                 for x in r.read(dest/'summary.json')]
        for file in sorted(dest.glob('r*/paths.jsonl')):
            paths = {x['task_id']: x for x in r.read(file)}
            comparator = base/'critic'/file.parent.name/'paths.jsonl'
            if dest.name != 'critic' and comparator.exists():
                other = {x['task_id']: x for x in r.read(comparator)}
                comparisons.append(dict(variant=dest.name, round=next(iter(paths.values()))['round'],
                                        **old.paired(paths, other, tasks)))
            specs = [x for p in file.parent.glob('**/logical_requests.jsonl') for x in r.read(p)]
            plans = [x['metadata']['plan'] for x in specs if 'plan' in x['metadata']]
            contexts = [x for x in specs if 'selected_columns' in x['metadata']]
            action_path = file.parent/'actions.jsonl'
            actions = r.read(action_path) if action_path.exists() else []
            def fields(selection):
                return {(t.casefold(), c.casefold()) for t, cols in selection.items() for c in cols}
            narrowed = sum(fields(x['metadata']['selected_columns']) <
                           fields(full[x['task_id']]['selected_columns']) for x in contexts)
            mechanisms.append(dict(variant=dest.name, round=next(iter(paths.values()))['round'],
                repair_requests=len(r.read(file.parent/'states.jsonl')), plans=len(plans),
                invalid_plans=sum(not x['valid'] for x in plans), modes=dict(Counter(x['mode'] for x in plans)),
                actions=dict(Counter(x['route'] for x in actions)),
                operations=dict(Counter(x.get('operation') for x in actions if 'operation' in x)),
                malformed_actions=sum(x['malformed'] for x in actions),
                repeated_candidates=sum(x.get('repeat_fallback', False) for x in actions),
                materialized_contexts=len(contexts), narrowed_contexts=narrowed,
                ambiguous_bindings=sum(b['ambiguous'] for x in plans for b in x.get('bindings', [])),
                fallback_reasons=dict(Counter(x['metadata'].get('demand_fallback') for x in contexts)),
                requested_fields=sum(len(x['inspect']) for x in plans),
                logical_probe_queries=sum(len(p['sql']) for x in specs for p in x['metadata'].get('probe_records', []))))
    out = base/'analysis';out.mkdir(exist_ok=True)
    for name, value in [('audits', audits), ('paired', comparisons), ('mechanisms', mechanisms)]:
        r.save(out/(name+'.json'), value, immutable=False)
    with (out/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines = ['# 后续修复的已完成比较', '',
             '正确数和token均包含共同初答及ReLink第一轮；总修复轮数包含ReLink。只列完整完成的配置。', '',
             '| 配置 | 总修复轮数 | 正确数 | 每题全过程token | 比ReLink第一轮新增正确 |',
             '|---|---:|---:|---:|---:|']
    for x in rows:
        if x['dataset'] == 'ALL':
            lines.append(f"| {x['variant']} | {x['total_repair_rounds']} | {x['correct']}/{x['n']} | "
                         f"{x['mean_all_tokens']:.2f} | {x['correct']-x['seed_correct']} |")
    (out/'结果汇总.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


def compare(population):
    root = Path(__file__).resolve().parent
    tasks = {x['id']: x for x in r.read((r.prior.OUT if population == 'development' else cohort.DATA)/'public.jsonl')}
    families = [('第一版', root), ('绑定修正', root/'bound_v2'), ('按需调用', root/'adaptive_v3'),
                ('机制对照', root/'evidence_controls'), ('局部修改', root/'patch_v4'),
                ('可读作用域', root/'closed_v5'), ('分支调度', root/'branches_v6'),
                ('修复起点对照', root/'linear_control'), ('代价计划', root/'scheduling_v7')]
    records, comparisons, seed = [], [], None
    baseline = None
    for family, folder in families:
        base = folder/'results'/population
        for marker in sorted(base.glob('*/complete.json')):
            dest = marker.parent
            if dest.name == 'critic' and baseline is not None:
                continue
            if dest.name == 'critic':
                baseline = dest
            if seed is None:
                seed = r.read(dest/'initial_paths.jsonl')
            for row in r.read(dest/'summary.json'):
                if row['dataset'] == 'ALL':
                    records.append(dict(family=family, variant=dest.name, extra_rounds=row['round'],
                                        correct=row['correct'], n=row['n'], mean_tokens=row['mean_all_tokens'],
                                        result=str(dest/f"r{row['round']}/paths.jsonl")))
    assert seed and baseline, 'Need a completed common comparator'
    available_rounds = sorted(int(p.parent.name[1:]) for p in baseline.glob('r*/paths.jsonl'))
    for row in records:
        if row['variant'] == 'critic':
            continue
        a = {x['task_id']: x for x in r.read(Path(row['result']))}
        baseline_round = min(row['extra_rounds'], max(available_rounds))
        b = {x['task_id']: x for x in r.read(baseline/f"r{baseline_round}/paths.jsonl")}
        comparisons.append(dict(family=row['family'], variant=row['variant'], extra_rounds=row['extra_rounds'],
                                comparator_extra_rounds=baseline_round,
                                **old.paired(a, b, tasks)))
    initial = sum(x['initial_accepted'] for x in seed)
    initial_tokens = sum(x['total_tokens']-x['repair_tokens'] for x in seed)/len(seed)
    seed_correct = sum(x['accepted'] for x in seed)
    seed_tokens = sum(x['total_tokens'] for x in seed)/len(seed)
    out = root/'results'/population/'comparison';out.mkdir(exist_ok=True)
    r.save(out/'paired.json', comparisons, immutable=False)
    r.save(out/'summary.json', records, immutable=False)
    lines = ['# 后续纠错结构的开发比较', '',
             f'共{len(seed)}题。共同初答{initial}题正确、每题{initial_tokens:.2f} token；'
             f'ReLink修复一轮后{seed_correct}题正确、每题{seed_tokens:.2f} token。', '',
             '下表正确数包含初答，token包含初答、ReLink以及实际发生的全部后续调用。'
             '额外轮数从ReLink第一轮之后计起，只对用户拒绝的候选继续修复。只列完整完成的配置。', '',
             '| 版本 | 后续方法 | 额外轮数 | 最终正确 | 每题全过程token |',
             '|---|---|---:|---:|---:|']
    for x in records:
        lines.append(f"| {x['family']} | {x['variant']} | {x['extra_rounds']} | {x['correct']}/{x['n']} | {x['mean_tokens']:.2f} |")
    lines += ['', '这些问题已经用于开发，结果用于选择候选结构，不构成独立确认。'
              '第一版demand的需求格式有误并回退完整结构，因此与demand_full的相同结果不能作为结构筛选有效的证据。'
              '绑定修正与按需调用另保留源码、输入哈希、逐题交付、token账本及核查结果。']
    lines += ['', '分支调度最多在ReLink之后交付三次SQL，CRITIC对照最多两次；两者最多各调用四次语言模型。'
              '分支调度在收到拒绝后才继续，进入路径的外部缓存响应全额计token。'
              '只有本路径此前已经调用并被拒绝的相同直接生成请求才主动跳过，不再次调用。']
    (out/'全部候选比较.md').write_text('\n'.join(lines)+'\n')
    mechanism_comparisons(root, population, tasks, out)
    print('\n'.join(lines), flush=True)


def mechanism_comparisons(root, population, tasks, out):
    """Compare executed controls, rather than attributing every gain to routing."""
    selected = root/'branches_v6/results'/population/'direct_first'
    if not (selected/'complete.json').exists():
        return
    candidate = {x['task_id']: x for x in r.read(selected/'r3/paths.jsonl')}
    controls = [
        ('CRITIC两轮', root/'results'/population/'critic', 2),
        ('不跳过重复请求', root/'branches_v6/results'/population/'no_memo', 3),
        ('先CRITIC再直接修改', root/'branches_v6/results'/population/'critic_first', 3),
        ('交错次序', root/'scheduling_v7/results'/population/'interleave', 3),
        ('继续最新SQL', root/'linear_control/results'/population/'latest', 3),
        ('重新推理复核', root/'scheduling_v7/results'/population/'calibrated_fresh', 3),
    ]
    comparisons = []
    for name, dest, rnd in controls:
        if (dest/'complete.json').exists():
            other = {x['task_id']: x for x in r.read(dest/f'r{rnd}/paths.jsonl')}
            comparisons.append(dict(control=name, candidate=str(selected), comparator=str(dest),
                candidate_extra_rounds=3, comparator_extra_rounds=rnd,
                **old.paired(candidate, other, tasks)))
    r.save(out/'mechanism_comparisons.json', comparisons, immutable=False)

    stages = []
    for folder in sorted(selected.glob('r*/paths.jsonl')):
        before = r.read(selected/'initial_paths.jsonl' if folder.parent.name=='r1' else
                        selected/f"r{int(folder.parent.name[1:])-1}/paths.jsonl")
        after = r.read(folder)
        stages.append(dict(round=int(folder.parent.name[1:]),
            deliveries=len(r.read(folder.parent/'states.jsonl')),
            skipped=len(r.read(folder.parent/'skipped.jsonl')),
            model_calls=sum(len(r.read(p)) for p in folder.parent.glob('**/logical_requests.jsonl')),
            newly_accepted=sum(x['accepted'] for x in after)-sum(x['accepted'] for x in before),
            total_added_tokens=sum(x['total_tokens'] for x in after)-sum(x['total_tokens'] for x in before)))
    r.save(out/'selected_stage_costs.json', stages, immutable=False)


def inventory():
    """Audit unique completed raw batches, keeping replay bills separate."""
    root = Path(__file__).resolve().parent
    batches = sorted({p.parent.resolve() for p in root.glob('**/batch/complete.json')})
    rows = []
    for folder in batches:
        actual = list(r.prior.completed(folder).values())
        for value in actual:
            assert value['prompt_tokens']==len(value['prompt_token_ids'])
            assert value['generation_tokens']==len(value['generated_token_ids'])
        rows.append(dict(batch=str(folder), calls=len(actual),
            input_tokens=sum(x['prompt_tokens'] for x in actual),
            output_tokens=sum(x['generation_tokens'] for x in actual),
            truncated_request_ids=[x['id'] for x in actual if x['finish_reason']=='length']))
    r.save(root/'results/raw_inventory.json',dict(batches=rows,
        actual_calls=sum(x['calls'] for x in rows),
        input_tokens=sum(x['input_tokens'] for x in rows),
        output_tokens=sum(x['output_tokens'] for x in rows),
        token_lengths_verified=True,
        note='Actual research expenditure across all variants; not the per-path logical cost used in comparisons.'),
        immutable=False)
    bank = r.CallBank(r.history_batches()+[p.parent for p in r.HERE.glob('**/batch/complete.json')]+batches)
    checked = []
    for population in ('development', 'exploration400'):
        for name, dest in [('selected', root/'branches_v6/results'/population/'direct_first'),
                           ('fresh', root/'scheduling_v7/results'/population/'calibrated_fresh')]:
            if not (dest/'complete.json').exists():
                continue
            paths = r.read(dest/'r3/paths.jsonl')
            ids = [rid for row in paths for rid in row['call_ids']]
            checked.append(dict(population=population, variant=name, paths=len(paths),
                logical_calls=len(ids), truncated_calls=[rid for rid in ids if bank.raw[rid]['finish_reason']=='length'],
                **old.audit(dest, bank)))
    r.save(root/'results/selected_path_audits.json', checked, immutable=False)
    reproducibility = []
    for population in ('development', 'exploration400'):
        replay = root/'branches_v6/results'/population/'direct_first'
        fresh = root/'scheduling_v7/results'/population/'calibrated_fresh'
        if not (fresh/'complete.json').exists():
            continue
        for file in sorted(fresh.glob('r*/*/records.jsonl')):
            relative = file.relative_to(fresh)
            oldfile = replay/relative
            if not oldfile.exists():
                continue
            previous = {x['task_id']: x for x in r.read(oldfile)}
            current = {x['task_id']: x for x in r.read(file)}
            changes = []
            for tid in sorted(previous.keys() & current.keys()):
                a=bank.raw[previous[tid]['request_id']];z=bank.raw[current[tid]['request_id']]
                changes.append(dict(task_id=tid,
                    identical_messages=a['messages_sha256']==z['messages_sha256'],
                    identical_rendered_prompt=a['rendered_prompt_sha256']==z['rendered_prompt_sha256'],
                    identical_model=a['model']==z['model'] and a['revision']==z['revision'],
                    identical_backend=a['backend_code_sha256']==z['backend_code_sha256'],
                    identical_output=a['raw_text']==z['raw_text']))
            reproducibility.append(dict(population=population,stage=str(relative.parent),
                matched_calls=len(changes), only_fresh=sorted(current.keys()-previous.keys()),
                only_replay=sorted(previous.keys()-current.keys()),
                changed_outputs=sum(not x['identical_output'] for x in changes),
                changed_inputs=sum(not x['identical_messages'] for x in changes),
                cases=changes))
    r.save(root/'results/fresh_replay_check.json', reproducibility, immutable=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('population');parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    (compare if args.compare else main)(args.population)
    inventory()
