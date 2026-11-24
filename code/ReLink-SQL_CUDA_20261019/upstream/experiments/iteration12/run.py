"""Two feedback-gated repair rounds, real inference, isolated binary acceptance."""
from pathlib import Path
import argparse
import concurrent.futures
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iteration4 import batch_public_generate as batch
from iteration4.collect_confirmation import completed
from iteration10 import share_stop_batch
from iteration11 import contexts
from iteration12 import methods
from iteration12.feedback import UserSimulator
from recoverylab.sql import execute

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent
OUT = HERE / 'results/pilot128'
OLD = EXP / 'iteration11/results/pilot128_v1'
REFERENCE = EXP / 'iteration8/data/main1000/private/references.json'
MAX_ROUNDS = 2


def read(path):
    path = Path(path)
    return ([json.loads(s) for s in path.read_text().splitlines() if s.strip()]
            if path.suffix == '.jsonl' else json.loads(path.read_text()))


def save(path, value, immutable=True):
    path = Path(path)
    text = (''.join(json.dumps(r, ensure_ascii=False, default=str) + '\n' for r in value)
            if path.suffix == '.jsonl' else json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
    if path.exists() and immutable:
        if path.read_text() != text:
            raise ValueError('Refusing changed committed artifact: ' + str(path))
        return
    batch.atomic_write(path, text)


def implementation_files():
    files = [HERE / n for n in ('PROTOCOL.md', 'run.py', 'methods.py', 'feedback.py')]
    files += [EXP / n for n in ('iteration4/batch_public_generate.py', 'iteration4/collect_confirmation.py',
              'iteration4/collect_paper_baselines.py', 'iteration2/collect.py', 'iteration2/share_adapter.py',
              'iteration3/repair_index.py', 'iteration3/column_context.py', 'iteration8/input_selectors.py',
              'iteration10/methods.py', 'iteration10/share_stop_batch.py', 'iteration10/sources/rsl_Instruction.py',
              'iteration11/contexts.py', 'recoverylab/sql.py', 'vendor/share/src/prompts/for_infer.py')]
    files += sorted((EXP / 'iteration4/prompts').glob('*.txt'))
    files += sorted((EXP / 'iteration8/sources').glob('template_select_*.txt'))
    return files


def prepare():
    if (OUT / 'prepared.json').exists():
        verify()
        return
    tasks = read(OLD / 'public.jsonl')
    mapping = [r for r in read(OLD / 'root_mapping.jsonl') if r['root'] == '7_F']
    raw = completed(OLD / 'roots/batch')
    observed = read(OLD / 'root_observations.json')
    requests = {r['id']: r for r in read(OLD / 'roots/requests.jsonl')}
    assert len(tasks) == len(mapping) == 128
    assert all(set(t) == methods.TASK_FIELDS for t in tasks)
    assert {r['task_id'] for r in mapping} == {t['id'] for t in tasks}
    initial = []
    for r in mapping:
        tid, rid = r['task_id'], r['request_id']
        value, req = raw[rid], requests[rid]
        assert (req['model'], req['revision']) == tuple(methods.model('RETRY7'))
        assert value['messages_sha256'] == batch.digest(req['messages'])
        initial.append(dict(task_id=tid, request_id=rid, sql=observed[rid]['raw_text'],
                            observation=observed[rid]['observation'],
                            prompt_tokens=value['prompt_tokens'], generation_tokens=value['generation_tokens']))
    full = {t['id']: contexts.old.full(t) for t in tasks}
    tables = {t['dataset'] + ':' + t['group']: methods.share.table_info_from_sqlite(dict(t, db_id=t['group']))
              for t in tasks}
    save(OUT / 'public.jsonl', tasks)
    save(OUT / 'initial.jsonl', initial)
    save(OUT / 'full_contexts.json', full)
    save(OUT / 'share_tables.json', tables)
    source_files = [OLD / n for n in ('public.jsonl', 'root_mapping.jsonl', 'root_observations.json',
                    'roots/requests.jsonl', 'roots/batch/complete.json', 'roots/batch/results.jsonl')]
    protected = read(EXP / 'iteration5/provisional_v1/manifest.json')
    save(OUT / 'prepared.json', dict(n=128, initial='7_F', max_repair_rounds=MAX_ROUNDS,
         methods=methods.METHODS, feedback='Simulated accurate binary user after each delivered SQL',
         files={str(p): batch.sha(p) for p in implementation_files() + source_files},
         inputs={n: batch.sha(OUT / n) for n in ('public.jsonl', 'initial.jsonl', 'full_contexts.json', 'share_tables.json')},
         reference_sha256=read(EXP / 'iteration8/results/main1000/score_complete.json')['reference_sha256'],
         old_model_sha256=protected['model_sha256'], preparation_read_references=False))
    print('PREPARED', 128, 'tasks; ten baselines; at most two feedback-gated repair rounds', flush=True)


def verify():
    marker = read(OUT / 'prepared.json')
    for p, value in marker['files'].items():
        assert batch.sha(p) == value, p
    for name, value in marker['inputs'].items():
        assert batch.sha(OUT / name) == value, name
    assert batch.sha(EXP / 'iteration5/provisional_v1/estimators.joblib') == marker['old_model_sha256']
    return marker


def initial_acceptance(simulator):
    if (OUT / 'initial_paths.jsonl').exists():
        return
    rows = read(OUT / 'initial.jsonl')
    feedback = [dict(task_id=r['task_id'], **simulator.assess(r['task_id'], r['observation'])) for r in rows]
    save(OUT / 'initial_feedback.jsonl', feedback)
    signals = {r['task_id']: r for r in feedback}
    paths = []
    for method in methods.METHODS:
        for r in rows:
            accepted = signals[r['task_id']]['accepted']
            paths.append(dict(task_id=r['task_id'], method=method, round=0, sql=r['sql'],
                observation=r['observation'], call_ids=[r['request_id']], critiques=[], rejected_sqls=[],
                accepted=accepted, initial_accepted=accepted, feedback=signals[r['task_id']]['feedback']))
    save(OUT / 'initial_paths.jsonl', paths)
    print('INITIAL ACCEPTANCE', sum(r['accepted'] for r in feedback), '/', len(rows), flush=True)


def previous_paths(rnd):
    return read(OUT / ('initial_paths.jsonl' if rnd == 1 else f'r{rnd-1}/paths.jsonl'))


def pending_state(previous, rnd):
    if previous['accepted']:
        return None
    return dict(task_id=previous['task_id'], method=previous['method'], round=rnd,
                sql=previous['sql'], observation=previous['observation'],
                rejected_sqls=previous['rejected_sqls'] + [previous['sql']],
                feedback=previous['feedback'], critiques=previous['critiques'])


def prepare_round(rnd):
    states = [s for r in previous_paths(rnd) if (s := pending_state(r, rnd)) is not None]
    save(OUT / f'r{rnd}/states.jsonl', states)
    print('ROUND', rnd, 'pending:', {m: sum(s['method'] == m for s in states) for m in methods.METHODS}, flush=True)
    return states


def spec(state, stage, messages, metadata=None, role=None, attempt=0):
    method = state['method']
    if role and role != 'generator':
        model = (str(EXP / 'models' / ('share-' + role + '-4bit')), methods.share.CHECKPOINTS[role][1])
        cap = 3000
    else:
        model = methods.model(method)
        cap = 2048 if stage in ('feedback', 'selection') else 1024
    return dict(task_id=state['task_id'], method=method, round=state['round'], stage=stage,
                request=dict(id=f"{state['task_id']}|I12_{method}_R{state['round']}_{stage}_A{attempt}",
                             messages=messages, model=model[0], revision=model[1], max_tokens=cap),
                metadata={k: v for k, v in (metadata or {}).items() if k != 'schema_text'})


def execute_batch(directory, specs, stopped=False):
    """Only exact within-batch requests share an output; each path retains its bill."""
    unique, mapping, seen = [], [], {}
    for s in specs:
        request = s['request']
        key = batch.digest([s['task_id'], request['model'], request['revision'], request['messages'], request['max_tokens']])
        if key not in seen:
            seen[key] = request['id']
            unique.append(request)
        mapping.append(dict(task_id=s['task_id'], method=s['method'], round=s['round'], stage=s['stage'],
                            request_id=seen[key], metadata=s['metadata']))
    save(directory / 'requests.jsonl', unique)
    save(directory / 'mapping.jsonl', mapping)
    dest = directory / 'batch'
    if not (dest / 'complete.json').exists():
        if not unique:
            save(dest / 'results.jsonl', [])
            save(dest / 'complete.json', dict(items=0, results_sha256=batch.sha(dest / 'results.jsonl')))
        else:
            if not (dest / 'manifest.json').exists():
                print((share_stop_batch.prepare if stopped else batch.prepare)(directory / 'requests.jsonl', dest), flush=True)
            script = Path(share_stop_batch.__file__) if stopped else Path(batch.__file__)
            command = [sys.executable, str(script)] + ([] if stopped else ['run']) + ['--output', str(dest)]
            subprocess.run(command, check=True)
    raw = completed(dest)
    rows = [dict(m, text=raw[m['request_id']]['text'], raw_text=raw[m['request_id']]['raw_text'],
                 request_ids=[m['request_id']], finish_reason=raw[m['request_id']]['finish_reason']) for m in mapping]
    save(directory / 'records.jsonl', rows)
    return rows


def generic_round(rnd, states, tasks, full):
    work = [s for s in states if s['method'] != 'SHARE_META']
    specs = []
    for s in work:
        stage = methods.initial_stage(s['method'])
        messages, extra = methods.build(tasks[s['task_id']], s, full[s['task_id']], stage)
        specs.append(spec(s, stage, messages, extra))
    first = execute_batch(OUT / f'r{rnd}/generic1', specs)
    by = {(r['task_id'], r['method']): r for r in first}
    specs = []
    for s in work:
        if methods.initial_stage(s['method']) == 'answer':
            continue
        messages, extra = methods.build(tasks[s['task_id']], s, full[s['task_id']], 'answer',
                                       by[s['task_id'], s['method']]['text'])
        specs.append(spec(s, 'answer', messages, extra))
    second = execute_batch(OUT / f'r{rnd}/generic2', specs)
    return first + second


def share_round(rnd, states, tasks, tables):
    work = [s for s in states if s['method'] == 'SHARE_META']
    outputs = {s['task_id']: {} for s in work}
    all_records = []
    for stage in methods.share.STAGES:
        calls = {s['task_id']: [] for s in work}
        last, metadata = {}, {}
        assistant = stage != 'generator_sql'
        for attempt in range(11 if assistant else 1):
            pending = [s for s in work if not calls[s['task_id']] or len(last[s['task_id']]['text'].strip()) <= 1]
            if not pending:
                break
            specs = []
            for s in pending:
                t = tasks[s['task_id']]
                messages, extra = methods.share_messages(t, s, stage, tables[t['dataset'] + ':' + t['group']], outputs[t['id']])
                metadata[t['id']] = extra
                specs.append(spec(s, stage, messages, extra, methods.share.ROLE[stage], attempt))
            rows = execute_batch(OUT / f'r{rnd}/share/{stage}/attempt{attempt}', specs, stopped=assistant)
            for r in rows:
                calls[r['task_id']] += r['request_ids']
                last[r['task_id']] = r
        records = []
        for s in work:
            tid = s['task_id']
            value = dict(last[tid], request_ids=calls[tid])
            if assistant and len(value['text'].strip()) <= 1:
                value['text'] = '[INVALID]'
            records.append(value)
            outputs[tid][stage] = value['text']
        save(OUT / f'r{rnd}/share/{stage}/records.jsonl', records)
        all_records += records
    return all_records


def deliver(rnd, states, records, tasks):
    marker = OUT / f'r{rnd}/deliveries_committed.json'
    if marker.exists():
        saved = read(marker)
        assert batch.sha(OUT / f'r{rnd}/deliveries.jsonl') == saved['delivery_sha256']
        assert batch.sha(OUT / f'r{rnd}/states.jsonl') == saved['states_sha256']
        return read(OUT / f'r{rnd}/deliveries.jsonl')
    prior = {(r['task_id'], r['method']): r for r in previous_paths(rnd)}
    public = {(s['task_id'], s['method']): s for s in states}
    groups = {}
    for record in records:
        groups.setdefault((record['task_id'], record['method']), []).append(record)

    def one(item):
        key, previous = item
        base = {k: v for k, v in previous.items() if k not in ('accepted', 'feedback')}
        base.update(round=rnd, carried=previous['accepted'])
        if previous['accepted']:
            return base
        state, steps = public[key], groups[key]
        finals = [r for r in steps if r['stage'] in ('answer', 'generator_sql')]
        assert len(finals) == 1, key
        final = finals[0]
        sql = (methods.share.extract_final_sql(final['text']) if key[1] == 'SHARE_META'
               else methods.decode(final['raw_text'], key[1]))
        calls = [rid for r in steps for rid in r['request_ids']]
        critiques = previous['critiques'] + ['SQL:\n' + state['sql'] + '\nCritique:\n' + r['text']
                                            for r in steps if r['stage'] == 'feedback']
        base.update(sql=sql, observation=execute(tasks[key[0]]['db_path'], sql, timeout=10, max_rows=50000),
                    call_ids=previous['call_ids'] + calls, critiques=critiques,
                    rejected_sqls=state['rejected_sqls'])
        return base

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(one, prior.items()))
    save(OUT / f'r{rnd}/deliveries.jsonl', rows)
    save(OUT / f'r{rnd}/deliveries_committed.json', dict(
        delivery_sha256=batch.sha(OUT / f'r{rnd}/deliveries.jsonl'),
        states_sha256=batch.sha(OUT / f'r{rnd}/states.jsonl'),
        next_acceptance_not_yet_read=True, rows=len(rows)))
    return rows


def accept_round(rnd, deliveries, simulator):
    previous = {(r['task_id'], r['method']): r for r in previous_paths(rnd)}
    feedback, paths = [], []
    for row in deliveries:
        before = previous[row['task_id'], row['method']]
        signal = (dict(accepted=before['accepted'], feedback=before['feedback']) if row['carried']
                  else simulator.assess(row['task_id'], row['observation']))
        if row['carried']:
            assert signal['accepted'] and row['call_ids'] == before['call_ids']
        feedback.append(dict(task_id=row['task_id'], method=row['method'], round=rnd, **signal))
        paths.append(dict(row, **signal))
    save(OUT / f'r{rnd}/feedback.jsonl', feedback)
    save(OUT / f'r{rnd}/paths.jsonl', paths)
    save(OUT / f'r{rnd}/complete.json', dict(paths_sha256=batch.sha(OUT / f'r{rnd}/paths.jsonl'),
        feedback_sha256=batch.sha(OUT / f'r{rnd}/feedback.jsonl'),
        deliveries_committed_sha256=batch.sha(OUT / f'r{rnd}/deliveries_committed.json')))
    print('ROUND COMPLETE', rnd, {m: sum(r['accepted'] for r in paths if r['method'] == m) for m in methods.METHODS}, flush=True)


def call_pool():
    initial = read(OUT / 'initial.jsonl')
    raw = {r['request_id']: r for r in initial}
    new = []
    for marker in sorted(OUT.glob('r*/**/batch/complete.json')):
        values = list(completed(marker.parent).values())
        assert not (set(raw) & {r['id'] for r in values})
        raw.update({r['id']: r for r in values})
        new += values
    return raw, new


def summarize(paths, tasks, raw):
    summary = []
    for method in methods.METHODS:
        for ds in ('ALL', 'spider', 'bird'):
            rows = [r for r in paths if r['method'] == method and (ds == 'ALL' or tasks[r['task_id']]['dataset'] == ds)]
            if not rows:
                continue
            errors = [r for r in rows if not r['initial_accepted']]
            inp = out = extra_in = extra_out = calls = 0
            for r in rows:
                inp += sum(raw[i]['prompt_tokens'] for i in r['call_ids'])
                out += sum(raw[i]['generation_tokens'] for i in r['call_ids'])
                extra_in += sum(raw[i]['prompt_tokens'] for i in r['call_ids'][1:])
                extra_out += sum(raw[i]['generation_tokens'] for i in r['call_ids'][1:])
                calls += len(r['call_ids']) - 1
            correct = sum(r['accepted'] for r in rows)
            rescued = sum(r['accepted'] for r in errors)
            summary.append(dict(method=method, dataset=ds, n=len(rows), initial_errors=len(errors),
                initial_correct=correct-rescued, rescued=rescued, correct=correct,
                repair_accuracy=100*rescued/len(errors) if errors else None,
                mean_repair_tokens=(extra_in+extra_out)/len(errors) if errors else 0,
                mean_repair_input_tokens=extra_in/len(errors) if errors else 0,
                mean_repair_output_tokens=extra_out/len(errors) if errors else 0,
                mean_all_tokens=(inp+out)/len(rows), total_input_tokens=inp, total_output_tokens=out,
                repair_calls=calls, unnecessary_repair_calls=sum(len(r['call_ids'])-1 for r in rows if r['initial_accepted']),
                initial_wrong_exec_ok=sum(tasks[r['task_id']]['initial_exec_ok'] for r in errors)))
    return summary


def report():
    verify()
    tasks = {t['id']: t for t in read(OUT / 'public.jsonl')}
    initial = {r['task_id']: r for r in read(OUT / 'initial.jsonl')}
    for tid in tasks:
        tasks[tid]['initial_exec_ok'] = initial[tid]['observation']['status'] == 'ok'
    raw, new = call_pool()
    summaries = []
    for rnd in range(1, MAX_ROUNDS+1):
        if (OUT / f'r{rnd}/complete.json').exists():
            rows = summarize(read(OUT / f'r{rnd}/paths.jsonl'), tasks, raw)
            summaries += [dict(round=rnd, **r) for r in rows]
    assert all(r['unnecessary_repair_calls'] == 0 for r in summaries)
    save(OUT / 'summary.json', summaries, immutable=False)
    accounting = dict(new_actual_calls=len(new), new_input_tokens=sum(r['prompt_tokens'] for r in new),
                      new_output_tokens=sum(r['generation_tokens'] for r in new),
                      truncated_calls=sum(r['finish_reason'] == 'length' for r in new), reused_initial_calls=128)
    accounting['new_total_tokens'] = accounting['new_input_tokens'] + accounting['new_output_tokens']
    save(OUT / 'accounting.json', accounting, immutable=False)
    lines = ['# 用户指出错误后的主要纠错基线：128题试验', '',
             '共同初答为70亿参数模型、完整表结构；所有方法只在模拟用户拒绝后进入下一轮，接受后停止。用户反馈只含“结果不正确，请修正”，没有参考SQL或正确结果。反馈被假设准确、覆盖每次交付。', '',
             '使用此前研究的Spider/BIRD各64题，属于开发试验；预留400题未使用。当前比较固定纠错流程，未重新训练两阶段路由。', '']
    for rnd in sorted({r['round'] for r in summaries}):
        total = [r for r in summaries if r['round'] == rnd and r['dataset'] == 'ALL']
        first = total[0]
        lines += [f'## 最多{rnd}轮修复', '',
                  f"共同初答正确{first['initial_correct']}/128，错误{first['initial_errors']}题；其中{first['initial_wrong_exec_ok']}题的SQL能够执行但结果判错。", '',
                  '| 方法 | 修复成功/初答错误 | 修复成功率 | 每个错误请求新增token | 全128题最终正确 | 每题全过程token |',
                  '|---|---:|---:|---:|---:|---:|']
        for r in total:
            lines.append(f"| {r['method']} | {r['rescued']}/{r['initial_errors']} | {r['repair_accuracy']:.2f}% | {r['mean_repair_tokens']:.2f} | {r['correct']}/128 | {r['mean_all_tokens']:.2f} |")
        lines += ['', '| 方法 | Spider修复成功 | BIRD修复成功 |', '|---|---:|---:|']
        for method in methods.METHODS:
            ds = {r['dataset']: r for r in summaries if r['round']==rnd and r['method']==method}
            a, b = ds['spider'], ds['bird']
            lines.append(f"| {method} | {a['rescued']}/{a['initial_errors']} | {b['rescued']}/{b['initial_errors']} |")
        lines += ['']
    lines += ['## 实际采集与解释', '',
              f"新增真实调用{accounting['new_actual_calls']}次、输入{accounting['new_input_tokens']}、输出{accounting['new_output_tokens']}，合计{accounting['new_total_tokens']} token；输出达到上限{accounting['truncated_calls']}次。首次128次调用复用历史记录，每条部署路径仍计入首次成本。", '',
              '每个错误请求新增token仅在共同初答错误集合上取平均；全过程token在全部128题上取平均，两者分母不同。输入包括用户反馈和历史信息，输出包括全部批评、选择、助手和SQL。', '',
              '正确答案不再进入修复是本轮准确反馈机制的结果，不能把零误改当作模型自主识错能力。论文方法均按[协议](PROTOCOL.md)规定的组件或本地适配解释，不能称原论文完整系统成绩。', '',
              'RELINK14为已有结构输入方法在用户反馈下的适配，和EXEC14使用相同模型、初答、反馈及执行报告。旧的80/128两阶段候选属于另一种停止设定，不能直接混表。', '',
              '逐轮轨迹、完整输入/输出账本、分数据集结果保存在[结果目录](results/pilot128)。当前试验只有一种首次模型和二元反馈；具体反馈、其他首次模型、独立数据与更大规模仍待验证。', '']
    batch.atomic_write(HERE / '用户反馈纠错基线结果.md', '\n'.join(lines))
    print('REPORT', len(summaries), 'summary rows;', accounting, flush=True)


def run():
    prepare()
    marker = verify()
    assert batch.sha(REFERENCE) == marker['reference_sha256']
    simulator = UserSimulator(REFERENCE)
    initial_acceptance(simulator)
    tasks = {t['id']: t for t in read(OUT / 'public.jsonl')}
    full, tables = read(OUT / 'full_contexts.json'), read(OUT / 'share_tables.json')
    for rnd in range(1, MAX_ROUNDS+1):
        if (OUT / f'r{rnd}/complete.json').exists():
            continue
        verify()
        states = prepare_round(rnd)
        records = generic_round(rnd, states, tasks, full)
        records += share_round(rnd, states, tasks, tables)
        deliveries = deliver(rnd, states, records, tasks)
        accept_round(rnd, deliveries, simulator)
        report()
    verify()
    report()
    save(OUT / 'complete.json', dict(prepared_sha256=batch.sha(OUT / 'prepared.json'),
         report_sha256=batch.sha(HERE / '用户反馈纠错基线结果.md'),
         summary_sha256=batch.sha(OUT / 'summary.json'), accounting=read(OUT / 'accounting.json'),
         rounds=MAX_ROUNDS, methods=methods.METHODS, old_model_unchanged=True))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run', 'report'])
    globals()[parser.parse_args().command]()
