"""Paired estimates and independent ledger checks for committed experiments."""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from iteration13 import runner as r, cohort

LABELS = dict(FRESH14='完整结构重新生成',RELINK='ReLink',RETRY14='简单重试',EXEC14='执行反馈修复',
    CRITIC='CRITIC 本地适配',CHESS_C='CHESS 修复组件',SHARE='SHARE 原提示五阶段',
    NO_SCOPE='去作用域处理',NO_BRIDGE='去关联补全',LEX_MATCH='长度匹配词项筛选',
    FULL_REPORT='完整结构＋执行报告',VIEW_REPORT='ReLink 结构＋执行报告',
    FULL_SQL_REPORT='完整结构＋旧 SQL＋执行报告',VIEW_SQL_REPORT='ReLink 结构＋旧 SQL＋执行报告')


def paired(a,b,tasks):
    assert set(a)==set(b)
    groups = defaultdict(list)
    for tid in sorted(a):
        t=tasks[tid]
        groups[t['dataset'],t['group']].append([int(a[tid]['accepted'])-int(b[tid]['accepted']),
                                              a[tid]['total_tokens']-b[tid]['total_tokens']])
    rng=np.random.default_rng(20260915)
    samples=np.zeros((10000,2));point=np.zeros(2)
    for ds in sorted({k[0] for k in groups}):
        arrays=[np.array(values,dtype=float) for (dataset,_),values in sorted(groups.items()) if dataset==ds]
        totals=np.array([v.sum(axis=0) for v in arrays]);sizes=np.array([len(v) for v in arrays])
        weight=sizes.sum()/len(a)
        index=rng.integers(0,len(arrays),size=(10000,len(arrays)))
        samples+=weight*totals[index].sum(axis=1)/sizes[index].sum(axis=1)[:,None]
        point+=weight*totals.sum(axis=0)/sizes.sum()
    lo,hi=np.quantile(samples,[.025,.975],axis=0)
    return dict(n=len(a),database_groups=len(groups),accuracy_difference_pp=100*point[0],
        accuracy_ci95_pp=[100*lo[0],100*hi[0]],mean_token_difference=point[1],token_ci95=[lo[1],hi[1]],
        relink_only_correct=sum(a[t]['accepted'] and not b[t]['accepted'] for t in a),
        comparator_only_correct=sum(b[t]['accepted'] and not a[t]['accepted'] for t in a),
        inference='Paired; database-cluster bootstrap within datasets; original dataset weights; 10000 resamples')


def audit(dest,bank):
    r.verify(dest)
    marker=r.read(dest/'complete.json')
    assert r.prior.batch.sha(dest/'summary.json')==marker['summary_sha256']
    previous={(x['task_id'],x['method']):x for x in r.read(dest/'initial_paths.jsonl')}
    rounds=sorted(dest.glob('r*/paths.jsonl'))
    rows_checked=0;logical_checked=0
    for source in rounds:
        folder=source.parent
        assert r.prior.batch.sha(source)==r.read(folder/'complete.json')['paths_sha256']
        assert r.prior.batch.sha(folder/'deliveries.jsonl')==r.read(folder/'deliveries_committed.json')['sha256']
        pending={(s['task_id'],s['method']) for s in r.read(folder/'states.jsonl')}
        for key in pending:
            assert not previous[key]['accepted'] and not previous[key].get('stopped')
        after={}
        for row in r.read(source):
            key=row['task_id'],row['method'];before=previous[key]
            assert row['call_ids'][:len(before['call_ids'])]==before['call_ids']
            if before['accepted']:
                assert row['accepted'] and row['call_ids']==before['call_ids'] and row['sql']==before['sql']
            assert row['initial_accepted']==before['initial_accepted']
            if key in pending:
                assert row['rejected_sqls']==before['rejected_sqls']+[before['sql']]
            expected=r.account(dict(row),bank)
            for field in ('total_tokens','repair_tokens','input_tokens','output_tokens'):
                assert expected[field]==row[field],(key,field)
            after[key]=row;rows_checked+=1
        assert set(after)==set(previous)
        previous=after
    requests_by_source={}
    for path in sorted(dest.glob('**/logical_requests.jsonl')):
        records={(x['task_id'],x['method']):x for x in r.read(path.parent/'records.jsonl')}
        for spec in r.read(path):
            record=records[spec['task_id'],spec['method']]
            rid=record['request_id'];source=bank.sources[rid]
            if source not in requests_by_source:
                requests_by_source[source]={x['id']:x for x in r.read(Path(source)/'requests.jsonl')}
            actual=requests_by_source[source][rid]
            stopped=spec['stage'] in r.methods.share.STAGES and spec['stage']!='generator_sql'
            assert r.CallBank.key(spec['task_id'],spec['request'],stopped)==r.CallBank.key(spec['task_id'],actual,stopped)
            encoding=r.read(Path(source)/'manifest.json')['prompt_encoding']
            assert (encoding=='explicit_no_special_tokens')==stopped
            logical_checked+=1
    return dict(path_rows_checked=rows_checked,logical_requests_checked=logical_checked,passed=True)


def main(population):
    base=r.HERE/'results'/population
    assert (base/'complete.json').exists()
    confirmation=population=='confirmation400'
    tasks={t['id']:t for t in r.read((cohort.DATA if confirmation else r.prior.OUT)/'public.jsonl')}
    destinations=([base] if population=='family_mistral7' else
                  [p.parent for p in sorted(base.glob('*/summary.json'))])
    bank=r.CallBank(r.history_batches()+[p.parent for p in sorted((r.HERE/'results').glob('**/batch/complete.json'))])
    audit_results={};tables=[];comparisons=[];ablation_lengths=[];input_changes=[]
    for dest in destinations:
        audit_results[dest.name]=audit(dest,bank)
        summary=r.read(dest/'summary.json')
        tables += [dict(experiment=dest.name,**row) for row in summary]
        for source in sorted(dest.glob('r*/paths.jsonl')):
            rows=r.read(source);arms={m:{x['task_id']:x for x in rows if x['method']==m} for m in {x['method'] for x in rows}}
            if 'RELINK' not in arms:
                # Structural confirmation shares the initial and main ReLink path.
                main_paths=base/'internal'/source.parent.name/'paths.jsonl'
                if not main_paths.exists():continue
                arms['RELINK']={x['task_id']:x for x in r.read(main_paths) if x['method']=='RELINK'}
            for method in sorted(arms):
                if method=='RELINK':continue
                comparisons.append(dict(experiment=dest.name,round=rows[0]['round'],comparator=method,
                    **paired(arms['RELINK'],arms[method],tasks)))
        specs=[s for p in dest.glob('r1/**/logical_requests.jsonl') for s in r.read(p)]
        relink={s['task_id']:s for s in specs if s['method']=='RELINK'}
        if not relink and confirmation:
            relink={s['task_id']:s for p in (base/'internal/r1').glob('**/logical_requests.jsonl')
                    for s in r.read(p) if s['method']=='RELINK'}
        for method in ('NO_SCOPE','NO_BRIDGE','LEX_MATCH'):
            comparable=[s for s in specs if s['method']==method and s['task_id'] in relink]
            if comparable:
                input_changes.append(dict(experiment=dest.name,method=method,requests=len(comparable),
                    changed=sum(s['request']['messages']!=relink[s['task_id']]['request']['messages'] for s in comparable)))
        for spec in specs:
            match=spec['metadata'].get('length_match')
            if match:ablation_lengths.append(dict(experiment=dest.name,task_id=spec['task_id'],**match))
    actual=[value for folder in sorted(base.glob('**/batch/complete.json')) for value in r.prior.completed(folder.parent).values()]
    for value in actual:
        assert value['prompt_tokens']==len(value['prompt_token_ids'])
        assert value['generation_tokens']==len(value['generated_token_ids'])
    analysis=base/'analysis';analysis.mkdir(exist_ok=True)
    r.save(analysis/'paired_comparisons.json',comparisons)
    r.save(analysis/'audit.json',dict(experiments=audit_results,new_actual_calls=len(actual),
        actual_input_tokens=sum(x['prompt_tokens'] for x in actual),actual_output_tokens=sum(x['generation_tokens'] for x in actual),
        actual_truncated_calls=sum(x['finish_reason']=='length' for x in actual),
        code_sha256=r.prior.batch.sha(__file__)))
    r.save(analysis/'length_matching.json',ablation_lengths)
    r.save(analysis/'ablation_input_changes.json',input_changes)
    with (analysis/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(tables[0]));writer.writeheader();writer.writerows(tables)
    lines=['# '+{'development':'开发集补充实验','confirmation400':'新问题确认实验','family_mistral7':'更换修复模型实验'}[population],
        '', '所有正确数均包含初答正确的题目；token 包含初答及全部修复调用。论文方法为当前本地适配，CHESS 仅使用修复组件。','']
    for dest in destinations:
        allrows=[x for x in tables if x['experiment']==dest.name and x['dataset']=='ALL']
        initial=r.read(dest/'initial_paths.jsonl');method=initial[0]['method'];initial=[x for x in initial if x['method']==method]
        lines += ['## '+dest.name,'',f"共同初答：{sum(x['accepted'] for x in initial)}/{len(initial)}，每题 {sum(x['total_tokens'] for x in initial)/len(initial):.2f} token。",'',
                  '| 方法 | 最多修复轮数 | 最终正确 | 正确率 | 每题全过程 token |',
                  '|---|---:|---:|---:|---:|']
        for row in allrows:
            lines.append(f"| {LABELS.get(row['method'],row['method'])} | {row['round']} | {row['correct']}/{row['n']} | {100*row['correct']/row['n']:.2f}% | {row['mean_all_tokens']:.2f} |")
        lines.append('')
    lines += ['## 配对比较','', '以下均为 ReLink 减去对照；负的 token 差表示 ReLink 消耗较少。区间按数据集分层、数据库组重采样计算，不能用区间包含零证明质量相同。','',
        '| 实验 | 轮数 | 对照 | 正确率差，百分点（95% 区间） | 每题 token 差（95% 区间） |',
        '|---|---:|---|---:|---:|']
    for x in comparisons:
        a=x['accuracy_ci95_pp'];t=x['token_ci95']
        lines.append(f"| {x['experiment']} | {x['round']} | {LABELS.get(x['comparator'],x['comparator'])} | {x['accuracy_difference_pp']:+.2f} [{a[0]:+.2f}, {a[1]:+.2f}] | {x['mean_token_difference']:+.2f} [{t[0]:+.2f}, {t[1]:+.2f}] |")
    if confirmation:
        pop=r.read(base/'population.json')
        lines += ['',f"抽取 {pop['selected']} 题，可评分 {pop['scorable']} 题；不可评分题的初答另消耗 {pop['excluded_initial_tokens']} token，未替换题目。此批为未用问题确认，不是数据库隔离的验证。"]
    if population=='family_mistral7':
        lines += ['', '修复生成模型为 Mistral-7B-Instruct-v0.3（4 bit）。FRESH14 是复用调度器的内部方法代号，此处实际使用的是 Mistral 70 亿参数模型。模型系列、参数规模和分词器同时变化。']
    if population=='development':
        lines += ['', '这些 128 题已用于开发，不能作为独立确认依据。完整结构重新生成在两轮预算下只进行一次追加生成。初答为 140 亿参数模型时，同提示的再次生成可能直接复用确定性初答；应与不修复结果一起阅读。']
    (analysis/'结果汇总.md').write_text('\n'.join(lines)+'\n')
    print('AUDITED AND SUMMARIZED',population,len(actual),'new calls',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('population',choices=['development','confirmation400','family_mistral7'])
    main(parser.parse_args().population)
