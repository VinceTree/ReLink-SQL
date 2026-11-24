"""Analyze committed full-dev trajectories without new model calls or tuning."""
from pathlib import Path
import argparse
import csv
import json
import time

from . import report
from .dataset import HERE, DATA, r
from ..iteration13.analyze import paired
from ..iteration15.paper_report import matched_quality as matched

OUT=HERE/'analysis'
MAIN=('CURRENT','SHARE','CRITIC','SELF_REFINE','SELF_DEBUG','REFLEXION')
TARGETS=(60,65,70,75,80,85,90,95)
FIELDS=('task_id','method','round','accepted','initial_accepted','total_tokens',
        'input_tokens','output_tokens','carried','call_ids')


def paths(path):
    if path.name=='paths.jsonl':
        assert r.prior.batch.sha(path)==r.read(path.parent/'complete.json')['paths_sha256']
    rows=[{k:x[k] for k in FIELDS if k in x} for x in r.read(path)]
    result={x['task_id']:x for x in rows}
    assert len(result)==len(rows)
    return result


def chains(cohort):
    methods={}
    for name in report.LABELS:
        dest=HERE/'results'/cohort/name
        if not (dest/'initial_paths.jsonl').exists():continue
        stages=[paths(dest/'initial_paths.jsonl')]
        for rnd in range(1,9):
            if not (dest/f'r{rnd}/complete.json').exists():break
            stages.append(paths(dest/f'r{rnd}/paths.jsonl'))
        if len(stages)>1:methods[name]=stages
    if not methods:return {},{}
    initial=next(iter(methods.values()))[0]
    for stages in methods.values():
        assert set(stages[0])==set(initial)
        for tid,row in stages[0].items():
            assert all(row[k]==initial[tid][k] for k in ('accepted','total_tokens','call_ids'))
    return initial,methods


def initial_costs():
    """Include even the initial requests that could not receive valid feedback."""
    folder=HERE/'initial';marker=r.read(folder/'complete.json')
    assert r.prior.batch.sha(folder/'initial.jsonl')==marker['initial_sha256']
    calls=r.read(folder/'generation/records.jsonl')
    by_id={x['request_id']:x['task_id'] for x in calls}
    sources={Path(x['reused_from']) for x in calls if 'reused_from' in x}
    sources.add(folder/'generation/generated/batch')
    costs={}
    for source in sources:
        assert r.prior.batch.sha(source/'results.jsonl')==r.read(source/'complete.json')['results_sha256']
        with (source/'results.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['id'] in by_id:
                    costs[by_id[row['id']]]=row['prompt_tokens']+row['generation_tokens']
    assert len(costs)==len(calls)==marker['n']
    return costs


def csv_write(path,rows):
    if not rows:return
    # Report generation may overlap with a progress refresh; publish atomically.
    import io
    stream=io.StringIO(newline='')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    report.write(path,stream.getvalue())


def generate():
    OUT.mkdir(exist_ok=True)
    tasks={t['id']:t for t in r.read(DATA/'public.jsonl')}
    failed=r.read(DATA/'private/complete.json')['unscorable']
    costs=initial_costs()
    points=[];configs=[];targets=[];comparisons=[];coverage=[];extensions={}
    inputs={str(HERE/'PROTOCOL.md'):r.prior.batch.sha(HERE/'PROTOCOL.md'),
            str(Path(__file__)):r.prior.batch.sha(__file__)}
    for cohort in ('spider','bird'):
        initial,methods=chains(cohort)
        if not methods:continue
        ids={tid for tid,t in tasks.items() if t['dataset']==cohort}
        assert set(initial)==ids-set(failed)
        excluded=ids-set(initial)
        coverage.append(dict(dataset=cohort,total=len(ids),scorable=len(initial),excluded=len(excluded),
            initial_correct=sum(x['accepted'] for x in initial.values()),
            all_initial_tokens=sum(costs[t] for t in ids),excluded_initial_tokens=sum(costs[t] for t in excluded)))
        for method in methods:
            dest=HERE/'results'/cohort/method
            r.verify(dest)
            marker=r.read(dest/'complete.json') if (dest/'complete.json').exists() else {}
            extensions[cohort,method]=dict(state='complete' if marker else 'running',
                                         stop_reason=marker.get('stop_reason'))
            for path in [dest/'initial_paths.jsonl',*dest.glob('r*/complete.json')]:
                inputs[str(path)]=r.prior.batch.sha(path)
        replay=r.methods.selectors.clone(matched.replay,chains=lambda _: (initial,methods),
            LABELS=report.LABELS,TARGETS=TARGETS,EXTENSIONS=extensions)
        cs,ps,ms=replay(cohort)
        for row in ps:
            row['all_request_mean_tokens']=(row['full_tokens']*len(initial)+sum(costs[t] for t in excluded))/len(ids)
        configs+=cs;points+=ps;targets+=ms
        current=methods.get('CURRENT',[])
        for method,stages in methods.items():
            if method=='CURRENT':continue
            for rnd in range(1,min(len(current),len(stages),6)):
                comparisons.append(dict(dataset=cohort,round=rnd,comparator=method,
                                        **paired(current[rnd],stages[rnd],tasks)))
    csv_write(OUT/'逐阶段正确率与token.csv',points)
    csv_write(OUT/'相近正确率回放.csv',targets)
    csv_write(OUT/'回放全部配置.csv',configs)
    csv_write(OUT/'配对区间.csv',comparisons)
    report.write(OUT/'统计来源.json',json.dumps(dict(inputs=inputs,coverage=coverage,
        complete=(HERE/'full_complete.json').exists(),target_tolerance_pp=matched.TOLERANCE_PP,
        stopping_seeds=list(matched.SEEDS),no_new_inference=True),ensure_ascii=False,indent=2)+'\n')
    lines=['# 完整开发集：总体正确率与token','',
        '所有正确数均包含初答正确题目，token包含初答和后续全部模型输入输出。两个数据集分别统计。当前结果来自同一批公开开发集，不是独立盲测。','',
        '主比较固定最多五次修复交付；基线另探索至八轮。以下只读取已提交阶段，缺失轮次不填补、不推算。CycleSQL缺少作者验证器权重，未计入已完成对照。','']
    for pop in coverage:
        ds=pop['dataset'];n=pop['scorable']
        initial_rate=100*pop['initial_correct']/n
        lines += [f'## {ds.upper()}','',
            f"共{pop['total']}题，统一可评分{n}题；初答正确{pop['initial_correct']}/{n}（{initial_rate:.2f}%）。"
            f"另有{pop['excluded']}题参考执行未通过限制，未替换；其初答共计{pop['excluded_initial_tokens']} token。",'',
            '### 五轮主比较','', '| 方法 | 完成轮次 | 最终正确 | 总体正确率 | 每题全过程token |',
            '|---|---:|---:|---:|---:|']
        for name in MAIN:
            rows=[p for p in points if p['cohort']==ds and p['method']==name and p['stage']<=5]
            if not rows:lines.append(f'| {report.LABELS[name]} | — | — | — | 待完成 |');continue
            row=rows[-1]
            lines.append(f"| {row['label']} | {row['stage']}{'（运行中）' if row['stage']<5 else ''} | "
                         f"{row['correct']}/{n} | {row['accuracy']:.2f}% | {row['full_tokens']:.2f} |")
        lines += ['', '### 相近总体正确率时的token','',
            '回放采用固定公开ID和20个种子决定是否继续下一阶段，接受后停止。目标容差为0至+0.5个百分点。配置由本数据集事后选择，仅用于描述费用曲线，不是预先选定策略的独立测试成绩。','',
            '| 方法 | '+' | '.join(f'{t}%' for t in TARGETS)+' |',
            '|---|'+'---:|'*len(TARGETS)]
        for name in MAIN:
            cells=[]
            for target in TARGETS:
                row=next((x for x in targets if x['cohort']==ds and x['method']==name and x['target']==target),None)
                if target<initial_rate:cell='初答已超过'
                elif row and row['status']=='covered':cell=f"{row['full_tokens']:.2f}"
                elif row and extensions[ds,name]['state']=='complete':cell='预算内未达'
                else:cell='待完成'
                cells.append(cell)
            lines.append('| '+' | '.join([report.LABELS[name],*cells])+' |')
        lines += ['', '逐阶段原始点、八轮扩展和DIN-SQL/CHESS组件的数字见CSV与实时进度；不将尚未达到目标解释为方法不可能达到。','']
    lines += ['## 配对比较','',
        '以下为第五阶段的ReLink-SQL减去对照，按数据库分组进行10000次配对重采样。负token差表示费用较低；正确率区间包含零不能据此证明不劣。','',
        '| 数据集 | 对照 | 正确率差，百分点（95%区间） | 每题token差（95%区间） |',
        '|---|---|---:|---:|']
    for row in comparisons:
        if row['round']!=5 or row['comparator'] not in MAIN:continue
        a=row['accuracy_ci95_pp'];t=row['token_ci95']
        lines.append(f"| {row['dataset']} | {report.LABELS[row['comparator']]} | "
            f"{row['accuracy_difference_pp']:+.2f} [{a[0]:+.2f}, {a[1]:+.2f}] | "
            f"{row['mean_token_difference']:+.2f} [{t[0]:+.2f}, {t[1]:+.2f}] |")
    lines += ['', '全部请求的平均费用另列在CSV的all_request_mean_tokens列，包含不可评分题的初答，不把这些题判为错误。主表费用使用与正确率相同的可评分集合。','',
        '[完整逐阶段结果](逐阶段正确率与token.csv) · [相近正确率统计](相近正确率回放.csv) · [配对区间](配对区间.csv) · [实时进度](../全量实验进度.md)']
    report.write(OUT/'总体正确率与token比较.md','\n'.join(lines)+'\n')
    print('FULL DEV ANALYSIS',len(points),'committed stage points',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    if args.wait:
        while not (HERE/'full_complete.json').exists():time.sleep(30)
    generate()
