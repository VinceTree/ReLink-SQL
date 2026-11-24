"""Compare saved trajectories with label-independent probabilistic stopping.

This is offline replay, not fresh model inference or a held-out experiment.
No request is routed using its future outcome or future token cost.
"""
from pathlib import Path
import csv
import hashlib
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / 'quality_cost'
SEEDS = tuple(range(20260918, 20260938))
PROBABILITIES = np.linspace(0, 1, 101)
TARGETS = (60, 65, 70, 73, 75, 78, 80, 81)
TOLERANCE_PP = 0.5
INPUTS = {}
CACHE = {}
EXTENSIONS = {}
LABELS = dict(RETRY7='70亿模型直接修改', RETRY14='140亿模型直接修改', EXEC14='执行反馈修改',
    FRESH14='完整结构重答', RELINK='原ReLink重复重答', CHESS_C='CHESS组件', CRITIC='CRITIC适配',
    SHARE='SHARE原提示', DIN_GENTLE='DIN-SQL模板', SELF_REFINE='Self-Refine适配', RSL_LINK='RSL-SQL组件',
    current='当前ReLink-SQL', expand_full='ReLink后完整结构重答', tail_fresh='相同前段后完整结构重答',
    critic_tail='相同前段后CRITIC', critic_toggle='程序去重后CRITIC', compiled_toggle='固定去重后独立生成')
STANDARD = {'RETRY7','RETRY14','EXEC14','FRESH14','CHESS_C','CRITIC','SHARE','DIN_GENTLE','SELF_REFINE','RSL_LINK'}
ORDER = ['current','RETRY7','RETRY14','EXEC14','FRESH14','RELINK','DIN_GENTLE','SELF_REFINE',
         'CHESS_C','CRITIC','RSL_LINK','SHARE','expand_full','tail_fresh','critic_tail','critic_toggle','compiled_toggle']
# Publication display scope only. All saved trajectories remain in the analysis.
PAPER_METHODS = ('DIN_GENTLE','SELF_REFINE','CHESS_C','CRITIC','SHARE')
PAPER_ORDER = ('current',*PAPER_METHODS)
DISPLAY_TARGETS = (60,65,70,75,80)


def load(relative, method=None):
    if relative not in CACHE:
        path = ROOT / relative
        with path.open('rb') as stream: INPUTS[relative] = hashlib.file_digest(stream, 'sha256').hexdigest()
        with path.open() as stream:
            CACHE[relative] = [{k:r[k] for k in ('task_id','method','round','accepted','initial_accepted',
                'total_tokens','carried','initial_tokens','call_ids') if k in r}
                for r in map(json.loads, stream)]
    selected = [r for r in CACHE[relative] if method is None or r['method'] == method]
    result = {r['task_id']:r for r in selected}
    assert len(result) == len(selected), (relative, method)
    return result


def chains(cohort):
    dev = cohort == 'development'
    base = ('experiments/iteration13/results/development/continuation' if dev else
            'experiments/iteration13/results/confirmation400/internal')
    initial = load(base+'/initial_paths.jsonl', 'RELINK')
    prefix = f'experiments/iteration14/branches_v6/results/{cohort}/direct_first'
    first = load(prefix+'/initial_paths.jsonl')
    common = [initial, first, load(prefix+'/r1/paths.jsonl'), load(prefix+'/r2/paths.jsonl')]
    tail = f'experiments/iteration15/results/{cohort}'
    result = {'current':common + [load(tail+'/compiled_multiplicity_native/r1/paths.jsonl'),
                                  load(tail+'/compiled_multiplicity_native/r2/paths.jsonl')]}
    methods = ['FRESH14','RELINK','SHARE'] if dev else ['RETRY14','EXEC14','FRESH14','RELINK','CHESS_C','CRITIC','SHARE']
    for method in methods:
        result[method] = [load(base+'/initial_paths.jsonl',method),load(base+'/r1/paths.jsonl',method),load(base+'/r2/paths.jsonl',method)]
    if dev:
        path = 'experiments/iteration12/results/pilot128/deployment_ledger.jsonl'
        # Unlike later trajectories, the original paths keep costs in a ledger.
        ledger_path = ROOT/path
        with ledger_path.open('rb') as stream: INPUTS[path] = hashlib.file_digest(stream,'sha256').hexdigest()
        ledger = {(r['task_id'],r['method'],r['round']):r for r in map(json.loads,ledger_path.read_text().splitlines())}
        for method in ['RETRY7','RETRY14','EXEC14','DIN_GENTLE','SELF_REFINE','CRITIC','CHESS_C','RSL_LINK']:
            steps=[initial]
            for stage in [1,2]:
                rows=load(f'experiments/iteration12/results/pilot128/r{stage}/paths.jsonl',method)
                paid={}
                for tid,row in rows.items():
                    entry=ledger[tid,method,stage]
                    assert entry['accepted']==row['accepted'] and entry['call_ids']==row['call_ids']
                    assert entry['initial_tokens']==initial[tid]['total_tokens']
                    paid[tid]=dict(row,total_tokens=entry['total_tokens'])
                steps.append(paid)
            result[method]=steps
    else:
        result['expand_full']=[initial,first,load('experiments/iteration13/results/matched_quality400/expand_full/r1/paths.jsonl')]
    result['tail_fresh']=common+[load(tail+'/fresh_restart/r1/paths.jsonl')]
    result['critic_tail']=common+[load(prefix+'/r3/paths.jsonl')]
    for method in ['critic_toggle','compiled_toggle']:
        result[method]=common+[load(tail+f'/{method}/r1/paths.jsonl'),load(tail+f'/{method}/r2/paths.jsonl')]
    for method in STANDARD|{'RELINK'}:
        if method not in result:continue
        dest=ROOT/'experiments/iteration16/results'/cohort/method
        if not (dest/'prepared.json').exists():continue
        completed=[]
        for stage in range(3,9):
            marker=dest/f'r{stage}/complete.json'
            if not marker.exists():break
            path=marker.parent/'paths.jsonl'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==json.loads(marker.read_text())['paths_sha256']
            result[method].append(load(str(path.relative_to(ROOT)),method));completed.append(stage)
        stop=None
        if (dest/'complete.json').exists():
            marker=dest/'complete.json'
            INPUTS[str(marker.relative_to(ROOT))]=hashlib.sha256(marker.read_bytes()).hexdigest()
            stop=json.loads(marker.read_text())['stop_reason']
        EXTENSIONS[cohort,method]=dict(completed_rounds=completed,stop_reason=stop,
                                     state='complete' if stop else 'running')
    return initial,result


def replay(cohort):
    initial,methods=chains(cohort)
    ids=sorted(initial);n=len(ids);initial_correct=sum(r['accepted'] for r in initial.values())
    errors=n-initial_correct
    initial_tokens=sum(r['total_tokens'] for r in initial.values())/n
    uniforms=np.array([[int.from_bytes(hashlib.sha256(f'{seed}|{tid}'.encode()).digest()[:8],'big')/2**64
                       for tid in ids] for seed in SEEDS])
    configs=[];checkpoints=[]
    for method,steps in methods.items():
        quality=[];cost=[];deliveries=[]
        cumulative=np.zeros(n)
        for stage,step in enumerate(steps):
            assert set(step)==set(ids)
            y=np.array([step[t]['accepted'] for t in ids],dtype=int)
            c=np.array([step[t]['total_tokens'] for t in ids],dtype=float)
            if stage:
                assert np.all(y>=quality[-1]) and np.all(c>=cost[-1])
                # Success stops immediately, so it cannot accumulate later cost.
                accepted=quality[-1].astype(bool)
                assert np.array_equal(c[accepted],cost[-1][accepted]),(method,stage)
                cumulative=cumulative+np.array([not step[t].get('carried',True) for t in ids])
            else:
                assert np.array_equal(y,np.array([initial[t]['accepted'] for t in ids]))
                assert np.array_equal(c,np.array([initial[t]['total_tokens'] for t in ids]))
            quality.append(y);cost.append(c);deliveries.append(cumulative.copy())
            checkpoints.append(dict(cohort=cohort,method=method,label=LABELS[method],stage=stage,n=n,
                initial_correct=initial_correct,correct=int(y.sum()),accuracy=100*y.sum()/n,
                full_tokens=float(c.mean()),repair_tokens_per_initial_error=float((c.mean()-initial_tokens)*n/errors),
                mean_repair_deliveries=float(cumulative.mean())))
        for stage in range(len(steps)-1):
            for probability in PROBABILITIES:
                # The decision uses only a public task identifier and fixed seed.
                # Outcome/cost arrays are consulted solely to score that decision.
                go=uniforms<probability
                y=np.where(go,quality[stage+1],quality[stage])
                c=np.where(go,cost[stage+1],cost[stage])
                d=np.where(go,deliveries[stage+1],deliveries[stage])
                counts=y.sum(axis=1)
                rates=100*counts/n
                tokens=c.mean(axis=1)
                configs.append(dict(cohort=cohort,method=method,label=LABELS[method],stage=stage,
                    probability=float(probability),n=n,initial_correct=initial_correct,
                    mean_correct=float(counts.mean()),accuracy=float(rates.mean()),
                    accuracy_seed_min=float(rates.min()),accuracy_seed_max=float(rates.max()),
                    full_tokens=float(tokens.mean()),token_seed_min=float(tokens.min()),token_seed_max=float(tokens.max()),
                    repair_tokens_per_initial_error=float((tokens.mean()-initial_tokens)*n/errors),
                    mean_repair_deliveries=float(d.mean()),seeds=len(SEEDS)))
    matched=[]
    for target in TARGETS:
        for method in methods:
            candidates=[r for r in configs if r['method']==method and target-1e-9<=r['accuracy']<=target+TOLERANCE_PP+1e-9]
            if candidates:
                best=min(candidates,key=lambda x:(x['full_tokens'],x['accuracy'],x['stage'],x['probability']))
                matched.append(dict(best,target=target,status='covered'))
            else:
                extension=EXTENSIONS.get((cohort,method),{})
                matched.append(dict(cohort=cohort,method=method,label=LABELS[method],target=target,status='not_covered',
                    extension_state=extension.get('state','not_extended'),stop_reason=extension.get('stop_reason')))
    return configs,checkpoints,matched


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def markdown(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,r))+' |' for r in rows]])


def missing_label(row):
    if row.get('extension_state')=='running':return '未达，续跑中'
    return {'round_budget_8':'八轮内未达','token_budget_6000':'token预算内未达',
            'no_gain_three_rounds':'未达，连续三轮无提升','unchanged_input_cycle':'未达，相同输入循环',
            'deterministic_request_cycle':'未达，相同输入循环'}.get(row.get('stop_reason'),'未覆盖')


def plot(configs,points):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/System/Library/Fonts/Hiragino Sans GB.ttc',size=8)
    colors={'current':'#154f84','RETRY7':'#d49b55','RETRY14':'#bc6c25','EXEC14':'#ce4257',
            'FRESH14':'#777777','RELINK':'#313131','CHESS_C':'#9163a7','CRITIC':'#429e89',
            'SHARE':'#a6804c','DIN_GENTLE':'#7f9a40','SELF_REFINE':'#609cbb','RSL_LINK':'#b57286',
            'expand_full':'#aa347e','tail_fresh':'#777777','critic_tail':'#429e89',
            'critic_toggle':'#9163a7','compiled_toggle':'#ce4257'}
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.7))
    for ax,(cohort,size) in zip(axes,[('exploration400',400),('development',128)]):
        for method in PAPER_ORDER:
            rows=[r for r in configs if r['cohort']==cohort and r['method']==method]
            if not rows:continue
            rows.sort(key=lambda r:(r['accuracy'],-r['full_tokens']))
            front=[];best=math.inf
            for r in reversed(rows):
                if r['full_tokens']<best:
                    front.append(r);best=r['full_tokens']
            front.reverse()
            ax.plot([r['accuracy'] for r in front],[r['full_tokens'] for r in front],
                    color=colors[method],lw=2.4 if method=='current' else 1.2,label=LABELS[method])
            full=[p for p in points if p['cohort']==cohort and p['method']==method]
            ax.scatter([r['accuracy'] for r in full],[r['full_tokens'] for r in full],color=colors[method],s=12)
        ax.set_title(f'{size}题：论文方法比较',fontproperties=font)
        ax.set_xlabel('总体正确率：最终正确题数 / 全部题目数（%）',fontproperties=font)
        ax.set_ylabel('每题全过程token（包含初答）',fontproperties=font)
        ax.grid(alpha=.2)
        ax.legend(prop=font,loc='upper left',bbox_to_anchor=(0,-.20),ncol=2,borderaxespad=0)
    fig.suptitle('论文主实验：线连接停止预算回放点，圆点为完整阶段实测结果',fontproperties=font)
    fig.tight_layout(pad=1.2)
    fig.savefig(OUT/'总体正确率与token.png',dpi=200)
    fig.savefig(OUT/'总体正确率与token.pdf')
    plt.close(fig)


def quality_table(matches,cohort,methods,targets=DISPLAY_TARGETS,detail=False):
    values=[]
    for method in methods:
        rows=[r for r in matches if r['cohort']==cohort and r['method']==method]
        if not rows:continue
        cells=[]
        for target in targets:
            r=next(r for r in rows if r['target']==target)
            cells.append((f"{r['full_tokens']:.2f}（{r['accuracy']:.2f}%）" if detail else f"{r['full_tokens']:.2f}")
                         if r['status']=='covered' else (missing_label(r) if detail else '—'))
        values.append([LABELS[method],*cells])
    return markdown(['方法']+[f'正确率{t}%' for t in targets],values)


def highest_checkpoint(points,cohort,method):
    rows=[r for r in points if r['cohort']==cohort and r['method']==method]
    maximum=max(r['correct'] for r in rows)
    return min((r for r in rows if r['correct']==maximum),key=lambda r:(r['full_tokens'],r['stage']))


def reports(points,matches):
    main=['# 总体正确率与token：论文方法比较','',
        '主表比较ReLink-SQL与CRITIC、SHARE、CHESS，以及128题上已测的DIN-SQL和Self-Refine。实验中的实现分别标为适配、组件或模板，不把局部实现当作原论文完整系统。方法范围依据论文比较任务确定，与结果高低无关。','',
        '总体正确率＝最终正确题数／全部题目数，包含初答正确数。400题初答为225/400（56.25%），128题为75/128（58.59%）。token均为每题全过程输入输出，包含初答及全部失败尝试。','',
        '## 相近总体正确率下的token','',
        '下表基于原有实验与增加预算后已完成的真实轨迹，采用固定概率停止回放；实际平均总体正确率须达到列示目标且高出不超过0.5个百分点，选择其中最低平均token配置。“—”表示本次已测配置没有覆盖该目标，具体停止原因见后文。回放使用20个固定种子，配置是开发样本的事后选择，不是新数据上的独立验证。','']
    for cohort,size in [('exploration400',400),('development',128)]:
        main += [f'### {size}题','',quality_table(matches,cohort,PAPER_ORDER),'']
    main += ['## 已测最高正确率与对应费用','',
        '各方法取本次最高正确率首次达到的已完成阶段，token包含此前全部调用。后续没有提高正确率的轮次继续保存在完整记录中，不把这些额外费用当作达到该正确率的必要成本。对照最多探索八轮，当前ReLink-SQL最多五次修复交付，交互上限不同。','']
    for cohort,size in [('exploration400',400),('development',128)]:
        values=[]
        for method in PAPER_ORDER:
            if not any(r['cohort']==cohort and r['method']==method for r in points):continue
            r=highest_checkpoint(points,cohort,method)
            values.append([r['label'],r['stage'],f"{r['correct']}/{size}",f"{r['accuracy']:.2f}%",f"{r['full_tokens']:.2f}"])
        main += [f'### {size}题','',markdown(['方法','修复轮次','最终正确数','总体正确率','每题token'],values),'']
    main += ['## 对照实现与停止条件','',
        'CRITIC与Self-Refine为SQL任务适配；SHARE使用作者原提示的本地五阶段流程；CHESS为列筛选与修复组件；DIN-SQL为gentle纠错模板。400题尚无DIN-SQL和Self-Refine结果，不用128题成绩补入。','',
        '续跑上限为八轮，下一轮开始前检查平均全过程6000 token预算；连续三轮没有新增正确时也可停止。未达到目标只说明本配置与预算内未达到。分方法的停止原因和五轮结果见[论文基线续跑结果](../../../iteration16/增加预算后的基线结果.md)。','',
        '## 论文主图','', '![总体正确率与token](总体正确率与token.png)','',
        '当前结果支持在共同覆盖的多个总体正确率水平上降低token消耗。未被对照覆盖的高正确率目标，不能据此推算对方费用或声称全范围领先。','',
        '常规重试、内部组合、旧ReLink版本和未纳入主表的组件记录见[补充对照与内部结构实验](补充对照与内部结构实验.md)。它们用于机制分析和复现，不列为独立论文方法。全部数字仍保存在[完整统计CSV](总体正确率比较.csv)。']
    (OUT/'总体正确率与token比较.md').write_text('\n'.join(main)+'\n')
    supplement=['# 补充对照与内部结构实验','',
        '本文件保留常规基线、内部设计对照和未纳入论文主表的组件。这里只调整呈现范围，不修改任何正确数、token或原始调用。内部组合用于分析设计选择，不能作为独立论文方法。RSL-SQL对应的字段关联组件本次不作为主要纠错对照；其原始结果仍可复核。','',
        '主表范围：CRITIC、SHARE、CHESS、DIN-SQL、Self-Refine与当前ReLink-SQL。主表见[总体正确率与token](总体正确率与token比较.md)。','']
    groups=[('常规基线',['current','RETRY7','RETRY14','EXEC14','FRESH14']),
            ('内部结构对照及消融分析材料',['current','RELINK','expand_full','tail_fresh','critic_tail','critic_toggle','compiled_toggle']),
            ('未纳入主表的组件记录',['current','RSL_LINK'])]
    for cohort,size in [('exploration400',400),('development',128)]:
        for title,group in groups:
            available=[m for m in group if any(r['cohort']==cohort and r['method']==m for r in matches)]
            if len(available)<2:continue
            supplement += [f'## {size}题：{title}','',quality_table(matches,cohort,available,TARGETS,True),'']
    supplement += ['## 不应由主表筛选掩盖的结果','',
        '400题中，ReLink后完整结构重答在73.50%总体正确率时为839.86 token，当前流程在73.25%时为870.08 token；78%目标下，程序去重后CRITIC也更便宜。共享相同ReLink第一步的流程，在低预算时费用相同。内部对照仍用于分析这些设计取舍，主表范围调整不改变这些事实。','',
        '详细记录：[全部完整阶段](完整阶段结果.csv)、[全部停止配置](停止配置回放.csv)、[总体正确率统计](总体正确率比较.csv)。']
    (OUT/'补充对照与内部结构实验.md').write_text('\n'.join(supplement)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    configs=[];points=[];matches=[]
    for cohort in ['development','exploration400']:
        a,b,c=replay(cohort);configs+=a;points+=b;matches+=c
    for name,data in [('停止配置回放.csv',configs),('完整阶段结果.csv',points),('总体正确率比较.csv',matches)]:
        write_csv(OUT/name,data)
    plot(configs,points)
    reports(points,matches)
    # Keep the previously shared link usable while making the new metric explicit.
    (OUT/'同等修复率下的token比较.md').write_text('# 比较口径已更新\n\n当前按全部题目的最终正确率比较token，见[总体正确率与token：论文方法比较](总体正确率与token比较.md)。原始调用记录不变。\n')
    for relative,digest in INPUTS.items():
        with (ROOT/relative).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==digest
    (OUT/'统计核对.json').write_text(json.dumps(dict(metric='overall_accuracy',denominator='all original requests',
        seeds=SEEDS,probability_step=.01,targets=TARGETS,tolerance_pp=TOLERANCE_PP,new_model_calls=0,
        offline_replay=True,selection='post-hoc grid, no per-query future-outcome routing',
        n_configs=len(configs),n_checkpoints=len(points),input_files=INPUTS,
        paper_methods=PAPER_METHODS,display_only_change=True,all_comparisons_retained=True,
        extensions={c+':'+m:v for (c,m),v in EXTENSIONS.items()}),ensure_ascii=False,indent=2)+'\n')
    for cohort in ['exploration400','development']:
        print(cohort)
        for method in ORDER:
            rows=[r for r in matches if r['cohort']==cohort and r['method']==method and r['status']=='covered']
            if rows:print(LABELS[method],[(r['target'],round(r['accuracy'],3),round(r['full_tokens'],2)) for r in rows])


if __name__=='__main__':main()
