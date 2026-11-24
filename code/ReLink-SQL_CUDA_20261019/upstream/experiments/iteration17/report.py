"""Read committed stages only; separate Spider and BIRD with explicit coverage."""
from pathlib import Path
import csv
import hashlib
import json
import os
import tempfile

HERE=Path(__file__).resolve().parent
LABELS=dict(CURRENT='ReLink-SQL',SHARE='SHARE原提示本地流程',CRITIC='CRITIC SQL适配',
            SELF_REFINE='Self-Refine SQL适配',SELF_DEBUG='Self-Debugging聊天流程适配',
            REFLEXION='Reflexion SQL适配',CYCLESQL='CycleSQL',
            DIN_GENTLE='DIN-SQL纠错模板',CHESS_C='CHESS列筛选组件')


def read(path):
    return ([json.loads(x) for x in path.read_text().splitlines()] if path.suffix=='.jsonl'
            else json.loads(path.read_text()))


def write(path,text):
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False) as stream:
        stream.write(text);temporary=stream.name
    os.replace(temporary,path)


def generate():
    stages=[];status=[]
    for cohort in ('smoke32','spider','bird'):
        for method,label in LABELS.items():
            dest=HERE/'results'/cohort/method
            sequence=[]
            if (dest/'initial_paths.jsonl').exists():
                paths=[dest/'initial_paths.jsonl']+[p.parent/'paths.jsonl' for p in sorted(
                    dest.glob('r*/complete.json'),key=lambda x:int(x.parent.name[1:]))]
                for path in paths:
                    rows=read(path)
                    if path.name!='initial_paths.jsonl':
                        assert hashlib.sha256(path.read_bytes()).hexdigest()==read(path.parent/'complete.json')['paths_sha256']
                    row=dict(cohort=cohort,method=method,label=label,round=rows[0]['round'],n=len(rows),
                        initial_correct=sum(x['initial_accepted'] for x in rows),correct=sum(x['accepted'] for x in rows),
                        accuracy=100*sum(x['accepted'] for x in rows)/len(rows),
                        mean_full_tokens=sum(x['total_tokens'] for x in rows)/len(rows),
                        mean_input_tokens=sum(x['input_tokens'] for x in rows)/len(rows),
                        mean_output_tokens=sum(x['output_tokens'] for x in rows)/len(rows),
                        source=str(path.relative_to(HERE)))
                    stages.append(row);sequence.append(row)
            if method=='CYCLESQL':state='待补：作者验证器权重不可获取'
            elif (dest/'error.json').exists():state='技术错误待处理'
            elif (dest/'complete.json').exists():state='完成'
            elif sequence:state='运行中'
            else:state='待运行'
            if cohort=='smoke32' and method not in ('CURRENT','SELF_DEBUG','REFLEXION'):continue
            status.append(dict(cohort=cohort,method=method,label=label,state=state,
                               latest=sequence[-1] if sequence else None))
    data=read(HERE/'data/prepared.json') if (HERE/'data/prepared.json').exists() else {}
    refs=read(HERE/'data/private/complete.json') if (HERE/'data/private/complete.json').exists() else {}
    initial=read(HERE/'initial/complete.json') if (HERE/'initial/complete.json').exists() else None
    generated=HERE/'initial/generation/generated/batch/results.jsonl'
    new_initial=sum(1 for _ in generated.open()) if generated.exists() else 0
    active=[]
    for path in HERE.glob('**/batch/manifest.json'):
        if (path.parent/'complete.json').exists():continue
        requests=path.parent/'requests.jsonl';outputs=path.parent/'results.jsonl'
        active.append(dict(batch=str(path.parent.relative_to(HERE)),
                           requested=sum(1 for _ in requests.open()) if requests.exists() else 0,
                           saved=sum(1 for _ in outputs.open()) if outputs.exists() else 0))
    progress=dict(selected=data.get('counts',{}),unscorable=refs.get('unscorable',{}),
                  initial_complete=initial is not None,initial_new_outputs=new_initial,
                  active_batches=active,methods=status)
    write(HERE/'进度.json',json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
    if stages:
        with (HERE/'逐轮结果.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(stages[0]));writer.writeheader();writer.writerows(stages)
    lines=['# 全量开发集实验进度与结果','',
        'Spider全部1034题；BIRD清理版全部1534题。两个数据集分别报告，不将混合结果替代分数据集结果。已有部分题目参与过方法开发，本轮不是独立盲测。','',
        f'参考执行核查：{refs.get("n",0)}题，当前执行限制下不可评分{len(refs.get("unscorable",{}))}题；不替换。正式质量与费用表采用各方法相同的可评分题目，全部请求成本另行列出。','',
        '共同初答：'+('完成，共2568题。' if initial else f'运行中，新生成已保存{new_initial}条；完全匹配的历史初答另行复用并全额计费。'),'',
        '总体正确率包含初答正确数；每题token包含初答及全部输入输出和失败尝试。反馈统一为官方单数据库执行匹配，不与旧内部反馈轨迹直接拼接。未完成阶段不计入结果。','']
    if active:
        lines += ['当前批次：'+ '；'.join(f"`{x['batch']}` 已保存{x['saved']}/{x['requested']}条" for x in active),'']
    for cohort,title in [('smoke32','接口检查：32题'),('spider','Spider'),('bird','BIRD')]:
        lines += ['## '+title,'','| 方法 | 完成修复阶段 | 初答正确 | 最终正确 | 总体正确率 | 每题token | 状态 |',
                  '|---|---:|---:|---:|---:|---:|---|']
        for item in status:
            if item['cohort']!=cohort:continue
            row=item['latest']
            cells=([str(row['round']),f"{row['initial_correct']}/{row['n']}",f"{row['correct']}/{row['n']}",
                    f"{row['accuracy']:.2f}%",f"{row['mean_full_tokens']:.2f}"] if row else ['—']*5)
            lines.append('| '+' | '.join([item['label'],*cells,item['state']])+' |')
        lines.append('')
    lines += ['CycleSQL缺少作者验证器权重，尚未运行；不以简化替代品充数。DIN-SQL与CHESS标为扩展组件对照。',
              '', '[实验协议](PROTOCOL.md) · [方法及原文来源](METHODS.md) · [机器可读进度](进度.json)']
    write(HERE/'全量实验进度.md','\n'.join(lines)+'\n')
    return progress


if __name__=='__main__':generate()
