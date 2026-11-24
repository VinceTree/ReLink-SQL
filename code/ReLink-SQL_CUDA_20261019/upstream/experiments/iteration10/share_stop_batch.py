"""Reuse the durable batch runner with SHARE's original streamed string stops."""
from pathlib import Path
from types import FunctionType
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from iteration4 import batch_public_generate as base

STOPS=['</FINAL_ANSWER>','<|EOT|>']


def drain(generator,uid_items,tokenizer,on_complete,progress=None):
    eos=set(tokenizer.eos_token_ids)
    states={u:dict(tokens=[],pieces=[],detokenizer=tokenizer.detokenizer,prompt_processed=0,completed=False) for u in uid_items}
    if progress is not None:progress.update(states)
    pending=set(uid_items)
    while pending:
        prompts,generated=generator.next()
        if not prompts and not generated:raise RuntimeError('Backend ended with unfinished SHARE requests')
        for r in prompts:states[r.uid]['prompt_processed']=max(states[r.uid]['prompt_processed'],r.progress[0])
        for r in generated:
            assert r.uid in pending
            state,item=states[r.uid],uid_items[r.uid];state['tokens'].append(int(r.token))
            finish=r.finish_reason;is_eos=r.token in eos;count=len(state['tokens'])
            assert finish in (None,'stop','length') and count<=item['request']['max_tokens']
            assert (finish=='stop')==is_eos
            if finish=='length':assert count==item['request']['max_tokens']
            if not is_eos:state['detokenizer'].add_token(int(r.token))
            if finish is not None:state['detokenizer'].finalize()
            state['pieces'].append(state['detokenizer'].last_segment)
            raw=''.join(state['pieces']);matched=next((s for s in STOPS if s in raw),None)
            if matched:
                if finish is None:generator.remove([r.uid])
                finish='stop_string'
            if finish is None:continue
            text=raw
            for stop in STOPS:text=text.split(stop,1)[0]
            row=dict(id=item['request']['id'],text=text,raw_text=raw,prompt_tokens=item['prompt_tokens'],
                generation_tokens=count,text_generation_tokens=count-int(is_eos),terminal_eos_count=int(is_eos),
                generated_token_ids=list(state['tokens']),prompt_token_ids=item['prompt_token_ids'],finish_reason=finish,
                matched_stop_string=matched,eos_token_ids=sorted(eos),rendered_prompt_sha256=item['rendered_prompt_sha256'],
                messages_sha256=item['messages_sha256'],prompt_add_special_tokens=item['add_special_tokens'])
            on_complete(item,row);state['completed']=True;pending.remove(r.uid)
    return states


def prepare(requests,output):
    result=base.prepare(requests,output,prompt_encoding='explicit_no_special_tokens')
    p=output/'manifest.json';m=json.loads(p.read_text())
    m['stop_policy']='Original SHARE streamed text stops plus tokenizer EOS; every emitted token including stop text is charged'
    m['share_stop_extension']=dict(source=str(Path(__file__).resolve()),sha256=base.sha(__file__),stops=STOPS,
        contract='Only drain_batch is replaced in the durable original runner; model, batching, prompt and interruption accounting are inherited')
    base.atomic_write(p,json.dumps(m,ensure_ascii=False,indent=2)+'\n')
    return result


def run(output):
    m=json.loads((output/'manifest.json').read_text());e=m['share_stop_extension']
    assert e['sha256']==base.sha(__file__) and e['stops']==STOPS
    assert m['prompt_encoding']=='explicit_no_special_tokens'
    # Keep the original runner file and all its validations intact. The extension
    # has its own committed hash above, including this injection point.
    runner=FunctionType(base.run.__code__,dict(base.run.__globals__,drain_batch=drain),base.run.__name__)
    return runner(output)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(run(a.output))
