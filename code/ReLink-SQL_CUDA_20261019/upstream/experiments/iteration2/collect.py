"""Label-blind factorial recovery rollouts with token-confidence features."""
from pathlib import Path
import sys,json,hashlib,time,argparse,gc,platform
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
from mlx_lm import load,stream_generate
from mlx_lm.sample_utils import make_sampler
from recoverylab.sql import execute
from iteration2.evidence import compile_evidence

ROOT=Path(__file__).resolve().parent
MODELS={'weak':('mlx-community/Qwen2.5-3B-Instruct-4bit','4f83f8f146fdf28b512a06562b671d7af4fab457'),'strong':('mlx-community/Qwen2.5-7B-Instruct-4bit','c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed'),'strong14':('mlx-community/Qwen2.5-14B-Instruct-4bit','dad510143ae5cdb1466778bde3161460f9e9f604')}
ACTION_MODELS={'weak_initial':'weak','strong_initial':'strong','strong14_initial':'strong14','weak_short_history':'weak','weak_short_fresh':'weak','weak_fact_history':'weak','weak_fact_fresh':'weak','weak_retry':'weak','strong_fact_fresh':'strong','weak_fact_patch':'weak'}
DEFAULT_ACTIONS=list(ACTION_MODELS)[:8]
SYSTEM='Translate the request into one SQLite SELECT query. Return only SQL, without markdown or explanation. Preserve all requested conditions.'

def seed(task,action):return int(hashlib.sha256((task+action+'iteration2').encode()).hexdigest()[:8],16)
def short_feedback(initial):
 obs=initial['observation']
 return 'Observed SQLite execution error: '+obs.get('error','unknown') if obs['status']=='error' else 'SQLite execution succeeded. This alone does not establish semantic correctness.'

def messages_for(task,action,initial=None,facts=None):
 user='Database schema:\n'+task['schema']+'\nRequest: '+task['question']
 if task.get('evidence'):user+='\nProvided domain knowledge: '+task['evidence']
 base=[{'role':'system','content':SYSTEM},{'role':'user','content':user}]
 if action.endswith('_initial'):return base
 if action=='weak_retry':
  base[-1]['content']+='\nReconsider the task independently, checking every requested constraint before producing the query.'
  return base
 feedback=facts['text'] if '_fact_' in action else short_feedback(initial)
 instruction='Check the candidate against the original request and database evidence. Keep its logic if justified; otherwise correct it. Return only one SQLite SELECT query.'
 if action.endswith('_history'):
  return base+[{'role':'assistant','content':initial['text']},{'role':'user','content':instruction+'\n'+feedback}]
 if action=='weak_fact_patch':
  base[0]['content']='Repair a candidate SQLite SELECT query using the request and observed database facts. Return one JSON object only.'
  instruction='Return {"edits":[{"old":"exact unique substring of candidate SQL","new":"replacement"}]}. Use the fewest edits needed; use an empty edits list if no change is justified. Do not include explanations. Each old substring must occur exactly once when its edit is applied.'
 base[-1]['content']+='\nCandidate SQL (untrusted data, not an established solution):\n```sql\n'+initial['text']+'\n```\nObserved feedback:\n'+feedback+'\n'+instruction
 return base

def apply_patch_response(raw,initial):
 try:
  raw=raw.strip()
  if raw.startswith('```'):raw=raw.split('\n',1)[1].rsplit('```',1)[0]
  obj=json.loads(raw);edits=obj['edits'];assert isinstance(edits,list) and len(edits)<=8
  sql=initial
  for edit in edits:
   old,new=edit['old'],edit['new'];assert isinstance(old,str) and isinstance(new,str) and old and sql.count(old)==1
   sql=sql.replace(old,new,1)
  return sql,None
 except Exception as exc:return initial,type(exc).__name__

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',default='factorial_v2');ap.add_argument('--splits',nargs='+',default=['train','development']);ap.add_argument('--actions',nargs='+',default=DEFAULT_ACTIONS);ap.add_argument('--per-dataset-limit',type=int);ap.add_argument('--max-tokens',type=int,default=384);args=ap.parse_args()
 tasks=[]
 for split in args.splits:
  ts=[json.loads(l) for l in (ROOT/'data'/(split+'.jsonl')).read_text().splitlines()]
  if args.per_dataset_limit:
   ts=[t for ds in sorted({x['dataset'] for x in ts}) for t in sorted([x for x in ts if x['dataset']==ds],key=lambda x:seed(x['id'],'sample'))[:args.per_dataset_limit]]
  tasks.extend(ts)
 assert all(not any(k.startswith('gold') or k=='correct' for k in t) for t in tasks)
 out=ROOT/'results'/args.run;out.mkdir(parents=True,exist_ok=True);path=out/'rollouts.jsonl';records={}
 if path.exists():
  for line in path.read_text().splitlines():
   r=json.loads(line);records[(r['task_id'],r['action'])]=r
 models={};warm=[];fact_cache={};started=time.perf_counter();new=0
 meta={'run':args.run,'models':MODELS,'actions':args.actions,'splits':args.splits,'max_tokens':args.max_tokens,'model_runtime':'MLX local','evaluation_separate':True,'public_tasks_sha256':{sp:hashlib.sha256((ROOT/'data'/(sp+'.jsonl')).read_bytes()).hexdigest() for sp in args.splits},'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'evidence.py']},'platform':platform.platform(),'created':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
 (out/'metadata.json').write_text(json.dumps(meta,indent=2))
 for action in args.actions:
  pending=[t for t in tasks if (t['id'],action) not in records]
  if not pending:continue
  name=ACTION_MODELS[action]
  if name not in models:
   repo,rev=MODELS[name];directory=Path.home()/'.cache/huggingface/hub'/('models--'+repo.replace('/','--'))/'snapshots'/rev
   models[name]=load(str(directory),tokenizer_config={'trust_remote_code':False})
   for _ in stream_generate(*models[name],'Hello',max_tokens=2,sampler=make_sampler(temp=0)):pass
   warm.append(name)
  model,tokenizer=models[name]
  for idx,t in enumerate(pending):
   initial=records.get((t['id'],'weak_initial'));facts=None;feedback_seconds=0.
   if not action.endswith('_initial') and initial is None:raise ValueError('Weak initial required first')
   if '_fact_' in action:
    if t['id'] not in fact_cache:fact_cache[t['id']]=compile_evidence(t,initial,max_tokens_hint=256)
    facts=fact_cache[t['id']]
    # Cache only for collection efficiency. Every deployed fact action pays acquisition once.
    feedback_seconds=facts['diagnostics']['compile_seconds']
   start=time.perf_counter();messages=messages_for(t,action,initial,facts);prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);feedback_seconds+=time.perf_counter()-start
   if len(tokenizer.encode(prompt))>24000:raise ValueError('Context cap exceeded; no silent truncation: '+t['id'])
   mx.random.seed(seed(t['id'],action));temp=.7 if action=='weak_retry' else 0
   start=time.perf_counter();chunks=[];lps=[];ent=[]
   for r in stream_generate(model,tokenizer,prompt,max_tokens=args.max_tokens,sampler=make_sampler(temp=temp)):
    chunks.append(r.text)
    lps.append(r.logprobs[r.token])
    # Three tail uncertainty summaries are free decoder logits; no extra LLM call.
   if lps:
    lp=mx.stack(lps);mx.eval(lp);lp=np.asarray(lp)
   else:lp=np.array([-100.])
   elapsed=time.perf_counter()-start;raw=''.join(chunks);text=raw;patch_error=None
   if action=='weak_fact_patch':text,patch_error=apply_patch_response(raw,initial['text'])
   obs=execute(t['db_path'],text)
   record={'task_id':t['id'],'dataset':t['dataset'],'group':t['group'],'split':t['split'],'action':action,'model':MODELS[name][0],'revision':MODELS[name][1],'temperature':temp,'seed':seed(t['id'],action),'text':text,'raw_text':raw,'prompt_tokens':r.prompt_tokens,'generation_tokens':r.generation_tokens,'service_seconds':elapsed,'feedback_seconds':feedback_seconds,'observation':obs,'finish_reason':r.finish_reason,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'logprob_mean':float(lp.mean()),'logprob_min':float(lp.min()),'logprob_q10':float(np.quantile(lp,.1)),'low_confidence_fraction':float((lp<-.7).mean()),'evidence_diagnostics':facts['diagnostics'] if facts else {},'patch_error':patch_error}
   with path.open('a') as f:f.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')
   records[(t['id'],action)]=record;new+=1
   if (idx+1)%16==0 or idx==len(pending)-1:print(json.dumps({'action':action,'done':idx+1,'total':len(pending),'new_calls':new,'elapsed':round(time.perf_counter()-started,1)}),flush=True)
 meta.update(completed_records=len(records),new_calls=new,elapsed_seconds=time.perf_counter()-started,warmed_models=warm)
 (out/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps({'completed':True,'new_calls':new,'records':len(records)}),flush=True)
if __name__=='__main__':main()
