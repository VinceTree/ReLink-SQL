"""Public-only cohort selection and fresh initial generation requests."""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib,json,sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from iteration4.prepare_confirmation96 import scan_history,source_data,family,norm,question_key,file_sha
from iteration4.collect_confirmation import initial_messages,MODELS
from iteration4 import batch_public_generate as batch
from recoverylab.sql import connect
HERE=Path(__file__).resolve().parent; EXP=HERE.parent
DATA=HERE/'data/main1000'; OUT=HERE/'results/main1000'
SEED='20260914-schema-main-v1'
def rank(s):return hashlib.sha256((SEED+'\0'+s).encode()).hexdigest()
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj,lines=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    text=(''.join(json.dumps(x,ensure_ascii=False,default=str)+'\n' for x in obj) if lines else json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')
    if p.exists() and p.read_text()!=text:raise ValueError('Immutable artifact '+str(p))
    p.write_text(text)
def select():
    assert not DATA.exists()
    h=scan_history(HERE);sources,dbs,meta=source_data();names={g for _,g in h['groups']}
    used=[p for paths in dbs.values() for db,p in paths.items() if norm(db) in names]
    sizes=defaultdict(list)
    for p in used:sizes[p.stat().st_size].append(p)
    hashes={};duplicates={};pools=defaultdict(list);seen=set()
    def hashed(p):
        if str(p) not in hashes:hashes[str(p)]=file_sha(p)
        return hashes[str(p)]
    for ds,path,rows in sources:
        f=family(ds)
        for i,r in enumerate(rows):
            db=r['db_id'];q=question_key(r['question'])
            if db not in dbs[f] or q in h['questions'] or q in seen:continue
            if f=='bird' and int(r['question_id']) in h['bird_ids']:continue
            if f=='spider':
                if norm(db) in names:continue
                if db not in duplicates:
                    p=dbs[f][db];duplicates[db]=[str(old) for old in sizes[p.stat().st_size] if hashed(old)==hashed(p)]
                if duplicates[db]:continue
            seen.add(q)
            ident='spider8_main:'+db+':'+q[:12] if f=='spider' else 'bird8_main:'+str(r['question_id'])
            t=dict(id=ident,dataset=f,group=db,split='confirmation',kind='sql',question=r['question'],evidence=r.get('evidence',''),db_path=str(dbs[f][db].resolve()))
            pools[f,db].append(dict(task=t,source=str(path),source_row=i,question_key=q))
    selected=[];selected_groups={}
    for f in ['spider','bird']:
        groups=sorted([db for fam,db in pools if fam==f and (f!='spider' or len(pools[f,db])>=15)],key=lambda db:rank(f+'\0'+db))
        if f=='spider':
            if len(groups)<24:raise ValueError('Not enough unseen Spider databases')
            groups=groups[:24]
        if sum(len(pools[f,db]) for db in groups)<500:raise ValueError('Insufficient new questions for '+f)
        selected_groups[f]=groups
        queues={db:sorted(pools[f,db],key=lambda r:rank(f+'\0'+db+'\0'+r['question_key'])) for db in groups}
        n=0;j=0
        while n<500:
            for db in groups:
                if j<len(queues[db]) and n<500:selected.append(queues[db][j]);n+=1
            j+=1
    assert len(selected)==len({r['question_key'] for r in selected})==1000
    snaps={}
    for r in selected:
        t=r['task'];p=Path(t['db_path'])
        if str(p) not in snaps:
            with connect(p) as con:ddl=[x[0] for x in con.execute("SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name") if x[0]]
            snaps[str(p)]=dict(sha256=hashed(p),bytes=p.stat().st_size,schema='\n\n'.join(x.rstrip(';')+';' for x in ddl))
        t['schema']=snaps[str(p)]['schema']
    write(DATA/'public.jsonl',[r['task'] for r in selected],True)
    write(DATA/'selection.json',[{k:v for k,v in r.items() if k!='task'}|dict(id=r['task']['id']) for r in selected])
    write(DATA/'freeze.json',dict(seed=SEED,count=1000,public_sha256=sha(DATA/'public.jsonl'),selection_sha256=sha(DATA/'selection.json'),protocol_sha256=sha(HERE/'PROTOCOL.md'),cohort_code_sha256=sha(__file__),sources=meta,selected_groups=selected_groups,groups=dict(Counter(r['task']['dataset']+':'+r['task']['group'] for r in selected)),historical_files=h['files'],historical_database_names=sorted(names),historical_question_hashes=sorted(h['questions']),spider_duplicate_checks=duplicates,snapshots={p:{k:v for k,v in m.items() if k!='schema'} for p,m in snaps.items()},references_used_for_selection=False))
    print(json.dumps(dict(count=1000,databases={f:len(g) for f,g in selected_groups.items()},spider_unseen=True,bird_unseen=False)))
def tasks():
    f=json.loads((DATA/'freeze.json').read_text())
    assert sha(DATA/'public.jsonl')==f['public_sha256']
    assert sha(HERE/'PROTOCOL.md')==f['protocol_sha256']
    return read(DATA/'public.jsonl')
def weak():
    req=[dict(id=t['id']+'|W',messages=initial_messages(t),model=MODELS['weak'][0],revision=MODELS['weak'][1],max_tokens=1024) for t in tasks()]
    write(OUT/'weak_requests.jsonl',req,True)
    print(batch.prepare(OUT/'weak_requests.jsonl',OUT/'weak_batch'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['select','weak']);a=p.parse_args();globals()[a.phase]()
