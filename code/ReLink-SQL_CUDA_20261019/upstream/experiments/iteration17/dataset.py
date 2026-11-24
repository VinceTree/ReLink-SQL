"""Prepare every official dev row; keep reference execution behind evaluation."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iteration13 import runner as r
from iteration4.prepare_confirmation96 import source_data, question_key
from recoverylab.sql import connect

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'


def verify():
    marker = r.read(DATA/'prepared.json')
    for path, digest in marker['files'].items():
        assert r.prior.batch.sha(path) == digest, path
    return marker


def prepare():
    if (DATA/'prepared.json').exists():
        return verify()
    sources, databases, _ = source_data()
    sources = [(('spider' if ds=='spider' else 'bird'), path, rows)
               for ds, path, rows in sources if path.name=='dev.json']
    old = {}
    history_files = [r.prior.OUT/'public.jsonl', r.HERE/'data/confirmation400/public.jsonl']
    for path in history_files:
        for task in r.read(path):
            key = (task['dataset'], task['group'], task['question'], task.get('evidence',''))
            old[key] = task
    tasks, selection, snapshots, schemas, used = [], [], {}, {}, set()
    for ds, source, rows in sources:
        for index, row in enumerate(rows):
            db = row['db_id']; db_path = str(databases[ds][db].resolve())
            key = (ds, db, row['question'], row.get('evidence',''))
            prior = old.get(key)
            tid = prior['id'] if prior else f'{ds}17_dev:{index}:{question_key(row["question"])[:12]}'
            if tid in used:
                tid = f'{ds}17_dev:{index}:{question_key(row["question"])[:12]}'
            used.add(tid)
            if db_path not in schemas:
                with connect(db_path) as connection:
                    ddl = [x[0] for x in connection.execute(
                        "SELECT sql FROM sqlite_master WHERE type IN ('table','view') "
                        "AND name NOT LIKE 'sqlite_%' ORDER BY name") if x[0]]
                schemas[db_path] = '\n\n'.join(x.rstrip(';')+';' for x in ddl)
                snapshots[db_path] = r.prior.batch.sha(db_path)
            task = dict(id=tid,dataset=ds,group=db,split='full_dev',kind='sql',
                        question=row['question'],evidence=row.get('evidence',''),
                        db_path=db_path,schema=schemas[db_path])
            assert set(task)==r.methods.TASK_FIELDS
            tasks.append(task)
            selection.append(dict(id=tid,dataset=ds,source=str(source),source_row=index,
                                  question_key=question_key(row['question']),
                                  overlaps_recent_development=prior is not None))
    assert len(tasks)==len(used)==2568
    assert Counter(t['dataset'] for t in tasks)=={'spider':1034,'bird':1534}
    r.save(DATA/'public.jsonl',tasks)
    r.save(DATA/'selection.json',selection)
    r.save(DATA/'full_contexts.json',{t['id']:r.contexts.old.full(t) for t in tasks})
    tables={}
    for t in tasks:
        key=t['dataset']+':'+t['group']
        if key not in tables:
            tables[key]=r.methods.share.table_info_from_sqlite(dict(t,db_id=t['group']))
    r.save(DATA/'share_tables.json',tables)
    files=[Path(__file__),HERE/'PROTOCOL.md',*history_files,
           *[p for _,p,_ in sources],*[DATA/n for n in
             ('public.jsonl','selection.json','full_contexts.json','share_tables.json')]]
    marker=dict(n=len(tasks),counts=dict(Counter(t['dataset'] for t in tasks)),
                database_counts={ds:len({t['group'] for t in tasks if t['dataset']==ds})
                                 for ds in ('spider','bird')},
                no_outcome_selection=True,independent_test=False,
                files={**{str(p):r.prior.batch.sha(p) for p in files},**snapshots})
    r.save(DATA/'prepared.json',marker)
    return marker


def references():
    verify()
    dest=DATA/'private'
    if (dest/'complete.json').exists():
        assert r.prior.batch.sha(dest/'references.json')==r.read(dest/'complete.json')['sha256']
        return
    tasks={t['id']:t for t in r.read(DATA/'public.jsonl')}
    sources={str(p):rows for _,p,rows in source_data()[0]}
    def one(item):
        import sqlglot
        task=tasks[item['id']];row=sources[item['source']][item['source_row']]
        assert question_key(row['question'])==item['question_key']
        sql=row['query'] if task['dataset']=='spider' else row['SQL']
        observed=r.stable(r.execute(task['db_path'],sql,timeout=10,max_rows=50000))
        try:ordered=bool(sqlglot.parse_one(sql,read='sqlite').args.get('order'))
        except sqlglot.errors.SqlglotError:ordered='order by' in sql.lower()
        return item['id'],dict(gold_sql=sql,gold_result=observed,order_matters=ordered)
    pairs=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for pair in pool.map(one,r.read(DATA/'selection.json')):
            pairs.append(pair)
            if len(pairs)%128==0:print('REFERENCE EXECUTION',len(pairs),'/ 2568',flush=True)
    values=dict(pairs)
    r.save(dest/'references.json',values)
    failures={tid:v['gold_result'].get('error','unknown') for tid,v in values.items()
              if v['gold_result']['status']!='ok'}
    r.save(dest/'complete.json',dict(n=len(values),sha256=r.prior.batch.sha(dest/'references.json'),
                                   unscorable=failures,no_replacement=True))
    print('REFERENCE COMPLETE',len(values),'unscorable',len(failures),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','references'])
    args=parser.parse_args()
    if args.phase=='prepare':print(prepare()['counts'])
    else:references()
