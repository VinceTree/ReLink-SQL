"""Select public held-out questions before accessing selected reference SQL."""
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import ast
import json
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iteration13.runner import HERE, read, save, prior
from iteration4.prepare_confirmation96 import source_data, question_key, norm, file_sha
from recoverylab.sql import connect, execute

DATA=HERE/'data/confirmation400'
SEED='20260915-feedback-confirmation-public-v1'


def audit_training():
    target=HERE/'results/cohort_capacity/share_training_audit.json'
    if target.exists():return read(target)
    questions=set();sources=[]
    for item in read(HERE/'sources/share_training_files.json'):
        rows=read(Path(item['path']));matched=failed=0;example_questions=set()
        for row in rows:
            # Only public input instructions; never inspect supervised outputs.
            for match in re.finditer(r'(?m)^\s*question\s*=\s*(.+)$',row['instruction']):
                try:q=ast.literal_eval(match.group(1).strip())
                except (ValueError,SyntaxError):failed+=1;continue
                if isinstance(q,str):example_questions.add(question_key(q));matched+=1
        questions.update(example_questions)
        sources.append(dict(**item,sha256=file_sha(item['path']),rows=len(rows),
            question_occurrences=matched,unique_input_question_hashes=len(example_questions),unparsed_question_lines=failed))
    result=dict(input_question_hashes=sorted(questions),sources=sources,
        scope='Conservative exact-normalized input-question exclusion, including prompt examples. BAM has SQL-only inputs; this is not a proof about all checkpoint training data.',
        author_training_config_sha256=file_sha(prior.EXP/'vendor/share/configs/data_config.json'))
    save(target,result);print('TRAINING INPUT AUDIT',[(s['role'],s['unique_input_question_hashes'],s['unparsed_question_lines']) for s in sources],flush=True)
    return result


def rank(value):return prior.batch.digest([SEED,value])


def audit_training_overlap():
    target=HERE/'results/cohort_capacity/share_training_overlap.json'
    if target.exists():return read(target)
    audit_training()
    sources,_,_=source_data()
    candidates={question_key(r['question']):norm(r['question']) for _,p,rows in sources if p.name=='dev.json' for r in rows}
    excluded=set();details=[]
    for item in read(HERE/'sources/share_training_files.json'):
        rows=read(Path(item['path']))
        # Full input-only scan also covers 1499 question lines that are not
        # valid Python string literals. No supervised output is inspected.
        text='\n'.join(norm(r['instruction'].replace('\\"','"').replace("\\'","'")) for r in rows)
        found={key for key,question in candidates.items() if question in text}
        excluded.update(found);details.append(dict(role=item['role'],matches=len(found),source_sha256=file_sha(item['path'])))
    result=dict(candidate_questions=len(candidates),excluded_question_hashes=sorted(excluded),roles=details,
                scope='Exact normalized candidate question substring in all published training INPUT instructions, including examples; supervised outputs ignored.')
    save(target,result);print('FULL TRAINING INPUT OVERLAP',len(excluded),flush=True)
    return result


def select():
    if (DATA/'selected.json').exists():return
    assert (HERE/'results/development/complete.json').exists(), 'Finish development before confirmation selection'
    assert (HERE/'CONFIRMATION.md').exists(), 'Record confirmation configuration first'
    training=audit_training();history=read(HERE/'results/cohort_capacity/history.json')
    overlap=audit_training_overlap()
    blocked=set(history['question_hashes'])|set(training['input_question_hashes'])|set(overlap['excluded_question_hashes'])
    sources,dbs,metadata=source_data();pools=defaultdict(list);seen=set();counts=Counter()
    for ds,path,rows in sources:
        if path.name!='dev.json':continue  # No official training split in confirmation.
        family='spider' if ds=='spider' else 'bird'
        for i,row in enumerate(rows):
            q=question_key(row['question']);db=row['db_id']
            if db not in dbs[family]:counts[family+'/missing_db']+=1;continue
            if q in blocked:counts[family+'/old_or_supervised_question']+=1;continue
            if q in seen:counts[family+'/duplicate_question']+=1;continue
            if family=='bird' and int(row['question_id']) in history['bird_ids']:
                counts[family+'/old_numeric_id']+=1;continue
            seen.add(q)
            tid=(f'spider13_confirm:{db}:{q[:12]}' if family=='spider' else f'bird13_confirm:{row["question_id"]}')
            task=dict(id=tid,dataset=family,group=db,split='confirmation',kind='sql',question=row['question'],
                      evidence=row.get('evidence',''),db_path=str(dbs[family][db].resolve()))
            pools[family,db].append(dict(task=task,source=str(path),source_row=i,question_key=q))
    chosen=[]
    for family in ('spider','bird'):
        groups=sorted([g for ds,g in pools if ds==family],key=lambda g:rank(family+':'+g))
        queues={g:sorted(pools[family,g],key=lambda r:rank(r['question_key'])) for g in groups}
        available=sum(map(len,queues.values()));counts[family+'/eligible']=available
        want=min(200,available);counts[family+'/selected']=want
        n=0;j=0
        while n<want:
            for g in groups:
                if j<len(queues[g]) and n<want:chosen.append(queues[g][j]);n+=1
            j+=1
    assert chosen and len(chosen)==len({r['question_key'] for r in chosen})
    schemas={};snapshots={}
    for row in chosen:
        task=row['task'];path=task['db_path']
        if path not in schemas:
            with connect(path) as con:
                ddl=[r[0] for r in con.execute("SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name") if r[0]]
            schemas[path]='\n\n'.join(x.rstrip(';')+';' for x in ddl)
            snapshots[path]=file_sha(path)
        task['schema']=schemas[path]
    save(DATA/'public.jsonl',[r['task'] for r in chosen])
    save(DATA/'selection.json',[dict(id=r['task']['id'],source=r['source'],source_row=r['source_row'],question_key=r['question_key']) for r in chosen])
    old_groups={tuple(x) for x in history['groups']}
    save(DATA/'selected.json',dict(seed=SEED,counts=dict(counts),question_count=len(chosen),
        public_sha256=file_sha(DATA/'public.jsonl'),selection_sha256=file_sha(DATA/'selection.json'),
        sources=metadata,snapshots=snapshots,code_sha256=file_sha(__file__),
        protocol_sha256=file_sha(HERE/'CONFIRMATION.md'),
        history_sha256=file_sha(HERE/'results/cohort_capacity/history.json'),
        training_audit_sha256=file_sha(HERE/'results/cohort_capacity/share_training_audit.json'),
        training_overlap_sha256=file_sha(HERE/'results/cohort_capacity/share_training_overlap.json'),
        unseen_database_questions=sum((r['task']['dataset'],norm(r['task']['group'])) not in old_groups for r in chosen),
        group_counts=dict(Counter(r['task']['dataset']+':'+r['task']['group'] for r in chosen)),
        reference_sql_used_for_selection=False,official_splits=['Spider dev','BIRD dev'],
        no_claim_of_database_disjoint_confirmation=True))
    print('SELECTED',dict(counts),flush=True)


def references():
    if (DATA/'private/complete.json').exists():return
    marker=read(DATA/'selected.json')
    assert file_sha(DATA/'public.jsonl')==marker['public_sha256']
    assert file_sha(DATA/'selection.json')==marker['selection_sha256']
    assert file_sha(HERE/'CONFIRMATION.md')==marker['protocol_sha256']
    sources,_,_=source_data();by_source={str(p):rows for _,p,rows in sources}
    for source in marker['sources']:assert file_sha(source['path'])==source['sha256']
    tasks={t['id']:t for t in read(DATA/'public.jsonl')}
    def one(row):
        import sqlglot
        task=tasks[row['id']];raw=by_source[row['source']][row['source_row']]
        assert question_key(raw['question'])==row['question_key']
        sql=raw['query'] if task['dataset']=='spider' else raw['SQL']
        observed=execute(task['db_path'],sql,timeout=10,max_rows=50000);observed['seconds']=0.0
        try:ordered=bool(sqlglot.parse_one(sql,read='sqlite').args.get('order'))
        except sqlglot.errors.SqlglotError:ordered='order by' in sql.lower()
        return row['id'],dict(gold_sql=sql,gold_result=observed,order_matters=ordered)
    with ThreadPoolExecutor(max_workers=4) as pool:refs=dict(pool.map(one,read(DATA/'selection.json')))
    save(DATA/'private/references.json',refs)
    save(DATA/'private/complete.json',dict(reference_sha256=file_sha(DATA/'private/references.json'),
        n=len(refs),unscorable=[tid for tid,r in refs.items() if r['gold_result']['status']!='ok'],
        no_replacement=True,selection_sha256=marker['selection_sha256']))
    print('REFERENCES READY',len(refs),'unscorable',sum(r['gold_result']['status']!='ok' for r in refs.values()),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['audit_training','audit_training_overlap','select','references'])
    globals()[parser.parse_args().phase]()
