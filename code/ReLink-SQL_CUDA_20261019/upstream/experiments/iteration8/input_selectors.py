"""Fixed input selectors; no labels, model loading, or edits to historic modules."""
from pathlib import Path
from types import FunctionType
import json,re,sqlite3
import sqlglot
from sqlglot import exp
from iteration3 import repair_index as ri, column_context as cc
HERE=Path(__file__).resolve().parent

def clone(fn,**replacement):
    g=dict(fn.__globals__);g.update(replacement)
    return FunctionType(fn.__code__,g,fn.__name__,fn.__defaults__,fn.__closure__)

def flat_structure(sql,tables):
    meta=dict(scopes=[],required=set(),unknown_tables=[],parse_error=None)
    T=dict(nodes=[dict(id='query',kind='query')],edges=[]);M=[]
    try:
        parsed=sqlglot.parse(sql,read='sqlite')
        if len(parsed)!=1 or not isinstance(parsed[0],exp.Query):raise ValueError('Expected one query')
        tree=parsed[0]
    except (ValueError,sqlglot.errors.SqlglotError) as e:
        meta['parse_error']=type(e).__name__;return T,M,meta
    aliases={};cte_names={ri._key(c.alias_or_name) for c in tree.find_all(exp.CTE)}
    for node in tree.find_all(exp.Table):
        table=ri._key(node.name);alias=ri._key(node.alias_or_name)
        if table in cte_names and table not in tables:continue
        exists=table in tables
        M.append(dict(kind='table',scope='flat',name=node.name,alias=node.alias_or_name,graph_nodes=[ri._tid(table)] if exists else [],status='bound' if exists else 'unresolved'))
        if exists:
            meta['required'].add(table);aliases.setdefault(alias,set()).add(table);aliases.setdefault(table,set()).add(table)
        else:meta['unknown_tables'].append(node.name)
    for node in tree.find_all(exp.Column):
        col=ri._key(node.name);qual=ri._key(node.table)
        candidates=aliases.get(qual,set()) if qual else meta['required']
        owners=sorted(t for t in candidates if col in tables[t]['columns'])
        # Flat parsing has no derived-column provenance. Preserve exact-name
        # candidates conservatively rather than dropping unresolved information.
        if not owners:owners=sorted(t for t in meta['required'] if col in tables[t]['columns'])
        M.append(dict(kind='column',scope='flat',name=node.name,qualifier=node.table,graph_nodes=[ri._cid(t,col) for t in owners],status='bound' if len(owners)==1 else 'ambiguous' if owners else 'unresolved'))
    meta['scopes']=[dict(id='flat',sources=[dict(alias=a,table=t,source_kind='table') for a,ts in aliases.items() for t in sorted(ts)])]
    return T,M,meta

def no_paths(tables,fks,seeds):return set(seeds),[]
NO_SCOPE=clone(cc.build_column_context,build_repair_context=clone(ri.build_repair_context,_structure=flat_structure))
NO_BRIDGE=clone(cc.build_column_context,build_repair_context=clone(ri.build_repair_context,_paths=no_paths))

def canonical_initial(w):
    # Historic compiler expects `text`; batch backend emits `raw_text`.
    # This explicit boundary prevents silently compiling an empty SQL string.
    text=w.get('raw_text',w.get('text'))
    if not isinstance(text,str):raise ValueError('Missing initial SQL')
    return dict(text=text,raw_text=text,observation=w['observation'])

def render(task,chosen,preserve_keys=False):
    index=ri._schema_index(task['db_path'])[0];tables=index['tables']
    selected={ri._key(t):{ri._key(c) for c in cols if ri._key(c) in tables[ri._key(t)]['columns']}
              for t,cols in chosen.items() if ri._key(t) in tables}
    selected={t:cols for t,cols in selected.items() if cols}
    if not selected:selected={t:set(m['columns']) for t,m in tables.items()}
    if preserve_keys:
        for t in selected:selected[t].update(c for c,m in tables[t]['columns'].items() if m['pk_position'])
        for e in index['fk_edges']:
            a,b=e['source_table'],e['target_table']
            if e['resolved'] and a in selected and b in selected:
                selected[a].update(ri._key(c) for c in e['source_columns']);selected[b].update(ri._key(c) for c in e['target_columns'])
    primary={t:[c for c,m in tables[t]['columns'].items() if m['pk_position']] for t in selected}
    definitions={};fallbacks={}
    for t,cols in sorted(selected.items()):
        m=tables[t]
        try:
            prune=cols!=set(m['columns']) and m['kind']=='table' and not any(x.get('hidden') for x in m['columns'].values())
            ddl,_,failure=cc._normalize_table(m['ddl'],t,cols,selected,primary,prune)
            if failure:raise ValueError(failure)
            definitions[t]=ddl
            if not prune:selected[t]=set(m['columns'])
        except (ValueError,sqlglot.errors.SqlglotError):
            definitions[t]=m['ddl'];selected[t]=set(m['columns']);fallbacks[t]='DDL normalization fallback'
    con=sqlite3.connect(':memory:')
    try:
        for ddl in definitions.values():con.execute(ddl)
    except sqlite3.Error:
        definitions={t:tables[t]['ddl'] for t in selected};selected={t:set(tables[t]['columns']) for t in selected};fallbacks['schema']='DDL validation fallback'
    finally:con.close()
    return dict(schema_text='\n\n'.join(definitions[t] for t in sorted(definitions)),selected_columns={tables[t]['name']:[m['name'] for c,m in tables[t]['columns'].items() if c in cols] for t,cols in sorted(selected.items())},fallbacks=fallbacks)

def full(task):
    return render(task,{t:list(m['columns']) for t,m in ri._schema_index(task['db_path'])[0]['tables'].items()})

def lexical(task):
    tables=ri._schema_index(task['db_path'])[0]['tables'];words=ri._words(task['question']+' '+task.get('evidence',''))
    scored=sorted(((3*len(ri._words(t)&words)+sum(bool(ri._words(c)&words) for c in m['columns']),t) for t,m in tables.items()),key=lambda v:(-v[0],v[1]))
    names=[t for s,t in scored if s>0][:3]
    if not names:return dict(full(task),selection_fallback='no lexical table match')
    chosen={}
    for t in names:
        cs=sorted(((len(ri._words(c)&words),c) for c in tables[t]['columns']),key=lambda v:(-v[0],v[1]))
        cols=[c for s,c in cs if s>0][:8]
        chosen[t]=cols or list(tables[t]['columns'])
    return render(task,chosen,True)

def mentioned(task,w):
    tables=ri._schema_index(task['db_path'])[0]['tables']
    try:
        sql=ri._input(canonical_initial(w))[0];tree=sqlglot.parse_one(sql,read='sqlite')
        selected={ri._key(n.name) for n in tree.find_all(exp.Table)}&set(tables)
    except (ValueError,AttributeError,sqlglot.errors.SqlglotError):selected=set()
    if not selected:return dict(full(task),selection_fallback='no known initial table')
    return render(task,{t:list(tables[t]['columns']) for t in selected})

def current(task,w,variant='V'):
    fn={'V':cc.build_column_context,'NO_SCOPE':NO_SCOPE,'NO_BRIDGE':NO_BRIDGE}[variant]
    try:
        result=fn(task,canonical_initial(w))
        # Keep complete trace as evidence that an ablation actually changed input.
        return result
    except sqlglot.errors.SqlglotError as e:return dict(full(task),selection_fallback='compiler exception '+type(e).__name__)

def deterministic(task,w):
    f=full(task)
    return dict(F=f,REPAIR=dict(f,append_user='\n\nPrevious SQL attempt (unverified):\n'+canonical_initial(w)['text']+'\nExecution error:\n'+str(w['observation'].get('error',''))),LEX=lexical(task),MENTION=mentioned(task,w),NO_SCOPE=current(task,w,'NO_SCOPE'),NO_BRIDGE=current(task,w,'NO_BRIDGE'),V=current(task,w))

def chess_messages(task,schema,stage):
    template=(HERE/'sources'/('template_select_'+stage+'.txt')).read_text()
    return [dict(role='user',content=template.format(DATABASE_SCHEMA=schema,QUESTION=task['question'],HINT=task.get('evidence','')))]

def parse_object(text):
    # Same JSON object interface; tolerate a Markdown fence, never repair by LLM.
    decoder=json.JSONDecoder()
    for m in re.finditer(r'\{',text):
        try:
            value,_=decoder.raw_decode(text[m.start():])
            if isinstance(value,dict):return value
        except json.JSONDecodeError:pass
    return None

def chess_selection(task,text,stage,previous):
    obj=parse_object(text);tables=ri._schema_index(task['db_path'])[0]['tables']
    allowed={ri._key(t):{ri._key(c) for c in cols} for t,cols in previous['selected_columns'].items()}
    chosen={};invalid=[]
    if obj is not None:
        if stage=='tables':
            raw=obj.get('table_names',[])
            if isinstance(raw,list):
                for name in raw:
                    if not isinstance(name,str):continue
                    t=ri._key(name.strip('`'))
                    if t in allowed:chosen[t]=list(allowed[t])
                    else:invalid.append(name)
        else:
            for name,cols in obj.items():
                if name=='chain_of_thought_reasoning':continue
                t=ri._key(name.strip('`'))
                if t not in allowed or not isinstance(cols,list):invalid.append(name);continue
                keep=[ri._key(c.strip('`')) for c in cols if isinstance(c,str) and ri._key(c.strip('`')) in allowed[t]]
                invalid.extend(t+'.'+str(c) for c in cols if not isinstance(c,str) or ri._key(c.strip('`')) not in allowed[t])
                if keep:chosen[t]=keep
    if not chosen:return dict(previous,selector_parse_ok=obj is not None,selector_fallback='no valid selection',invalid_names=invalid)
    return dict(render(task,chosen,preserve_keys=True),selector_parse_ok=True,selector_fallback=None,invalid_names=invalid)
