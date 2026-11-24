"""Infer question requirements independently, then align rejected SQL states."""
import re
from functools import lru_cache
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope
from . import repair

r, common = repair.r, repair.common
VERSION='question_contract_alignment_v2'
PLAN_CAP=384
AGGREGATES={'count','sum','avg','min','max'}


def plan_messages(task,state):
    r.methods.validate(task,state)
    prompt=(
        'Extract the intended answer requirements from this question and SQLite catalog. '
        'Do not generate SQL or diagnose an earlier answer. Use only requirements supported by the question/evidence. '
        'Return one compact JSON object, with no examples or extra text:\n'
        'meaning: one short sentence specifying the answer, requested output format, population and aggregation unit; '
        'outputs: array of [exact table name, exact column name] pairs whose values are returned or aggregated into the answer; '
        'filters: array of catalog pairs used only to restrict the population, not returned as output; '
        'aggregates: operations needed anywhere in the query, drawn from count, sum, avg, min, max; '
        'grain: single, grouped, rows, or unknown; '
        'group_by: array of catalog field pairs; '
        'distinct: whether final result rows must be deduplicated, yes, no, or unknown; '
        'order: final result ordering, asc, desc, none, or unknown; '
        'limit: an explicitly required positive integer, otherwise null; '
        'inspect: at most two objects with field (a catalog pair) and target '
        '(a short verbatim string from the question/evidence whose stored representation may be uncertain). '
        'Use empty arrays or unknown when uncertain. Keep the entire object within 280 tokens.\n'
        'Question: '+task['question']+'\nEvidence: '+task.get('evidence','')+
        '\nCatalog:\n'+common.catalog(task['db_path']))
    return [dict(role='user',content=prompt)],dict(version=VERSION,old_sql_visible=False)


def parse(text,task):
    raw=r.methods.selectors.parse_object(text)
    errors=[]
    if not isinstance(raw,dict):
        raw={};errors.append('invalid JSON')
    tables=r.methods.selectors.ri._schema_index(task['db_path'])[0]['tables']
    def field(value):
        if not isinstance(value,list) or len(value)!=2 or not all(isinstance(x,str) for x in value):
            errors.append('invalid field pair');return None
        t,c=map(str.casefold,value)
        if t not in tables or c not in tables[t]['columns']:
            errors.append('unbound field: '+str(value));return None
        return [t,c]
    def fields(value):
        if not isinstance(value,list):
            errors.append('invalid field list');return []
        if len(value)>16:
            errors.append('more than 16 binding pairs; retained first 16')
        return [p for v in value[:16] if (p:=field(v)) is not None]
    result=dict(meaning=str(raw.get('meaning',''))[:500],outputs=fields(raw.get('outputs',[])),
        filters=fields(raw.get('filters',[])),group_by=fields(raw.get('group_by',[])))
    result['fields']=[list(x) for x in sorted({tuple(x) for x in result['outputs']+result['filters']})]
    aggregate=raw.get('aggregates',[])
    result['aggregates']=[x for x in aggregate if x in AGGREGATES] if isinstance(aggregate,list) and all(isinstance(x,str) for x in aggregate) else []
    result['aggregates_specified']='aggregates' in raw and isinstance(aggregate,list) and all(isinstance(x,str) and x in AGGREGATES for x in aggregate)
    if 'aggregates' in raw and not result['aggregates_specified']:
        errors.append('invalid aggregate operations')
    for key,valid in [('grain',{'single','grouped','rows','unknown'}),('distinct',{'yes','no','unknown'}),
                      ('order',{'asc','desc','none','unknown'})]:
        value=raw.get(key)
        if isinstance(value,str):value=value.casefold().strip()
        result[key]=value if isinstance(value,str) and value in valid else 'unknown'
        if key in raw and (not isinstance(value,str) or value not in valid):
            errors.append('invalid '+key)
    limit=raw.get('limit');result['limit']=limit if type(limit) is int and 0<limit<=10000 else None
    if limit is not None and result['limit'] is None:errors.append('invalid limit')
    inspect=raw.get('inspect',[]);result['inspect']=[]
    if isinstance(inspect,list):
        for item in inspect[:2]:
            if not isinstance(item,dict):
                errors.append('invalid inspection');continue
            bound=field(item.get('field'));target=item.get('target','')
            supported=isinstance(target,str) and target.casefold() in (task['question']+' '+task.get('evidence','')).casefold()
            if bound and supported and 0<len(target)<=100:
                result['inspect'].append(dict(field=bound,target=target))
            elif bound:
                errors.append('invalid inspection target or absent from question/evidence')
    else:errors.append('invalid inspection list')
    result.update(valid=not errors and bool(result['meaning']),invalid=errors)
    return result


def profile(task,sql):
    tables=r.methods.selectors.ri._schema_index(task['db_path'])[0]['tables']
    try:
        tree=repair.query(sql)
        scopes=list(traverse_scope(tree))
    except Exception as error:
        return dict(parsed=False,fields=[],output_fields=[],unresolved=[type(error).__name__],aggregates=[],grain='unknown',
                    group_by=[],distinct='unknown',order='unknown',limit=None)
    all_fields=set();output_fields=set();groups=set();unresolved=[]
    for scope in scopes:
        sources={a.casefold():n.name.casefold() for a,n in scope.sources.items()
                 if isinstance(n,exp.Table) and n.name.casefold() in tables}
        for col in scope.columns:
            if col.find_ancestor(exp.Select) is not scope.expression:
                continue
            owners=([sources[col.table.casefold()]] if col.table.casefold() in sources else
                    [] if col.table else list(dict.fromkeys(sources.values())))
            owners=[t for t in owners if col.name.casefold() in tables[t]['columns']]
            if len(owners)!=1:
                unresolved.append(col.sql());continue
            bound=(owners[0],col.name.casefold());all_fields.add(bound)
            if scope is scopes[-1]:
                node=col
                while node.parent is not None and node.parent is not scope.expression:
                    node=node.parent
                if any(node is item for item in scope.expression.expressions):output_fields.add(bound)
            if scope is scopes[-1] and col.find_ancestor(exp.Group) is not None:
                groups.add(bound)
    outer=scopes[-1].expression if scopes else tree
    if scopes:
        sources={a.casefold():n.name.casefold() for a,n in scopes[-1].sources.items()
                 if isinstance(n,exp.Table) and n.name.casefold() in tables}
        for item in outer.expressions:
            if item.is_star:
                qualifier=item.table.casefold() if isinstance(item,exp.Column) else ''
                owners=[sources[qualifier]] if qualifier in sources else [] if qualifier else list(sources.values())
                output_fields.update((owner,col) for owner in owners for col in tables[owner]['columns'])
        all_fields.update(output_fields)
    outer_aggregates={n.key for n in outer.find_all(exp.AggFunc)
                if n.find_ancestor(exp.Select) is outer and n.find_ancestor(exp.Window) is None and n.key in AGGREGATES}
    aggregates={n.key for n in tree.find_all(exp.AggFunc)
                if n.find_ancestor(exp.Window) is None and n.key in AGGREGATES}
    grouped=bool(outer.args.get('group'))
    order=outer.args.get('order');limit=outer.args.get('limit')
    descending=order.expressions[0].args.get('desc') if order and order.expressions else None
    limit_value=limit.expression.name if limit and isinstance(limit.expression,exp.Literal) else ''
    return dict(parsed=True,fields=sorted(all_fields),output_fields=sorted(output_fields),unresolved=unresolved[:8],aggregates=sorted(aggregates),
        grain='grouped' if grouped else 'single' if outer_aggregates else 'rows',group_by=sorted(groups),
        distinct='yes' if outer.args.get('distinct') else 'no',
        order='none' if order is None else 'desc' if descending else 'asc',
        limit=int(limit_value) if limit_value.isdigit() else None)


def align(task,history,plan):
    requirements={tuple(x) for x in plan['fields']+plan['group_by']}
    candidates=[]
    for index,event in enumerate(history):
        observed=profile(task,event['sql'])
        missing=sorted(requirements-{tuple(x) for x in observed['fields']})
        outputs={tuple(x) for x in plan['outputs']}
        extra=sorted({tuple(x) for x in observed['output_fields']}-outputs) if outputs else []
        mismatch={k:dict(proposed=plan[k],observed=observed[k]) for k in ('grain','distinct','order')
                  if plan[k]!='unknown' and plan[k]!=observed[k]}
        if plan['aggregates_specified'] and set(plan['aggregates'])!=set(observed['aggregates']):
            mismatch['aggregates']=dict(proposed=plan['aggregates'],observed=observed['aggregates'])
        if plan['limit'] is not None and plan['limit']!=observed['limit']:
            mismatch['limit']=dict(proposed=plan['limit'],observed=observed['limit'])
        invalid=not observed['parsed'] or event['observation'].get('status')!='ok'
        rank=(int(invalid),len(missing),len(extra),len(mismatch),int(index==0),index)
        candidates.append(dict(index=index,event=event['event'],rank=list(rank),
            missing_fields=missing,extra_output_fields=extra,mismatches=mismatch,profile=observed))
    selected=min(candidates,key=lambda x:tuple(x['rank']))
    return selected,candidates


@lru_cache(maxsize=512)
def inspect_value(db_path,table,column,target):
    tables=r.methods.selectors.ri._schema_index(db_path)[0]['tables']
    t=common.quote(tables[table]['name'])
    c=common.quote(tables[table]['columns'][column]['name'])
    words=sorted(set(re.findall(r'\w+',target)),key=lambda x:(-len(x),x))
    terms=list(dict.fromkeys([target]+[x for x in words if len(x)>=3][:2]))
    tests=['LOWER(CAST('+c+' AS TEXT)) LIKE LOWER('+exp.Literal.string('%'+v+'%').sql(dialect='sqlite')+')' for v in terms]
    sql='SELECT DISTINCT '+c+' FROM '+t+' WHERE '+' OR '.join(tests)+' ORDER BY LENGTH(CAST('+c+' AS TEXT)), CAST('+c+' AS TEXT) LIMIT 5'
    observation=r.execute(db_path,sql,timeout=3,max_rows=5)
    return dict(sql=sql,observation=r.stable(observation))


def answer_messages(task,state,history,plan_text,anchor_policy='aligned'):
    r.methods.validate(task,state)
    plan=parse(plan_text,task);selected,candidates=align(task,history,plan)
    if anchor_policy=='latest':selected=candidates[-1]
    elif anchor_policy!='aligned':raise ValueError(anchor_policy)
    anchor=history[selected['index']];values=[];probes=[]
    for item in plan['inspect']:
        t,c=item['field'];probe=inspect_value(task['db_path'],t,c,item['target']);probes.append(probe)
        values.append(dict(field=item['field'],target=item['target'],
            observed=common.observation(probe['observation']),sample_is_incomplete=True))
    # Bound only presentation, never the complete execution record used by the evaluator.
    report=dict(missing_proposed_fields=selected['missing_fields'],extra_proposed_output_fields=selected['extra_output_fields'],
                operator_differences=selected['mismatches'],
                selected_result=repair.short_report(dict(latest_result=common.observation(anchor['observation']),
                                                         tried_changes=[]))['latest_result'])
    prompt=(
        'Generate one SQLite SELECT query answering the original question. '
        'A separate reading of the question proposed the requirements below; they may be mistaken. '
        'The program selected a previously rejected SQL by comparing catalog bindings and operators. '
        'Use its justified parts, reconcile the proposed requirements with the question, and correct omissions. '
        'Observed database values are incomplete samples, never a reference answer. Return only the complete SQL.\n'
        'Question: '+task['question']+'\nEvidence: '+task.get('evidence','')+
        '\nCatalog:\n'+common.catalog(task['db_path'])+
        '\nProposed requirements:\n'+common.compact({k:v for k,v in plan.items() if k not in ('invalid','inspect','fields','aggregates_specified')})+
        '\nSelected rejected SQL:\n'+anchor['sql']+
        '\nProgram comparison:\n'+common.compact(report)+
        '\nValue checks:\n'+common.compact(values))
    metadata=dict(version=VERSION,anchor_policy=anchor_policy,plan=plan,selected=selected,candidates=candidates,probes=probes,
                  history_events=[x['event'] for x in history],critic_in_candidate=False)
    return [dict(role='user',content=prompt)],metadata
