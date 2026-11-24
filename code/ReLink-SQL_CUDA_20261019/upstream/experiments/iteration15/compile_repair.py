"""Compile bounded SQL changes from independently predicted requirements."""
from collections import defaultdict
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope
from . import contract

r,common=contract.r,contract.common
VERSION='bounded_requirement_compiler_v3'


def affinity(text):
    text=text.upper()
    if any(x in text for x in ('INT','REAL','FLOA','DOUB','NUM','DEC')):return 'number'
    if any(x in text for x in ('CHAR','CLOB','TEXT')):return 'text'
    return text


def propose(task,history,plan_text,policy='requirements'):
    plan=contract.parse(plan_text,task)
    chosen,_=contract.align(task,history,plan)
    original=history[chosen['index']]['sql']
    meta=dict(version=VERSION,policy=policy,plan=plan,anchor=chosen,operations=[],probes=[])
    try:
        tree=contract.repair.query(original)
        if not isinstance(tree,exp.Select):return None,meta
        scope=list(traverse_scope(tree))[-1]
    except Exception:
        return None,meta
    if policy=='toggle':
        tree.set('distinct',None if tree.args.get('distinct') else exp.Distinct())
        meta['operations'].append('toggle_distinct')
        return tree.sql(dialect='sqlite'),meta
    if policy!='requirements':raise ValueError(policy)
    tables=r.methods.selectors.ri._schema_index(task['db_path'])[0]['tables']
    sources={a:n.name.casefold() for a,n in scope.sources.items()
             if isinstance(n,exp.Table) and n.name.casefold() in tables}
    aliases=defaultdict(list)
    for alias,table in sources.items():aliases[table].append(alias)
    by_lower={a.casefold():a for a in sources}
    def binding(column):
        if column.table:
            choices=[by_lower[column.table.casefold()]] if column.table.casefold() in by_lower else []
        else:choices=list(sources)
        choices=[a for a in choices if column.name.casefold() in tables[sources[a]]['columns']]
        return (sources[choices[0]],column.name.casefold()) if len(choices)==1 else None
    def column(pair):
        t,c=pair
        if len(aliases[t])!=1:return None
        return exp.column(tables[t]['columns'][c]['name'],table=aliases[t][0],quoted=True)
    simple=not plan['aggregates'] and not tree.args.get('group') and not tree.args.get('having')
    plain=all(isinstance(x,exp.Column) or isinstance(x,exp.Star) for x in tree.expressions)
    desired=list(dict.fromkeys(tuple(x) for x in plan['outputs']))
    # An unresolved requested output can be a derived measurement; do not drop it.
    safe_outputs=not any('unbound field' in e or 'binding pairs' in e for e in plan['invalid'])
    if simple and plain and desired and safe_outputs:
        actual=[];resolved=True
        for item in tree.expressions:
            if item.is_star:
                qualifier=item.table.casefold() if isinstance(item,exp.Column) else ''
                if qualifier:
                    owners=[by_lower[qualifier]] if qualifier in by_lower else []
                else:owners=list(sources) if len(sources)==len(scope.sources) else []
                if not owners:resolved=False
                for owner in owners:actual.extend((sources[owner],c) for c in tables[sources[owner]]['columns'])
            else:
                found=binding(item)
                if found is None:resolved=False
                else:actual.append(found)
        proposed=[column(pair) for pair in desired]
        if resolved and all(x is not None for x in proposed) and actual!=desired:
            tree.set('expressions',proposed);meta['operations'].append('project_requested_fields')
    if plan['distinct'] in ('yes','no'):
        wanted=plan['distinct']=='yes'
        if wanted!=bool(tree.args.get('distinct')):
            tree.set('distinct',exp.Distinct() if wanted else None)
            meta['operations'].append('set_result_distinct')
    def predicates():
        found=[]
        for col in tree.find_all(exp.Column):
            if col.find_ancestor(exp.Select) is not tree or col.find_ancestor(exp.Where) is None:
                continue
            parent=col.parent
            if parent.find(exp.Select) is not None:continue
            if isinstance(parent,(exp.EQ,exp.NEQ,exp.GT,exp.GTE,exp.LT,exp.LTE,exp.Like,exp.In,exp.Between)) and any(parent.find_all(exp.Literal)):
                pair=binding(col)
                if pair:found.append((col,pair))
        return found
    current=predicates();present={pair for _,pair in current}
    expected={tuple(x) for x in plan['filters']}
    for table in sorted({t for t,c in present|expected}):
        old=[x for x in present-expected if x[0]==table]
        new=[x for x in expected-present if x[0]==table]
        if len(old)!=1 or len(new)!=1:continue
        before,after=old[0],new[0]
        kind=affinity(tables[table]['columns'][before[1]]['type'])
        if not kind or kind!=affinity(tables[table]['columns'][after[1]]['type']):continue
        for col,pair in current:
            if pair==before:col.set('this',exp.to_identifier(tables[table]['columns'][after[1]]['name'],quoted=True))
        meta['operations'].append(dict(rebind_filter=[before,after]))
    raw=r.methods.selectors.parse_object(plan_text)
    requested={tuple(x['field']) for x in plan['inspect']}
    if isinstance(raw,dict) and isinstance(raw.get('inspect'),list):
        for pair in raw['inspect'][:2]:
            if isinstance(pair,list) and len(pair)==2 and all(isinstance(x,str) for x in pair):
                t,c=map(str.casefold,pair)
                if t in tables and c in tables[t]['columns']:requested.add((t,c))
    checked=set();question=(task['question']+' '+task.get('evidence','')).casefold()
    for col,pair in predicates():
        if len(checked)>=2:break
        parent=col.parent
        if pair not in requested or pair in checked or not isinstance(parent,exp.EQ):continue
        literal=parent.expression if parent.this is col else parent.this
        if not isinstance(literal,exp.Literal) or not literal.is_string:continue
        text=literal.this
        if len(text)<2 or text.casefold() not in question:continue
        checked.add(pair);t,c=pair
        probe=contract.inspect_value(task['db_path'],t,c,text);meta['probes'].append(probe)
        obs=probe['observation']
        if obs.get('status')!='ok' or len(obs.get('rows',[]))!=1:continue
        value=obs['rows'][0][0]
        if not isinstance(value,str) or value==text:continue
        qt,qc=common.quote(tables[t]['name']),common.quote(tables[t]['columns'][c]['name'])
        check='SELECT 1 FROM '+qt+' WHERE '+qc+' = '+literal.sql(dialect='sqlite')+' LIMIT 1'
        existing=r.execute(task['db_path'],check,timeout=3,max_rows=1)
        meta['probes'].append(dict(sql=check,observation=r.stable(existing)))
        if existing.get('status')=='ok' and not existing['rows']:
            literal.replace(exp.Literal.string(value));meta['operations'].append(dict(ground_value=list(pair)))
    if (plan['grain']=='single' and set(plan['aggregates'])=={'count'} and not plan['group_by']
            and tree.args.get('group') and not tree.args.get('limit') and not tree.args.get('offset')):
        tree=exp.select(exp.Count(this=exp.Star())).from_(tree.subquery('required_groups'))
        meta['operations'].append('count_groups')
    return (tree.sql(dialect='sqlite') if meta['operations'] else None),meta
