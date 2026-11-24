"""Choose one observable SQL rewrite using only previously rejected results."""
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope
from . import compile_repair as local, contract

r=contract.r
VERSION='observable_relational_repair_v4'


def count_alternative(task,sql,plan):
    tree=contract.repair.query(sql)
    if not isinstance(tree,exp.Select):return None
    counts=[n for n in tree.find_all(exp.Count)
            if n.find_ancestor(exp.Select) is tree and n.find_ancestor(exp.Window) is None]
    if len(counts)!=1:return None
    node=counts[0];value=node.this
    if isinstance(value,exp.Distinct):
        if len(value.expressions)!=1:return None
        node.set('this',value.expressions[0].copy())
    elif isinstance(value,exp.Column) and not value.is_star:
        node.set('this',exp.Distinct(expressions=[value.copy()]))
    elif isinstance(value,exp.Star):
        # COUNT(*) does not identify an entity. Only use one declared primary key
        # from a uniquely bound table explicitly mentioned in proposed outputs.
        tables=r.methods.selectors.ri._schema_index(task['db_path'])[0]['tables']
        outputs={t for t,c in plan['outputs']}
        sources=list(traverse_scope(tree))[-1].sources
        bindings=[(a,n.name.casefold()) for a,n in sources.items()
                  if isinstance(n,exp.Table) and n.name.casefold() in outputs]
        if len(bindings)!=1:return None
        alias,t=bindings[0]
        keys=[c['name'] for c in tables[t]['columns'].values() if c['pk_position']]
        if len(keys)!=1:return None
        node.set('this',exp.Distinct(expressions=[exp.column(keys[0],table=alias,quoted=True)]))
    else:return None
    return tree.sql(dialect='sqlite')


def propose(task,history,plan_text,policy='multiplicity'):
    if policy!='multiplicity':raise ValueError(policy)
    plan=contract.parse(plan_text,task);anchor,_=contract.align(task,history,plan)
    sql=history[anchor['index']]['sql']
    metadata=dict(version=VERSION,policy=policy,plan=plan,anchor=anchor,
                  operations=[],probes=[],search=[])
    rejected={contract.repair.result_key(x['observation']) for x in history}-{None}
    seen=set()
    # The priority is fixed before this experiment. At most three alternatives
    # are executed; only the first new complete result can be delivered.
    for operation in ('result_multiplicity','count_multiplicity','bound_requirements'):
        try:
            if operation=='result_multiplicity':
                candidate,detail=local.propose(task,history,plan_text,'toggle')
            elif operation=='count_multiplicity':
                candidate=count_alternative(task,sql,plan);detail={}
            else:
                candidate,detail=local.propose(task,history,plan_text,'requirements')
        except (ValueError,TypeError,IndexError,KeyError):
            candidate=None;detail={}
        metadata['probes']+=detail.get('probes',[])
        if candidate is None or candidate in seen:
            metadata['search'].append(dict(operation=operation,status='unavailable_or_identical_sql'))
            continue
        seen.add(candidate)
        observation=r.execute(task['db_path'],candidate,timeout=10,max_rows=50000)
        metadata['probes'].append(dict(sql=candidate,observation=r.stable(observation)))
        key=contract.repair.result_key(observation)
        status='incomplete' if key is None else 'known_rejected_result' if key in rejected else 'new_result'
        metadata['search'].append(dict(operation=operation,status=status))
        if status=='new_result':
            metadata['operations']=[operation]
            metadata['local_operations']=detail.get('operations',[])
            return candidate,metadata
    return None,metadata
