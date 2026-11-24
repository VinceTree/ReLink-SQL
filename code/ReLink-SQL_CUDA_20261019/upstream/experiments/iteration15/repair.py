"""Compile rejected execution histories and operator effects into repair input."""
from functools import lru_cache
import json

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from iteration13 import runner as r
from iteration14 import demand as common

VERSION = 'observed_effect_repair_v1'
CLAUSES = ('expressions', 'from_', 'joins', 'where', 'group', 'having', 'order', 'limit', 'distinct')


def query(sql):
    parsed = sqlglot.parse(sql, read='sqlite')
    if len(parsed)!=1 or not isinstance(parsed[0], exp.Query):
        raise ValueError('Expected one SELECT query')
    return parsed[0]


def result_key(observation):
    if observation.get('status')!='ok' or 'rows' not in observation or 'columns' not in observation:
        return None
    # Stronger than the evaluator's equivalence: no rounding, row sorting, or
    # column permutations. Full values are used, never a displayed sample.
    rows = [[[type(v).__name__, v] for v in row] for row in observation['rows']]
    return r.prior.batch.digest([len(observation['columns']), rows])


def clause_map(sql):
    try:
        tree = query(sql)
    except (ValueError, sqlglot.errors.SqlglotError):
        return {}
    result = {}
    for index, scope in enumerate(tree.find_all(exp.Select)):
        for key in CLAUSES:
            value=scope.args.get(key)
            if value is None or value==[]:
                continue
            nodes=value if isinstance(value,list) else [value]
            result[f's{index}.{key}']=' '.join(n.sql(dialect='sqlite',normalize=True,comments=False) for n in nodes)
    return result


def history_facts(history):
    classes, attempts, changes = {}, [], []
    previous = None
    for event in history:
        key=result_key(event['observation']);parts=clause_map(event['sql'])
        group=None
        if key is not None:
            group=classes.setdefault(key,len(classes)+1)
        attempts.append(dict(event=event['event'],result_group=group,parts=parts))
        if previous is not None:
            changed=[k for k in sorted(parts.keys()|previous['parts'].keys())
                     if parts.get(k)!=previous['parts'].get(k)]
            changes.append(dict(changed_clauses=changed,
                same_result=key is not None and key==previous['key'],
                differences=[dict(clause=k,before=previous['parts'].get(k,'')[:160],
                                  after=parts.get(k,'')[:160]) for k in changed[:3]]))
        previous=dict(parts=parts,key=key)
    return dict(attempts=attempts,changes=changes)


def route(history):
    latest=history[-1]
    if not clause_map(latest['sql']) or latest['observation'].get('status')!='ok':
        return 'rebuild'
    if len(history)>1:
        key=result_key(latest['observation'])
        if key is not None and key==result_key(history[-2]['observation']):
            return 'reground'
    return 'edit'


def interventions(sql):
    try:
        root=query(sql)
    except (ValueError,sqlglot.errors.SqlglotError):
        return []
    options=[]
    for sid,scope in enumerate(root.find_all(exp.Select)):
        for key in ('distinct','where','having','limit'):
            if scope.args.get(key) is None:
                continue
            altered=root.copy();target=list(altered.find_all(exp.Select))[sid]
            target.set(key,None)
            if key=='limit':
                target.set('offset',None)
            options.append(dict(scope=sid,removed=key,
                original=scope.args[key].sql(dialect='sqlite')[:200],
                sql=altered.sql(dialect='sqlite')))
    return options[:3]


def predicate_field(task,sql):
    """Resolve one literal predicate field; ambiguity causes a skip."""
    index=r.methods.selectors.ri._schema_index(task['db_path'])[0]['tables']
    try:
        root=query(sql)
        for scope in traverse_scope(root):
            sources={alias.casefold():node.name.casefold() for alias,node in scope.sources.items()
                     if isinstance(node,exp.Table) and node.name.casefold() in index}
            for column in scope.columns:
                if column.find_ancestor(exp.Select) is not scope.expression:
                    continue
                parent=column.parent
                if not isinstance(parent,(exp.EQ,exp.NEQ,exp.GT,exp.GTE,exp.LT,exp.LTE,exp.Like,exp.In,exp.Between)):
                    continue
                if not any(parent.find_all(exp.Literal)):
                    continue
                owners=([sources[column.table.casefold()]] if column.table.casefold() in sources else
                        [] if column.table else list(dict.fromkeys(sources.values())))
                owners=[t for t in owners if column.name.casefold() in index[t]['columns']]
                if len(owners)==1:
                    return owners[0],column.name.casefold()
    except (ValueError,sqlglot.errors.SqlglotError):
        pass
    return None


@lru_cache(maxsize=1024)
def count_check(db_path, sql):
    statement='SELECT COUNT(*) AS result_rows FROM ('+sql.rstrip(';')+') AS effect_probe'
    observed=r.execute(db_path,statement,timeout=3,max_rows=1)
    return dict(sql=statement,observation=r.stable(observed))


def compile_input(task,history,with_checks=True):
    facts=history_facts(history);latest=history[-1];mode=route(history)
    report=dict(mode=mode,attempts=len(history),
        result_groups=[x['result_group'] for x in facts['attempts']],
        latest_result=common.observation(latest['observation']),
        tried_changes=facts['changes'][-2:])
    if mode!='edit':
        parts=facts['attempts'][-1]['parts']
        report['last_rejected_choices']={k:v[:200] for k,v in list(parts.items())[:5]}
    probes=[]
    if with_checks:
        effects=[]
        for intervention in interventions(latest['sql']):
            probe=count_check(task['db_path'],intervention['sql']);probes.append(probe)
            obs=probe['observation']
            effects.append(dict(scope=intervention['scope'],removed=intervention['removed'],
                original=intervention['original'],
                resulting_rows=obs['rows'][0][0] if obs['status']=='ok' else None,
                status=obs['status']))
        if effects:
            report['operator_checks']=effects
        field=predicate_field(task,latest['sql'])
        if field:
            probe=common.field_probe(task['db_path'],*field)
            probes.extend(dict(sql=sql,observation=None) for sql in probe['sql'])
            report['stored_field_values']=dict(table=field[0],column=field[1],
                statistics=probe['statistics'],sample=probe['values']['rows'][:5] if probe['values'].get('rows') else [],
                sample_is_incomplete=True)
    return report,probes


def short_report(report,limit=1800):
    result=json.loads(common.compact(report))
    obs=result['latest_result']
    if 'columns' in obs:
        obs['columns']=obs['columns'][:4]
    if 'rows' in obs:
        obs['rows']=[[v[:64] if isinstance(v,str) else v for v in row[:4]] for row in obs['rows'][:2]]
        obs['sample_only']=True;obs['first_columns_only']=True
    for change in result['tried_changes']:
        change['differences']=change['differences'][:1]
        for diff in change['differences']:
            diff['before']=diff['before'][:100];diff['after']=diff['after'][:100]
    if 'last_rejected_choices' in result:
        result['last_rejected_choices']={k:v[:100] for k,v in list(result['last_rejected_choices'].items())[:4]}
    for key in ('last_rejected_choices','tried_changes','stored_field_values'):
        if len(common.compact(result))<=limit:
            break
        result.pop(key,None)
    if len(common.compact(result))>limit:
        result.pop('operator_checks',None)
    result['text_is_a_bounded_summary']=True
    return result


def build(task,state,full,history,variant):
    r.methods.validate(task,state)
    if variant=='direct':
        return common.make_build('direct')(task,state,full)
    report,probes=compile_input(task,history,with_checks=variant!='no_checks')
    instruction=(
        'Generate one SQLite SELECT query that answers the question. The user has rejected the earlier results. '
        'The execution history below records observed results and changes, not a correct answer. '
        'A rejected query does not imply that each of its clauses is wrong. '
        'Operator checks temporarily remove one clause to measure its effect; their outputs are not proposed answers. '
        'Match the requested output columns, aggregation level, filtering population, duplicate handling and value format. '
        'Return only one complete SQL query.\n')
    if report['mode']=='edit':
        instruction+='Revise the latest query using the question and the observed effects.\nLatest query:\n'+state['sql']+'\n'
    else:
        instruction+='Reconstruct the query from the question and catalog; previous modifications have not established a correct interpretation.\n'
    text=(instruction+'Question: '+task['question']+'\nEvidence: '+task.get('evidence','')
          +'\nColumn catalog and declared joins:\n'+common.catalog(task['db_path'])
          +'\nObserved failure record:\n'+common.compact(short_report(report)))
    metadata=dict(version=VERSION,variant=variant,compiled=report,probes=probes,
                  history_events=[x['event'] for x in history],
                  history_digests=[r.prior.batch.digest([x['sql'],r.stable(x['observation'])]) for x in history])
    return [dict(role='user',content=text)],metadata
