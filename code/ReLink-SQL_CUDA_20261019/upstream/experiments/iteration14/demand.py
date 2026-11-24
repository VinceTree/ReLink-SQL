"""Public repair demands drive schema materialization and bounded data probes."""
from functools import lru_cache
import json

from iteration13 import runner as r

VERSION = 'repair_demand_v1'
PLAN_CAP = 384
MODES = {'edit', 'rebind', 'rethink'}
FAULTS = {'binding', 'predicate', 'join', 'aggregate', 'projection', 'set', 'ordering', 'unknown'}


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), default=str)


@lru_cache(maxsize=64)
def catalog(db_path):
    schema = r.methods.selectors.ri._schema_index(db_path)[0]
    lines = []
    for table in schema['tables'].values():
        cols = [c['name'] + (':PK' if c['pk_position'] else '') for c in table['columns'].values()]
        lines.append(table['name'] + '(' + ','.join(cols) + ')')
    for edge in schema['fk_edges']:
        if edge['resolved']:
            lines.append('FK ' + edge['source_table'] + '(' + ','.join(edge['source_columns']) + ') -> '
                         + edge['target_table'] + '(' + ','.join(edge['target_columns']) + ')')
    return '\n'.join(lines)


def observation(obs):
    result = {k: obs[k] for k in ('status', 'error', 'columns') if k in obs}
    if 'rows' in obs:
        rows = obs['rows']
        result.update(row_count=len(rows), rows=[
            [str(v)[:120] if isinstance(v, (str, bytes)) else v for v in row] for row in rows[:3]],
            sample_only=len(rows) > 3)
    return result


def parse_plan(text, task):
    plan = r.methods.selectors.parse_object(text)
    tables = r.methods.selectors.ri._schema_index(task['db_path'])[0]['tables']
    errors = []
    if not isinstance(plan, dict):
        return dict(mode='rethink', fault='unknown', change=text[:1600], needs={}, inspect=[],
                    valid=False, invalid=['invalid JSON'])
    if plan.get('mode') not in MODES or plan.get('fault') not in FAULTS:
        errors.append('invalid action or fault')
    if not isinstance(plan.get('change'), str):
        errors.append('missing textual repair requirement')
    chosen = {}
    needs = plan.get('needs', {})
    if not isinstance(needs, dict):
        errors.append('invalid needs'); needs = {}
    for table, columns in needs.items():
        key = table.casefold()
        if key not in tables or not isinstance(columns, list):
            errors.append('unknown table or invalid columns: ' + table); continue
        chosen[key] = []
        for column in columns:
            if not isinstance(column, str) or column.casefold() not in tables[key]['columns']:
                errors.append('unknown column: ' + str(column)); continue
            chosen[key].append(column.casefold())
    inspect = []
    raw_probes = plan.get('inspect', [])
    if not isinstance(raw_probes, list):
        errors.append('invalid inspect'); raw_probes = []
    for item in raw_probes[:2]:
        if not isinstance(item, list) or len(item) != 2 or not all(isinstance(x, str) for x in item):
            errors.append('invalid field probe'); continue
        table, column = (x.casefold() for x in item)
        if table not in tables or column not in tables[table]['columns']:
            errors.append('unknown probe field'); continue
        inspect.append([table, column])
        chosen.setdefault(table, []).append(column)
    return dict(mode=plan.get('mode') if plan.get('mode') in MODES else 'rethink',
                fault=plan.get('fault') if plan.get('fault') in FAULTS else 'unknown',
                change=str(plan.get('change', text))[:1600], needs=chosen, inspect=inspect,
                valid=not errors, invalid=errors)


def materialize(task, state, full, plan):
    if not plan['valid'] or plan['mode'] == 'rethink':
        return dict(full, demand_fallback='invalid or broad diagnosis')
    context = r.contexts.recovery_context(task, dict(text=state['sql'], observation=state['observation']))
    chosen = {t.casefold(): set(c.casefold() for c in columns)
              for t, columns in context['selected_columns'].items()}
    for table, columns in plan['needs'].items():
        chosen.setdefault(table, set()).update(columns)
    schema = r.methods.selectors.ri._schema_index(task['db_path'])[0]
    closed, paths = r.methods.selectors.ri._paths(schema['tables'], schema['fk_edges'], set(chosen))
    for table in closed - set(chosen):
        chosen[table] = set(schema['tables'][table]['columns'])
    result = r.methods.selectors.render(task, chosen, preserve_keys=True)
    return dict(result, demand_paths=paths)


def quote(identifier):
    return '"' + identifier.replace('"', '""') + '"'


@lru_cache(maxsize=512)
def field_probe(db_path, table, column):
    # Callers must validate against the public catalog first. Quoting is still
    # required: actual schema identifiers can contain spaces and punctuation.
    index = r.methods.selectors.ri._schema_index(db_path)[0]['tables']
    assert table in index and column in index[table]['columns']
    t, c = quote(index[table]['name']), quote(index[table]['columns'][column]['name'])
    stats_sql = f'SELECT COUNT(*), COUNT({c}), COUNT(DISTINCT {c}), MIN({c}), MAX({c}) FROM {t}'
    values_sql = f'SELECT DISTINCT {c} FROM {t} WHERE {c} IS NOT NULL ORDER BY LENGTH(CAST({c} AS TEXT)), CAST({c} AS TEXT) LIMIT 8'
    stats = r.execute(db_path, stats_sql, timeout=10, max_rows=20)
    values = r.execute(db_path, values_sql, timeout=10, max_rows=20)
    return dict(table=table, column=column, statistics=observation(stats), values=observation(values),
                values_are_sample=True, sql=[stats_sql, values_sql])


def make_build(variant):
    def build(task, state, full, stage='answer', auxiliary=None):
        r.methods.validate(task, state)
        task_text = '\nQuestion: ' + task['question'] + '\nEvidence: ' + task.get('evidence', '')
        obs = observation(state['observation'])
        rejected = list(dict.fromkeys(state['rejected_sqls']))[-2:]
        base = task_text + '\nRejected SQL: ' + state['sql'] + '\nExecution: ' + compact(obs)
        if variant == 'direct':
            prompt = ('Correct a SQLite query rejected by the user. Reconcile the requested result with the query. '
                      'Keep justified logic and correct unsupported choices. Return only one complete SELECT query.\n'
                      'Complete column catalog:\n' + catalog(task['db_path']) + base
                      + '\nPreviously rejected candidates: ' + compact(rejected))
            return [dict(role='user', content=prompt)], dict(version=VERSION, variant=variant)
        if stage == 'feedback':
            prompt = ('The user rejected this SQLite query. Identify the repair requirement using the question, '
                      'catalog and observed result. Do not invent the correct result or treat execution success '
                      'as correctness. Choose edit for a localized logic correction, rebind for a wrong or missing '
                      'table/column choice, or rethink when the query needs broader reconstruction. '
                      'List catalog fields needed by the repair, including fields absent from the rejected SQL. '
                      'Request a field inspection only if its actual stored values or null/distinct counts would '
                      'resolve a specific uncertainty. Return only JSON in this format:\n'
                      '{"mode":"edit|rebind|rethink","fault":"binding|predicate|join|aggregate|projection|set|ordering|unknown",'
                      '"change":"brief precise repair requirement","needs":{"table":["column"]},'
                      '"inspect":[["table","column"]]}\n'
                      'At most two inspected fields; no replacement SQL.\nComplete column catalog:\n'
                      + catalog(task['db_path']) + base
                      + '\nPreviously rejected candidates: ' + compact(rejected))
            return [dict(role='user', content=prompt)], dict(version=VERSION, variant=variant)
        plan = parse_plan(auxiliary, task)
        context = full if variant == 'demand_full' else materialize(task, state, full, plan)
        probes = [field_probe(task['db_path'], t, c) for t, c in plan['inspect']]
        public_probes = [{k: v for k, v in x.items() if k != 'sql'} for x in probes]
        prompt = ('Write one correct SQLite SELECT query. The repair requirement is a fallible diagnosis; '
                  'resolve it against the original question and available database facts. '
                  'Samples are incomplete and do not define the required answer. '
                  'Return only the complete SQL.\nSchema needed for repair:\n' + context['schema_text']
                  + task_text + '\nRepair requirement: ' + compact(plan)
                  + '\nRequested database evidence: ' + compact(public_probes))
        if plan['mode'] == 'edit':
            prompt += '\nCurrent rejected SQL: ' + state['sql'] + '\nExecution: ' + compact(obs)
            prompt += '\nPreserve unrelated justified logic while applying the required correction.'
        else:
            prompt += '\nReconstruct the query from the question and the resolved schema requirements.'
        prompt += '\nAlready rejected; do not repeat unchanged: ' + compact(rejected)
        return [dict(role='user', content=prompt)], dict(version=VERSION, variant=variant, plan=plan,
            selected_columns=context['selected_columns'], probe_records=probes,
            demand_fallback=context.get('demand_fallback'), demand_paths=context.get('demand_paths', []))
    return build


def spec(*args, **kwargs):
    result = r.spec(*args, **kwargs)
    if result['stage'] == 'feedback':
        result['request']['max_tokens'] = PLAN_CAP
    return result


def round_runner(variant):
    if variant == 'critic':
        return r.round_run
    return r.methods.selectors.clone(r.round_run, build=make_build(variant), spec=spec)
