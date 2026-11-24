"""Column-dependent context on the frozen table-path closure.

This is an explicitly incomplete semantic retrieval heuristic. All parsed SQL
column dependencies, key endpoints, and ambiguous candidates are retained;
unknown coverage falls back conservatively. No token truncation is used.
"""
from __future__ import annotations

import copy
import json
import re
import sqlite3
import time

import sqlglot
from sqlglot import exp

try:
    from .repair_index import build_repair_context, _input, _key, _words
except ImportError:
    from repair_index import build_repair_context, _input, _key, _words

VERSION = 'column_dependency_context_v1'
LEXICAL_COLUMNS_PER_TABLE = 4


def _column_id(node):
    return tuple(json.loads(node[len('column:'):])) if node.startswith('column:') else None


def _references_retained(reference, selected, primary):
    schema = reference.this
    if not isinstance(schema, exp.Schema) or not isinstance(schema.this, exp.Table):
        return False
    table = _key(schema.this.name)
    columns = [c.name for c in schema.expressions] or primary.get(table, [])
    return table in selected and bool(columns) and all(_key(c) in selected[table] for c in columns)


def _constraint_retained(node, table, selected, primary):
    if isinstance(node, exp.ColumnConstraint):
        return _constraint_retained(node.args.get('kind'), table, selected, primary)
    if isinstance(node, exp.Constraint):
        return all(_constraint_retained(child, table, selected, primary) for child in node.expressions)
    if isinstance(node, exp.ForeignKey):
        return (all(_key(c.name) in selected[table] for c in node.expressions)
                and _references_retained(node.args['reference'], selected, primary))
    if isinstance(node, exp.Reference):
        return _references_retained(node, selected, primary)
    if isinstance(node, exp.PrimaryKey):
        return all(_key(c.name) in selected[table] for c in node.expressions)
    if isinstance(node, exp.UniqueColumnConstraint):
        values = node.this.expressions if isinstance(node.this, exp.Schema) else node.expressions
        return all(_key(c.name) in selected[table] for c in values)
    # CHECK constraints may be removed if one of their referenced columns was
    # omitted. Other known column-level constraints have no column dependency.
    return all(_key(c.name) in selected[table] for c in node.find_all(exp.Column)) if node is not None else False


def _normalize_table(ddl, table, keep, selected, primary, prune):
    tree = sqlglot.parse_one(ddl, read='sqlite')
    if not isinstance(tree, exp.Create) or not isinstance(tree.this, exp.Schema):
        return ddl, [], 'non_column_create_definition'
    if not prune:
        return tree.sql(dialect='sqlite').rstrip(';') + ';', [], None
    expressions, removed_constraints = [], []
    for node in tree.this.expressions:
        if isinstance(node, exp.ColumnDef):
            if _key(node.name) not in keep:
                continue
            node = node.copy()
            constraints = []
            for constraint in node.args.get('constraints') or []:
                if _constraint_retained(constraint, table, selected, primary):
                    constraints.append(constraint)
                else:
                    removed_constraints.append(constraint.sql(dialect='sqlite'))
            node.set('constraints', constraints)
            expressions.append(node)
        elif _constraint_retained(node, table, selected, primary):
            expressions.append(node)
        else:
            removed_constraints.append(node.sql(dialect='sqlite'))
    if not any(isinstance(node, exp.ColumnDef) for node in expressions):
        return ddl, [], 'no_complete_column_definition'
    tree.this.set('expressions', expressions)
    return tree.sql(dialect='sqlite').rstrip(';') + ';', removed_constraints, None


def build_column_context(task, initial, mode='closure_columns', token_budget=768):
    """Same interface as build_repair_context, plus explicit column omissions.

    all_columns normalizes the identical selected-table set without pruning,
    isolating column selection from SQLGlot formatting. Neither mode certifies
    semantic completeness or truncates retained columns to meet a budget.
    """
    if mode not in ('closure_columns', 'all_columns'):
        raise ValueError('mode must be closure_columns or all_columns')
    started = time.perf_counter()
    base = build_repair_context(task, initial, mode='closure', token_budget=token_budget)
    context_seconds = base['seconds']
    offline_seconds = base['cost']['offline_schema_build_seconds']
    sql, observation = _input(initial)
    full_fallback = []
    whole_tables = {}
    try:
        tree = sqlglot.parse_one(sql, read='sqlite')
        if any(not node.expressions for node in tree.find_all(exp.Select)):
            full_fallback.append('empty_select_projection')
    except (ValueError, sqlglot.errors.SqlglotError):
        tree = None
        full_fallback.append('query_parse_failure')
    if not base['diagnostics']['parse_ok']:
        full_fallback.append('scope_index_parse_failure')
    # Determine exact coverage using only M/G and the public compiler error.
    G = base['index']['G']
    graph_columns = {node['id']: node for node in G['nodes'] if node['kind'] == 'column'}
    global_names = {_key(node['name']) for node in graph_columns.values()}
    for mapping in base['index']['M']:
        if mapping['kind'] == 'table' and mapping['status'] == 'unresolved':
            full_fallback.append('unresolved_table_reference')
        if mapping['kind'] == 'column' and mapping['status'] in ('unresolved', 'recursive_unresolved'):
            if _key(mapping['name']) not in global_names and mapping['name'] != '*':
                full_fallback.append('unresolved_column_without_global_exact_owner')
    if full_fallback:
        base = build_repair_context(task, initial, mode='full', token_budget=token_budget)
        context_seconds += base['seconds']
        offline_seconds += base['cost']['offline_schema_build_seconds']
        G = base['index']['G']
        graph_columns = {node['id']: node for node in G['nodes'] if node['kind'] == 'column'}
    table_nodes = {_key(node['name']): node for node in G['nodes'] if node['kind'] in ('table', 'view')}
    columns_by_table = {table: {} for table in table_nodes}
    for node in graph_columns.values():
        columns_by_table[_key(node['table'])][_key(node['name'])] = node
    selected_tables = {_key(table) for table in base['selected_tables']}
    selected = {table: set() for table in selected_tables}
    reasons = {table: {} for table in selected_tables}
    def retain(table, column, reason):
        if table in selected and _key(column) in columns_by_table[table]:
            column = _key(column)
            selected[table].add(column)
            reasons[table].setdefault(column, []).append(reason)
    if tree is not None:
        wildcard = any(star.find_ancestor(exp.Count) is None for star in tree.find_all(exp.Star))
        special_join = any(join.args.get('using') or str(join.args.get('method', '')).upper() == 'NATURAL'
                           for join in tree.find_all(exp.Join))
        if wildcard or special_join:
            for table in selected_tables:
                whole_tables[table] = 'query_wildcard' if wildcard else 'using_or_natural_join'
    if full_fallback:
        whole_tables.update({table: 'global_reference_coverage_fallback' for table in selected_tables})
    for mapping in base['index']['M']:
        if mapping['kind'] != 'column':
            continue
        for gid in mapping['graph_nodes']:
            pair = _column_id(gid)
            if pair:
                retain(*pair, 'ambiguous_sql_candidate' if mapping['status'] == 'ambiguous' else 'sql_reference')
        if mapping['status'] in ('unresolved', 'recursive_unresolved'):
            for table in selected_tables:
                if _key(mapping['name']) in columns_by_table[table]:
                    retain(table, mapping['name'], 'global_exact_unresolved_column')
    error = str(observation.get('error', ''))
    match = re.match(r'(?:no such column|ambiguous column name):\s*(.+)', error, re.I)
    if match:
        raw = match.group(1).strip()
        matching = [m for m in base['index']['M'] if m['kind'] == 'column'
                    and _key('.'.join(x for x in [m.get('qualifier'), m['name']] if x)) == _key(raw)]
        name = matching[0]['name'] if matching else raw.rsplit('.', 1)[-1]
        for table in selected_tables:
            retain(table, name, 'global_exact_error_column')
    primary = {}
    for table in selected_tables:
        cols = columns_by_table[table]
        primary[table] = [c['name'] for c in sorted(cols.values(), key=lambda c: c['pk_position']) if c['pk_position']]
        for column in primary[table]:
            retain(table, column, 'primary_key')
        if any(c.get('hidden') for c in cols.values()):
            whole_tables[table] = 'generated_or_hidden_column_dependencies'
    path_edges = {eid for pair in base['index']['selection']['closure_pairs'] for eid in pair.get('fk_edges', [])}
    for edge in G['edges']:
        if edge['kind'] == 'foreign_key' and edge['id'] in path_edges:
            for pair in edge['column_pairs']:
                for gid in (pair['source'], pair['target']):
                    decoded = _column_id(gid) if gid else None
                    if decoded:
                        retain(*decoded, 'selected_composite_fk_endpoint')
    words = _words(str(task.get('question', '')) + ' ' + str(task.get('evidence', '')))
    for table in selected_tables:
        scores = sorted(((len(_words(column) & words), column) for column in columns_by_table[table]), key=lambda pair: (-pair[0], pair[1]))
        for score, column in scores[:LEXICAL_COLUMNS_PER_TABLE]:
            if score:
                retain(table, column, 'question_evidence_lexical_soft')
        if not selected[table]:
            whole_tables[table] = 'no_grounded_column_seed'
        if table_nodes[table]['kind'] == 'view':
            whole_tables[table] = 'view_definition_not_column_prunable'
    if mode == 'all_columns':
        whole_tables.update({table: 'all_columns_normalization_control' for table in selected_tables})
    for table, reason in whole_tables.items():
        for column in columns_by_table[table]:
            retain(table, column, reason)
    definitions, removed, normalization_fallback = {}, {}, {}
    for table in sorted(selected_tables):
        ddl = table_nodes[table]['ddl']
        try:
            text, dropped, failure = _normalize_table(ddl, table, selected[table], selected, primary,
                                                       prune=table not in whole_tables)
        except (ValueError, sqlglot.errors.SqlglotError):
            text, dropped, failure = ddl, [], 'ddl_parse_failure'
        if failure:
            normalization_fallback[table] = failure
            selected[table] = set(columns_by_table[table])
            whole_tables[table] = failure
        definitions[table] = text
        removed[table] = dropped
    # Validate complete definitions in a throw-away, in-memory database; never
    # execute a modified CREATE statement against the user's source database.
    con = sqlite3.connect(':memory:')
    try:
        for table in sorted(definitions):
            con.execute(definitions[table])
    except sqlite3.Error:
        normalization_fallback['__schema__'] = 'normalized_ddl_failed_sqlite_validation'
        for table in selected_tables:
            definitions[table] = table_nodes[table]['ddl']
            selected[table] = set(columns_by_table[table])
            whole_tables[table] = 'sqlite_definition_validation_fallback'
            removed[table] = []
    finally:
        con.close()
    schema_text = '\n\n'.join(definitions[table] for table in sorted(definitions))
    selected_columns = {table_nodes[table]['name']: [c['name'] for key, c in columns_by_table[table].items() if key in selected[table]]
                        for table in sorted(selected_tables)}
    omissions = {table_nodes[table]['name']: [c['name'] for key, c in columns_by_table[table].items() if key not in selected[table]]
                for table in sorted(selected_tables)}
    overflow = len(schema_text) > max(0, int(token_budget)) * 4
    mapped = [m for m in base['index']['M'] if m['kind'] == 'column' and m['graph_nodes']]
    retained_mapped = sum(all((pair := _column_id(gid)) and pair[0] in selected and pair[1] in selected[pair[0]]
                              for gid in m['graph_nodes']) for m in mapped)
    diagnostic = dict(base['diagnostics'], version=VERSION, mode=mode,
                      status='overflow_retained_column_dependencies' if overflow else 'coverage_fallback' if full_fallback or normalization_fallback else 'ok',
                      overflow=overflow, schema_chars=len(schema_text), estimated_schema_tokens=(len(schema_text) + 3) // 4,
                      selected_column_count=sum(map(len, selected.values())), omitted_column_count=sum(map(len, omissions.values())),
                      whole_table_fallback_count=len(whole_tables), global_coverage_fallback=bool(full_fallback),
                      global_fallback_reasons=sorted(set(full_fallback)), normalization_fallback=normalization_fallback,
                      lexical_columns_per_table=LEXICAL_COLUMNS_PER_TABLE, semantic_completeness_claimed=False,
                      mapped_reference_count=len(mapped), retained_mapped_reference_count=retained_mapped,
                      retained_mapped_reference_coverage=retained_mapped / max(1, len(mapped)),
                      offline_schema_build_seconds=offline_seconds,
                      full_schema_included=base['diagnostics']['full_schema_included'] and not any(omissions.values()))
    seconds = time.perf_counter() - started
    base.update(schema_text=schema_text, selected_columns=selected_columns, column_omissions=omissions,
                diagnostics=diagnostic, seconds=seconds)
    base['index']['column_selection'] = {'version': VERSION, 'mode': mode, 'reasons': reasons,
                                       'whole_table_fallbacks': whole_tables, 'dropped_constraints': removed,
                                       'selected_columns': selected_columns, 'omissions': omissions,
                                       'global_fallback_reasons': sorted(set(full_fallback))}
    base['feedback'] += ' Column selection preserves parsed references and selected key paths; semantic completeness is unverified.'
    base['text'] = base['feedback']
    base['cost']['total_seconds'] = seconds
    base['cost']['offline_schema_build_seconds'] = offline_seconds
    base['cost']['table_context_seconds'] = context_seconds
    base['cost']['column_selection_seconds'] = max(0., seconds - context_seconds)
    return base
