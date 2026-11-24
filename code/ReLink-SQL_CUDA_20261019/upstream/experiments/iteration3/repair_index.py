"""Inspectable SQL tree / schema graph / bindings with FK-path closure.

Inspired by BookRAG's T/G/M organization; SQL bindings and declared FK paths
are computed deterministically rather than generated as a knowledge graph.
This is a retrieval prototype, not a proof of semantic repair completeness.
"""
from __future__ import annotations

from collections import OrderedDict, deque
import copy
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sqlite3
import time

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope

VERSION = 'tree_schema_path_closure_v1'
MAX_FK_HOPS = 3
_CACHE = OrderedDict()


def clear_schema_cache():
    _CACHE.clear()


def _key(name):
    return str(name).casefold()


def _quote(name):
    return '"' + str(name).replace('"', '""') + '"'


def _words(text):
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', str(text)).lower()
    stop = {'a', 'an', 'the', 'in', 'of', 'to', 'and', 'or', 'is', 'are', 'was', 'were',
            'which', 'what', 'how', 'many', 'all', 'for', 'with', 'from', 'by', 'that',
            'this', 'those', 'these', 'show', 'list', 'find', 'give', 'return', 'please'}
    return set(re.findall(r'[a-z0-9]+', text)) - stop


def _input(initial):
    if isinstance(initial, str):
        text, observation = initial, {}
    else:
        text, observation = str(initial.get('text', '')), initial.get('observation') or {}
    blocks = re.findall(r'```(?:sql|sqlite)?\s*(.*?)```', text, re.I | re.S)
    text = (blocks[0] if blocks else text).strip()
    start = re.search(r'\b(SELECT|WITH)\b', text, re.I)
    return (text[start.start():] if start else text), {
        key: observation[key] for key in ('status', 'error') if key in observation}


def _tid(name):
    return 'table:' + _key(name)


def _cid(table, column):
    return 'column:' + json.dumps([_key(table), _key(column)], ensure_ascii=False)


def _schema_index(path):
    started = time.perf_counter()
    path = Path(path).resolve()
    stat = path.stat()
    wal = Path(str(path) + '-wal')
    wal_stat = (wal.stat().st_size, wal.stat().st_mtime_ns) if wal.exists() else None
    key = (str(path), stat.st_size, stat.st_mtime_ns, wal_stat)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key], True, 0., time.perf_counter() - started
    con = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=.2)
    try:
        con.execute('PRAGMA query_only=ON')
        con.set_progress_handler(lambda: int(time.perf_counter() - started > 3.), 1000)
        tables = {}
        rows = con.execute("SELECT name,type,sql FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
        for name, kind, ddl in rows:
            columns = {}
            for row in con.execute(f'PRAGMA table_xinfo({_quote(name)})'):
                columns[_key(row[1])] = {'name': row[1], 'type': row[2] or 'unspecified',
                                        'not_null': bool(row[3]), 'pk_position': row[5], 'hidden': row[6]}
            grouped = {}
            for row in con.execute(f'PRAGMA foreign_key_list({_quote(name)})'):
                grouped.setdefault(row[0], []).append(row)
            fks = []
            for fid, parts in grouped.items():
                parts.sort(key=lambda row: row[1])
                fks.append({'id': f'fk:{_key(name)}:{fid}', 'source_table': _key(name),
                            'target_table': _key(parts[0][2]), 'source_columns': [row[3] for row in parts],
                            'target_columns': [row[4] for row in parts]})
            tables[_key(name)] = {'name': name, 'kind': kind, 'ddl': (ddl or '').rstrip(';') + ';',
                                  'columns': columns, 'foreign_keys': fks}
        nodes, edges, fk_edges = [], [], []
        for table, metadata in tables.items():
            nodes.append({'id': _tid(table), 'kind': metadata['kind'], 'name': metadata['name'], 'ddl': metadata['ddl']})
            for column, info in metadata['columns'].items():
                nodes.append({'id': _cid(table, column), 'kind': 'column', 'table': metadata['name'], **info})
                edges.append({'kind': 'contains', 'source': _tid(table), 'target': _cid(table, column)})
            for original in metadata['foreign_keys']:
                edge = copy.deepcopy(original)
                target = tables.get(edge['target_table'])
                if target and any(c is None for c in edge['target_columns']):
                    pk = sorted((c for c in target['columns'].values() if c['pk_position']), key=lambda c: c['pk_position'])
                    if len(pk) == len(edge['target_columns']):
                        edge['target_columns'] = [c['name'] for c in pk]
                edge['kind'] = 'foreign_key'
                edge['resolved'] = bool(target and all(c is not None and _key(c) in target['columns'] for c in edge['target_columns']))
                edge['column_pairs'] = [{'source': _cid(table, a), 'target': _cid(edge['target_table'], b) if b else None}
                                        for a, b in zip(edge['source_columns'], edge['target_columns'])]
                fk_edges.append(edge)
                edges.append(edge)
        result = {'tables': tables, 'G': {'nodes': nodes, 'edges': edges}, 'fk_edges': fk_edges}
        _CACHE[key] = result
        while len(_CACHE) > 16:
            _CACHE.popitem(last=False)
        elapsed = time.perf_counter() - started
        return result, False, elapsed, 0.
    finally:
        con.close()


def _own_columns(scope):
    return [column for column in scope.columns if column.find_ancestor(exp.Select, exp.Union) is scope.expression]


def _structure(sql, tables):
    """Return T, M, and binding metadata; unresolved links stay explicit."""
    T = {'nodes': [{'id': 'query', 'kind': 'query'}], 'edges': []}
    M, metadata = [], {'scopes': [], 'required': set(), 'unknown_tables': [], 'parse_error': None}
    try:
        parsed = sqlglot.parse(sql, read='sqlite')
        if len(parsed) != 1 or not isinstance(parsed[0], exp.Query):
            raise ValueError('Expected one query')
        scopes = list(traverse_scope(parsed[0]))
    except (ValueError, sqlglot.errors.SqlglotError) as error:
        metadata['parse_error'] = type(error).__name__
        return T, M, metadata
    scope_ids = {id(scope): f's{index}' for index, scope in enumerate(scopes)}
    sources = {id(scope): {alias.casefold(): (alias, source) for alias, (_, source) in scope.selected_sources.items()} for scope in scopes}

    def source_binding(source, column, seen):
        if isinstance(source, exp.Table):
            table = _key(source.name)
            if table in tables and _key(column) in tables[table]['columns']:
                return {'nodes': [_cid(table, column)], 'tables': [table], 'status': 'bound'}
            return None
        if isinstance(source, Scope):
            names = source.outer_columns or source.expression.named_selects
            selects = list(source.expression.selects)
            found = [i for i, name in enumerate(names) if _key(name) == _key(column)]
            if len(found) == 1 and found[0] < len(selects):
                expression = selects[found[0]]
                refs = [c for c in expression.find_all(exp.Column)
                        if c.find_ancestor(exp.Select, exp.Union) is source.expression]
                mapped = [resolve(source, c, seen) for c in refs]
                return {'nodes': sorted({n for r in mapped for n in r.get('nodes', [])}),
                        'tables': sorted({t for r in mapped for t in r.get('tables', [])}),
                        'status': 'derived', 'source_scope': scope_ids[id(source)]}
            if '*' in names:
                return resolve(source, exp.column(column), seen)
        return None

    def resolve(scope, column, seen=frozenset()):
        marker = (id(scope), _key(column.table), _key(column.name))
        if marker in seen:
            return {'status': 'recursive_unresolved', 'nodes': [], 'tables': []}
        seen = seen | {marker}
        available = sources[id(scope)]
        candidates = []
        for key, (alias, source) in available.items():
            if column.table and _key(column.table) != key:
                continue
            binding = source_binding(source, column.name, seen)
            if binding is not None:
                candidates.append({**binding, 'alias': alias})
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            return {'status': 'ambiguous', 'nodes': sorted({n for r in candidates for n in r['nodes']}),
                    'tables': sorted({t for r in candidates for t in r['tables']}),
                    'candidate_aliases': [r['alias'] for r in candidates]}
        if scope.can_be_correlated and scope.parent:
            outer = resolve(scope.parent, column, seen)
            if outer.get('nodes'):
                return {**outer, 'correlated': True, 'resolved_scope': scope_ids[id(scope.parent)]}
        return {'status': 'unresolved', 'nodes': [], 'tables': []}

    def source_tables(scope, seen=frozenset()):
        if id(scope) in seen:
            return set()
        found = set()
        for _, source in sources[id(scope)].values():
            if isinstance(source, exp.Table) and _key(source.name) in tables:
                found.add(_key(source.name))
            elif isinstance(source, Scope):
                found.update(source_tables(source, seen | {id(scope)}))
        return found

    clauses = {}
    for scope in scopes:
        sid = scope_ids[id(scope)]
        parent = scope_ids.get(id(scope.parent), 'query')
        T['nodes'].append({'id': sid, 'kind': 'scope', 'scope_type': scope.scope_type.name,
                           'correlated': bool(scope.is_correlated_subquery)})
        T['edges'].append({'source': parent, 'target': sid, 'kind': 'nested_scope'})
        scope_info = {'id': sid, 'parent': parent, 'sources': [], 'correlated': bool(scope.is_correlated_subquery)}
        for argument, value in scope.expression.args.items():
            if value is not None and argument in ('expressions', 'from_', 'joins', 'where', 'group', 'having', 'order', 'limit', 'with_', 'qualify'):
                cid = sid + ':' + argument
                clauses[(sid, argument)] = cid
                T['nodes'].append({'id': cid, 'kind': 'clause', 'clause': argument})
                T['edges'].append({'source': sid, 'target': cid, 'kind': 'contains'})
        for _, (alias, source) in sources[id(scope)].items():
            if isinstance(source, exp.Table):
                table = _key(source.name)
                node = {'tree_node': f'ref{len(M)}', 'scope': sid, 'kind': 'table', 'name': source.name,
                        'alias': alias, 'graph_nodes': [_tid(table)] if table in tables else [],
                        'status': 'bound' if table in tables else 'unresolved'}
                if table in tables:
                    metadata['required'].add(table)
                else:
                    metadata['unknown_tables'].append(source.name)
                scope_info['sources'].append({'alias': alias, 'table': table, 'source_kind': 'table'})
            else:
                node = {'tree_node': f'ref{len(M)}', 'scope': sid, 'kind': 'table', 'name': alias, 'alias': alias,
                        'graph_nodes': [_tid(t) for t in sorted(source_tables(source))] if isinstance(source, Scope) else [],
                        'status': 'derived', 'source_scope': scope_ids.get(id(source))}
                scope_info['sources'].append({'alias': alias, 'source_scope': scope_ids.get(id(source)), 'source_kind': 'derived'})
            M.append(node)
            T['nodes'].append({'id': node['tree_node'], 'kind': 'table_reference', 'name': node['name']})
            T['edges'].append({'source': clauses.get((sid, 'from_'), sid), 'target': node['tree_node'], 'kind': 'contains'})
        for column in _own_columns(scope):
            binding = resolve(scope, column)
            node = {'tree_node': f'ref{len(M)}', 'scope': sid, 'kind': 'column', 'name': column.name,
                    'qualifier': column.table, 'graph_nodes': binding.pop('nodes', []), **binding}
            M.append(node)
            current = column
            while current.parent is not None and current.parent is not scope.expression:
                current = current.parent
            clause = current.arg_key if current.parent is scope.expression else 'expressions'
            T['nodes'].append({'id': node['tree_node'], 'kind': 'column_reference', 'name': column.name})
            T['edges'].append({'source': clauses.get((sid, clause), sid), 'target': node['tree_node'], 'kind': 'contains'})
        metadata['scopes'].append(scope_info)
    return T, M, metadata


def _paths(tables, fks, seeds):
    adjacency = {table: [] for table in tables}
    for edge in fks:
        if edge['resolved'] and edge['source_table'] in tables and edge['target_table'] in tables:
            a, b = edge['source_table'], edge['target_table']
            adjacency[a].append((b, edge['id'])); adjacency[b].append((a, edge['id']))
    def distances(start):
        distance, queue = {start: 0}, deque([start])
        while queue:
            node = queue.popleft()
            if distance[node] >= MAX_FK_HOPS:
                continue
            for nxt, _ in adjacency[node]:
                if nxt not in distance:
                    distance[nxt] = distance[node] + 1; queue.append(nxt)
        return distance
    distances_by_seed = {s: distances(s) for s in seeds}
    selected, pairs = set(seeds), []
    ordered = sorted(seeds)
    for i, source in enumerate(ordered):
        for target in ordered[i + 1:]:
            forward, reverse = distances_by_seed[source], distances_by_seed[target]
            length = forward.get(target)
            if length is None:
                pairs.append({'source': source, 'target': target, 'status': 'no_declared_path_within_3_hops'})
                continue
            bridge = {node for node in tables if node in forward and node in reverse and forward[node] + reverse[node] == length}
            used_edges = set()
            counts = {source: 1}
            for node in sorted(bridge, key=lambda x: (forward[x], x)):
                for nxt, eid in adjacency[node]:
                    if nxt in bridge and forward[nxt] == forward[node] + 1:
                        used_edges.add(eid); counts[nxt] = counts.get(nxt, 0) + counts.get(node, 0)
            selected.update(bridge)
            pairs.append({'source': source, 'target': target, 'status': 'connected', 'distance': length,
                          'path_count': counts.get(target, 0), 'tables': sorted(bridge), 'fk_edges': sorted(used_edges)})
    return selected, pairs


def _schema_text(tables, selected):
    return '\n\n'.join(tables[name]['ddl'] for name in sorted(selected))


def build_repair_context(task, initial, mode='closure', token_budget=768):
    """Return complete table definitions and inspectable T/G/M structures.

    closure retains every seed plus all shortest FK-path connecting tables
    within three hops. Budget overflow never truncates this set. flat uses the
    same ranking but no path expansion. The estimate is ceil(chars/4); callers
    must meter actual tokenizer cost. This does not claim semantic completeness.
    """
    if mode not in ('closure', 'flat', 'full'):
        raise ValueError('mode must be closure, flat, or full')
    started = time.perf_counter()
    sql, observation = _input(initial)
    schema, hit, offline_seconds, lookup_seconds = _schema_index(task['db_path'])
    online_started = time.perf_counter()
    tables = schema['tables']
    T, M, structure = _structure(sql, tables)
    required = set(structure['required'])
    ranking = {table: {'table': table, 'priority': 0, 'score': 0., 'reasons': ['original_sql_reference'], 'required': True}
               for table in required}
    def seed(table, priority, reason, score=0.):
        if table not in ranking:
            ranking[table] = {'table': table, 'priority': priority, 'score': score, 'reasons': [], 'required': False}
        ranking[table]['reasons'].append(reason)
        ranking[table]['priority'] = min(priority, ranking[table]['priority'])
        ranking[table]['score'] = max(score, ranking[table]['score'])
    error = str(observation.get('error', '')) if observation.get('status') == 'error' else ''
    match = re.match(r'(?:no such column|ambiguous column name):\s*(.+)', error, re.I)
    exact_owners = []
    if match:
        raw = match.group(1).strip()
        matching = [m for m in M if m['kind'] == 'column' and _key('.'.join(x for x in [m.get('qualifier'), m['name']] if x)) == _key(raw)]
        column = matching[0]['name'] if matching else raw.rsplit('.', 1)[-1]
        scopes = {m['scope'] for m in matching}
        visible = {s['table'] for sc in structure['scopes'] if sc['id'] in scopes for s in sc['sources'] if s.get('table')}
        exact_owners = [table for table, info in tables.items() if _key(column) in info['columns']]
        preferred = [table for table in exact_owners if table in visible]
        for table in preferred or exact_owners:
            seed(table, 1 if table in visible else 2, 'exact_error_column_owner')
    unknown = list(structure['unknown_tables'])
    table_error = re.match(r'no such table:\s*(.+)', error, re.I)
    if table_error:
        unknown.append(table_error.group(1).strip())
    spelling_candidates = []
    for name in sorted(set(unknown)):
        candidates = sorted(((SequenceMatcher(None, _key(name), table).ratio(), table) for table in tables), reverse=True)
        for score, table in candidates[:3]:
            if score >= .55:
                seed(table, 3, 'spelling_candidate_not_confirmed', score)
                spelling_candidates.append({'unresolved': name, 'candidate': table, 'score': score})
    words = _words(str(task.get('question', '')) + ' ' + str(task.get('evidence', '')))
    lexical = sorted([(3 * len(_words(table) & words) + sum(bool(_words(c) & words) for c in info['columns']), table)
                      for table, info in tables.items()], key=lambda x: (-x[0], x[1]))
    soft = next(((score, table) for score, table in lexical if score > 0 and table not in required), None)
    if soft:
        seed(soft[1], 4, 'question_evidence_lexical_soft', soft[0])
    seeds = set(ranking)
    closed, pairs = _paths(tables, schema['fk_edges'], seeds)
    budget_chars = max(0, int(token_budget)) * 4
    ranked = sorted(ranking.values(), key=lambda r: (r['priority'], -r['score'], r['table']))
    status = 'ok'
    if mode == 'full':
        selected = set(tables)
    elif mode == 'closure':
        selected = closed
    else:
        selected = set(required)
        for item in ranked:
            proposed = selected | {item['table']}
            if len(_schema_text(tables, proposed)) <= budget_chars:
                selected = proposed
    schema_text = _schema_text(tables, selected)
    if len(schema_text) > budget_chars:
        full = _schema_text(tables, set(tables))
        if mode == 'closure' and len(full) <= len(schema_text):
            selected = set(tables); schema_text = full; status = 'full_fallback_overflow'
        else:
            status = 'overflow_preserved_complete_tables'
    missing = sum(pair['status'] != 'connected' for pair in pairs)
    ambiguous = sum(pair.get('path_count', 0) > 1 for pair in pairs)
    column_bindings = [m for m in M if m['kind'] == 'column']
    covered = sum(bool(m['graph_nodes']) and m['status'] in ('bound', 'derived') for m in column_bindings)
    selected_columns = {tables[t]['name']: [c['name'] for c in tables[t]['columns'].values()] for t in sorted(selected)}
    feedback = ('SQLite error: ' + error + '\n' if error else '') + (
        'Schema facts are declared metadata; lexical/spelling candidates and FK paths do not establish the intended query. '
        'Execution success does not prove semantic correctness.')
    if missing:
        feedback += f' {missing} seed pairs lack a declared FK path within {MAX_FK_HOPS} hops; undeclared joins are not ruled out.'
    if status != 'ok':
        feedback += ' Context budget exceeded; complete definitions were retained.'
    diagnostics = {'version': VERSION, 'mode': mode, 'status': status, 'schema_cache_hit': hit,
                   'offline_schema_build_seconds': offline_seconds, 'schema_lookup_seconds': lookup_seconds,
                   'parse_ok': structure['parse_error'] is None, 'scope_count': len(structure['scopes']),
                   'required_table_count': len(required), 'seed_table_count': len(seeds),
                   'selected_table_count': len(selected), 'bridge_table_count': len(closed - seeds) if mode == 'closure' else 0,
                   'missing_fk_path_count': missing, 'ambiguous_fk_path_count': ambiguous,
                   'ambiguous_binding_count': sum(m['status'] == 'ambiguous' for m in M),
                   'correlated_binding_count': sum(bool(m.get('correlated')) for m in M),
                   'column_binding_count': len(column_bindings), 'bound_column_count': covered,
                   'scope_binding_coverage': covered / max(1, len(column_bindings)),
                   'schema_chars': len(schema_text), 'estimated_schema_tokens': (len(schema_text) + 3) // 4,
                   'token_budget_hint': token_budget, 'overflow': len(schema_text) > budget_chars,
                   'full_schema_included': selected == set(tables), 'semantic_completeness_claimed': False}
    index = {'version': VERSION, 'T': T, 'G': copy.deepcopy(schema['G']), 'M': M,
             'selection': {'required_seeds': sorted(required), 'ranked_seeds': ranked,
                           'global_exact_error_owners': exact_owners, 'spelling_candidates': spelling_candidates,
                           'soft_lexical_candidate': soft[1] if soft else None, 'closure_pairs': pairs,
                           'closure_tables': sorted(closed), 'selected_tables': sorted(selected),
                           'omitted_seeds': sorted(seeds - selected)}, 'scope_metadata': structure['scopes']}
    diagnostics['online_seconds'] = time.perf_counter() - online_started
    seconds = time.perf_counter() - started
    return {'schema_text': schema_text, 'text': feedback, 'feedback': feedback,
            'index': index, 'selected_tables': [tables[t]['name'] for t in sorted(selected)],
            'selected_columns': selected_columns, 'column_omissions': {}, 'diagnostics': diagnostics,
            'seconds': seconds, 'cost': {'total_seconds': seconds, 'offline_schema_build_seconds': offline_seconds,
                                       'online_seconds': diagnostics['online_seconds'], 'cache_lookup_seconds': lookup_seconds}}
