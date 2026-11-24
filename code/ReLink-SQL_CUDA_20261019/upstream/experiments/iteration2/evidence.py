"""Bounded, label-blind SQLite evidence for recovery routing.

Only ``question``, ``schema``, ``db_path`` and ``evidence`` are public task
inputs. In fact, schema is introspected from the read-only database. The
``initial`` argument is a string or a mapping with text and observation.
No query is repaired automatically and no correctness label is inferred.

Requires sqlglot==30.18.0. ``max_tokens_hint`` is a character-based budget
hint, not a tokenizer count; the caller must meter the resulting prompt.
"""
from __future__ import annotations

from collections import deque
from difflib import SequenceMatcher
from pathlib import Path
import json
import re
import sqlite3
import time

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope


MAX_TABLES = 3
MAX_PROBES = 6
PROBE_TIMEOUT = 0.075
TOTAL_TIMEOUT = 1.5


def _quote(name):
    return '"' + str(name).replace('"', '""') + '"'


def _key(name):
    return str(name).casefold()


def _words(text):
    return set(re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text)).lower()))


def _sql(text):
    blocks = re.findall(r"```(?:sqlite|sql)?\s*(.*?)```", text, re.S | re.I)
    text = (blocks[0] if blocks else text).strip()
    start = re.search(r"\b(SELECT|WITH)\b", text, re.I)
    return text[start.start():].strip() if start else text


def _public_initial(initial):
    if isinstance(initial, str):
        return initial, {}
    # Deliberate whitelist. Never copy the rollout's gold-derived fields.
    observation = initial.get('observation') or {}
    return str(initial.get('text', '')), {
        k: observation[k] for k in ('status', 'error', 'columns', 'rows')
        if k in observation
    }


def _metadata(con):
    result = {}
    names = con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    for (name,) in names:
        columns = {}
        for row in con.execute(f'PRAGMA table_xinfo({_quote(name)})'):
            columns[_key(row[1])] = {'name': row[1], 'type': row[2],
                                    'declared_not_null': bool(row[3]),
                                    'primary_key_position': int(row[5])}
        fks = []
        for row in con.execute(f'PRAGMA foreign_key_list({_quote(name)})'):
            fks.append({'id': row[0], 'sequence': row[1], 'from_column': row[3],
                        'to_table': row[2], 'to_column': row[4]})
        unique = []
        for row in con.execute(f'PRAGMA index_list({_quote(name)})'):
            if row[2]:
                index_columns = [x[2] for x in con.execute(f'PRAGMA index_info({_quote(row[1])})')]
                unique.append({'columns': index_columns, 'partial': bool(row[4]),
                               'expression_index': any(x is None for x in index_columns)})
        result[_key(name)] = {'name': name, 'columns': columns,
                             'foreign_keys': fks, 'unique_indexes': unique}
    return result


def _sources_and_values(tree, catalog):
    """Resolve base-table references per SQL scope; skip unresolved CTE outputs."""
    referenced, column_resolution = [], {}
    if tree is None:
        return referenced, []
    for scope in traverse_scope(tree):
        physical_sources = {}
        for alias, source in scope.sources.items():
            if isinstance(source, exp.Table):
                table = _key(source.name)
                physical_sources[_key(alias)] = table
                if table not in referenced:
                    referenced.append(table)
        for column in scope.columns:
            if column.table:
                table = physical_sources.get(_key(column.table))
                candidates = [table] if table in catalog else []
            else:
                candidates = list({t for t in physical_sources.values()
                                   if t in catalog and _key(column.name) in catalog[t]['columns']})
            if len(candidates) == 1 and _key(column.name) in catalog[candidates[0]]['columns']:
                column_resolution[id(column)] = (candidates[0], _key(column.name))
    values = []
    for node in tree.walk():
        pairs = []
        if isinstance(node, exp.EQ):
            pairs = [(node.left, node.right), (node.right, node.left)]
        elif isinstance(node, exp.In) and not node.args.get('query'):
            pairs = [(node.this, val) for val in node.expressions]
        for column, literal in pairs:
            if (isinstance(column, exp.Column) and isinstance(literal, exp.Literal)
                    and literal.is_string and id(column) in column_resolution):
                table, field = column_resolution[id(column)]
                item = (table, field, literal.this, 'SQL equality/IN literal')
                if item not in values and len(literal.this) <= 120:
                    values.append(item)
    return referenced, values


def _quoted_terms(question):
    # Quoted request terms are hypotheses to look up, not known database values.
    terms = re.findall(r'"([^"\n]{1,80})"|“([^”\n]{1,80})”|‘([^’\n]{1,80})’|(?<!\w)\x27([^\x27\n]{1,80})\x27(?!\w)', question)
    return list(dict.fromkeys(next(x for x in group if x) for group in terms))[:3]


def _suggestions(name, choices):
    plain = name.rsplit('.', 1)[-1].strip('"`[]').casefold()
    scored = [(SequenceMatcher(None, plain, _key(candidate.rsplit('.', 1)[-1])).ratio(), candidate)
              for candidate in choices]
    return [candidate for score, candidate in sorted(scored, key=lambda x: (-x[0], x[1]))[:3]
            if score >= 0.55]


def _fk_paths(catalog, selected):
    graph = {table: [] for table in catalog}
    for table, meta in catalog.items():
        grouped = {}
        for fk in meta['foreign_keys']:
            grouped.setdefault(fk['id'], []).append(fk)
        for fk_id, parts in grouped.items():
            parts.sort(key=lambda p: p['sequence'])
            target = _key(parts[0]['to_table'])
            if target not in graph:
                continue
            edge = {'from_table': meta['name'], 'to_table': parts[0]['to_table'],
                    'from_columns': [p['from_column'] for p in parts],
                    'to_columns': [p['to_column'] for p in parts], 'constraint_id': fk_id}
            graph[table].append((target, edge))
            graph[target].append((table, edge))
    result = []
    for pos, source in enumerate(selected):
        for target in selected[pos + 1:]:
            queue, seen = deque([(source, [])]), {source}
            while queue:
                node, path = queue.popleft()
                if node == target:
                    if path and path not in result:
                        result.append(path)
                    break
                if len(path) >= 3:
                    continue
                for nxt, edge in graph[node]:
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append((nxt, path + [edge]))
    return result[:2]


def _format_fact(fact):
    kind = fact['kind']
    if kind == 'execution':
        if fact['status'] == 'error':
            return 'Observed SQLite error: ' + fact['error']
        return f"SQLite executed; returned {fact['row_count']} rows, columns {fact['columns']}. This does not certify correctness."
    if kind == 'unknown_identifier':
        return f"Unresolved {fact['identifier_type']}: {fact['identifier']}; schema spelling candidates (not a correction): {fact['candidates']}."
    if kind == 'table_schema':
        columns = ', '.join(c['name'] + ':' + (c['type'] or 'unspecified')
                            + (' NOT NULL(declared)' if c['declared_not_null'] else '')
                            + (f" PK#{c['primary_key_position']}" if c['primary_key_position'] else '')
                            for c in fact['columns'])
        return f"Schema {fact['table']}({columns})."
    if kind == 'foreign_key_path':
        return 'Declared FK path (not proof of intended join): ' + '; '.join(
            f"{e['from_table']}({','.join(e['from_columns'])}) -> {e['to_table']}({','.join(c or '[referenced PK]' for c in e['to_columns'])})"
            for e in fact['edges']) + '.'
    if kind == 'unique_constraint':
        return f"Declared unique index on {fact['table']}({', '.join(str(c) for c in fact['columns'])}); partial={fact['partial']}, expression={fact['expression_index']}."
    if kind == 'literal_probe':
        literal = json.dumps(fact['value'], ensure_ascii=False)
        if fact['status'] == 'ok':
            result = 'at least one match' if fact['found'] else 'no match'
            return f"Unfiltered column probe: {fact['table']}.{fact['column']} = {literal}: {result} under SQLite column affinity/collation."
        return f"Column probe {fact['table']}.{fact['column']} timed out/failed; no evidence obtained."
    if kind == 'small_column_domain':
        return f"Complete observed non-NULL column domain under SQLite DISTINCT, {fact['table']}.{fact['column']}: {json.dumps(fact['values'], ensure_ascii=False)}. This does not identify the intended value."
    return ''


def compile_evidence(task, initial, max_tokens_hint=256):
    """Return {text, facts, diagnostics, probe_seconds} without label access.

    facts contain structured, scoped observations; text chooses whole facts
    within approximately 4*max_tokens_hint characters. All probes are bounded
    read-only existence checks and complete small column domains, never
    arbitrary row samples. Domain checks only follow an absent SQL literal;
    a 9-value cutoff rejects domains larger than 8 observed non-NULL values.
    Numeric diagnostics are safe features for a learned router. The source SQL
    and user task remain the caller's responsibility when constructing a new
    user message; this module creates no assistant-history messages.
    """
    started = time.perf_counter()
    text, observation = _public_initial(initial)
    sql = _sql(text)
    question = str(task.get('question', ''))
    db_path = task['db_path']
    diagnostics = dict(parse_ok=0, execution_ok=int(observation.get('status') == 'ok'),
                       execution_error=int(observation.get('status') == 'error'),
                       zero_rows=int(observation.get('status') == 'ok' and not observation.get('rows')),
                       unknown_table_count=0, unknown_column_count=0, referenced_table_count=0,
                       selected_table_count=0, fk_path_count=0, probe_attempts=0,
                       probe_found_count=0, probe_zero_count=0, probe_failed_count=0,
                       domain_probe_attempts=0, complete_domain_count=0,
                       metadata_failed=0, parse_failed=0, text_omitted_facts=0)
    facts = []
    if observation.get('status') == 'error':
        facts.append({'kind': 'execution', 'status': 'error', 'error': str(observation.get('error', 'unknown'))[:300]})
    elif observation.get('status') == 'ok':
        facts.append({'kind': 'execution', 'status': 'ok', 'row_count': len(observation.get('rows', [])),
                      'columns': observation.get('columns', [])})
    try:
        statements = sqlglot.parse(sql, read='sqlite')
        tree = statements[0] if len(statements) == 1 else None
        diagnostics['parse_ok'] = int(tree is not None)
        diagnostics['parse_failed'] = int(tree is None)
    except (sqlglot.errors.ParseError, ValueError):
        tree = None
        diagnostics['parse_failed'] = 1
    probe_seconds = 0.0
    con = None
    try:
        con = sqlite3.connect(Path(db_path).resolve().as_uri() + '?mode=ro', uri=True, timeout=0.1)
        con.execute('PRAGMA query_only=ON')
        con.set_progress_handler(lambda: int(time.perf_counter() - started > TOTAL_TIMEOUT), 1000)
        catalog = _metadata(con)
        referenced, probes = _sources_and_values(tree, catalog)
        diagnostics['referenced_table_count'] = len(referenced)
        unknown = [t for t in referenced if t not in catalog]
        diagnostics['unknown_table_count'] = len(unknown)
        for name in unknown:
            facts.append({'kind': 'unknown_identifier', 'identifier_type': 'table', 'identifier': name,
                          'candidates': _suggestions(name, [m['name'] for m in catalog.values()])})
        known_refs = [t for t in referenced if t in catalog]
        words = _words(question + ' ' + str(task.get('evidence', '')))
        def table_score(table):
            meta = catalog[table]
            return (8 * int(table in known_refs) + 3 * len(_words(table) & words)
                    + sum(bool(_words(c['name']) & words) for c in meta['columns'].values()))
        # Do not probe unrelated lexicographic tables when nothing matches.
        selected = [t for t in sorted(catalog, key=lambda t: (-table_score(t), t)) if table_score(t) > 0][:MAX_TABLES]
        diagnostics['selected_table_count'] = len(selected)
        error = str(observation.get('error', ''))
        match = re.search(r'no such column:\s*(.+)', error, re.I)
        if match:
            identifier = match.group(1).strip()
            diagnostics['unknown_column_count'] = 1
            choices = {catalog[t]['name'] + '.' + c['name'] for t in (known_refs or selected)
                       for c in catalog[t]['columns'].values()}
            facts.append({'kind': 'unknown_identifier', 'identifier_type': 'column', 'identifier': identifier,
                          'candidates': _suggestions(identifier, choices)})
        # Literal probes precede complete schema in the text budget; base prompts
        # normally already include the full schema.
        for value in _quoted_terms(question):
            fields = [(len(_words(c['name']) & words), t, key)
                      for t in selected for key, c in catalog[t]['columns'].items()
                      if any(kind in c['type'].upper() for kind in ('CHAR', 'TEXT', 'CLOB'))]
            for _, table, field in sorted(fields, key=lambda x: (-x[0], x[1], x[2]))[:2]:
                candidate = (table, field, value, 'quoted request term')
                if not any(p[:3] == candidate[:3] for p in probes):
                    probes.append(candidate)
        for table, field, value, source in probes[:MAX_PROBES]:
            if time.perf_counter() - started >= TOTAL_TIMEOUT or diagnostics['probe_attempts'] >= MAX_PROBES:
                break
            meta, col = catalog[table], catalog[table]['columns'][field]
            before = time.perf_counter()
            con.set_progress_handler(lambda: int(time.perf_counter() - before > PROBE_TIMEOUT
                                                 or time.perf_counter() - started > TOTAL_TIMEOUT), 1000)
            diagnostics['probe_attempts'] += 1
            fact = {'kind': 'literal_probe', 'table': meta['name'], 'column': col['name'],
                    'value': value, 'source': source, 'scope': 'unfiltered base column; SQLite affinity/collation'}
            try:
                found = con.execute(f'SELECT 1 FROM {_quote(meta["name"])} WHERE {_quote(col["name"])} = ? LIMIT 1', (value,)).fetchone() is not None
                fact.update(status='ok', found=found)
                diagnostics['probe_found_count' if found else 'probe_zero_count'] += 1
            except sqlite3.Error:
                fact.update(status='unavailable', found=None)
                diagnostics['probe_failed_count'] += 1
            elapsed = time.perf_counter() - before
            probe_seconds += elapsed
            fact['seconds'] = elapsed
            facts.append(fact)
            if (fact['status'] == 'ok' and fact['found'] is False and source.startswith('SQL')
                    and diagnostics['probe_attempts'] < MAX_PROBES
                    and time.perf_counter() - started < TOTAL_TIMEOUT):
                # Enumerate only genuinely small domains, not an arbitrary
                # sample of rows/values. A ninth value makes it incomplete.
                before = time.perf_counter()
                con.set_progress_handler(lambda: int(time.perf_counter() - before > PROBE_TIMEOUT
                                                     or time.perf_counter() - started > TOTAL_TIMEOUT), 1000)
                diagnostics['probe_attempts'] += 1
                diagnostics['domain_probe_attempts'] += 1
                try:
                    domain = [r[0] for r in con.execute(
                        f'SELECT DISTINCT {_quote(col["name"])} FROM {_quote(meta["name"])} '
                        f'WHERE {_quote(col["name"])} IS NOT NULL LIMIT 9')]
                    if len(domain) <= 8 and all(isinstance(v, (str, int, float)) and len(str(v)) <= 120 for v in domain):
                        facts.append({'kind': 'small_column_domain', 'table': meta['name'],
                                      'column': col['name'], 'values': domain,
                                      'scope': 'complete observed non-NULL SQLite DISTINCT domain; not allowed values or intended answer'})
                        diagnostics['complete_domain_count'] += 1
                except sqlite3.Error:
                    diagnostics['probe_failed_count'] += 1
                probe_seconds += time.perf_counter() - before
        paths = _fk_paths(catalog, selected)
        diagnostics['fk_path_count'] = len(paths)
        facts.extend({'kind': 'foreign_key_path', 'edges': path} for path in paths)
        for table in selected:
            meta = catalog[table]
            for unique in meta['unique_indexes'][:2]:
                facts.append({'kind': 'unique_constraint', 'table': meta['name'], **unique})
            facts.append({'kind': 'table_schema', 'table': meta['name'], 'columns': list(meta['columns'].values()),
                          'scope': 'SQLite declared schema only; absent NOT NULL is not proof that NULL is possible; FK metadata is not data-integrity validation'})
    except (sqlite3.Error, OSError, ValueError, sqlglot.errors.OptimizeError) as error:
        diagnostics['metadata_failed'] = 1
        # Do not echo filesystem paths or the exception as presumed query facts.
        facts.append({'kind': 'metadata_unavailable', 'reason': type(error).__name__})
    finally:
        if con is not None:
            con.close()
    qualifier = ('Scoped database facts; zero matches/empty output do not prove an error. '
                 'Constraints are declared metadata, not integrity checks; absence of NOT NULL does not prove nullability.')
    budget = max(256, int(max_tokens_hint) * 4)
    lines = [qualifier]
    used, shown = len(qualifier), 0
    for fact in facts:
        line = _format_fact(fact)
        if line and used + len(line) + 1 <= budget:
            lines.append(line)
            used += len(line) + 1
            shown += 1
    diagnostics['text_omitted_facts'] = len(facts) - shown
    diagnostics['text_chars'] = used
    diagnostics['compile_seconds'] = time.perf_counter() - started
    return {'text': '\n'.join(lines), 'facts': facts,
            'diagnostics': diagnostics, 'probe_seconds': probe_seconds}
