"""Public, question-grounded first inputs and observed recovery state features."""
from collections import Counter
import math
import re

from iteration3 import repair_index as ri
from iteration8 import input_selectors as old

VERSION = 'first_relation_skeleton_v1'
SEED_TABLES = 3


def terms(text):
    words = ri._words(text)
    # A uniform inflection rule; no dataset-specific entity dictionaries.
    return {w[:-1] if len(w) > 3 and w.endswith('s') and not w.endswith('ss') else w
            for w in words}


def first_context(task):
    index = ri._schema_index(task['db_path'])[0]
    tables = index['tables']
    query = terms(task['question'] + ' ' + task.get('evidence', ''))
    docs = {t: terms(m['name']) | set().union(*(terms(c) for c in m['columns']))
            for t, m in tables.items()}
    df = Counter(w for words in docs.values() for w in words)
    weight = {w: math.log(1 + (len(tables) + 1) / (df[w] + 1)) for w in df}
    scores = {t: sum(weight[w] for w in query & words)
              + 2 * sum(weight.get(w, 0.) for w in query & terms(t))
              for t, words in docs.items()}
    seeds = [t for t in sorted(tables, key=lambda t: (-scores[t], t))
             if scores[t] > 0][:SEED_TABLES]
    full = old.full(task)
    if not seeds:
        return dict(full, version=VERSION, seeds=[], first_fallback='no_question_match',
                    table_scores=scores)
    selected = {}
    for t, m in tables.items():
        if t in seeds:
            selected[t] = list(m['columns'])
        else:
            keep = {c for c, v in m['columns'].items() if v['pk_position'] or terms(c) & query}
            for edge in index['fk_edges']:
                if not edge['resolved']:
                    continue
                if edge['source_table'] == t:
                    keep.update(ri._key(c) for c in edge['source_columns'])
                if edge['target_table'] == t:
                    keep.update(ri._key(c) for c in edge['target_columns'])
            selected[t] = sorted(keep) if keep else list(m['columns'])
    ctx = old.render(task, selected, preserve_keys=True)
    return dict(ctx, version=VERSION, seeds=seeds, first_fallback=None, table_scores=scores)


def chosen_pairs(context):
    return {(ri._key(t), ri._key(c)) for t, cols in context['selected_columns'].items() for c in cols}


def root_features(task, first, full, model_key, prompt_tokens):
    index = ri._schema_index(task['db_path'])[0]
    tables = index['tables']
    selected = chosen_pairs(first)
    all_pairs = chosen_pairs(full)
    question = (task['question'] + ' ' + task.get('evidence', '')).lower()
    values = dict(
        model_3=float(model_key == '3'), model_7=float(model_key == '7'), model_14=float(model_key == '14'),
        first_full=float(selected == all_pairs), prompt_tokens=float(prompt_tokens),
        table_count=float(len(tables)), column_count=float(len(all_pairs)),
        selected_columns=float(len(selected)), selected_fraction=len(selected) / max(1, len(all_pairs)),
        fk_count=float(sum(e['resolved'] for e in index['fk_edges'])),
        composite_fk_count=float(sum(e['resolved'] and len(e['source_columns']) > 1 for e in index['fk_edges'])),
        question_words=float(len(terms(question))), evidence_chars=float(len(task.get('evidence', ''))),
        # Only request operations, not database/task identities or reference SQL.
        aggregation=float(bool(re.search(r'average|total|count|number|sum|maximum|minimum', question))),
        comparison=float(bool(re.search(r'more than|less than|greater|higher|lower|at least|between', question))),
        ordering=float(bool(re.search(r'most|least|highest|lowest|top|descending|ascending', question))),
        negation=float(bool(re.search(r'\bnot\b|without|never|except', question))),
    )
    # Public graph statistics, with relation completeness independent of naming.
    degrees = Counter()
    for e in index['fk_edges']:
        if e['resolved']:
            degrees[e['source_table']] += 1
            degrees[e['target_table']] += 1
    values.update(fk_degree_max=float(max(degrees.values(), default=0)),
                  keyless_tables=float(sum(not any(c['pk_position'] for c in t['columns'].values())
                                           for t in tables.values())))
    return values


def recovery_context(task, record):
    # The existing ReLink-SQL generator receives no first SQL or long critique.
    try:
        ctx = old.current(task, record)
    except RecursionError:
        # Generated SQL can exceed the parser's recursion limit. Apply the
        # same conservative full-context behavior to every such failure.
        ctx = dict(old.full(task), diagnostics=dict(parse_ok=False,
                   global_coverage_fallback=True, parser_recursion_fallback=True))
    # Store deterministic evidence only; discard wall-clock/cache-hit measurements.
    diag = ctx.get('diagnostics', {})
    stable = {k: v for k, v in diag.items()
              if 'seconds' not in k and k not in ('schema_cache_hit',)}
    return dict(schema_text=ctx['schema_text'], selected_columns=ctx['selected_columns'],
                diagnostics=stable, bindings=ctx.get('index', {}).get('M', []))


def observed_features(first_record, first_context, recovery):
    obs = first_record['observation']
    diag = recovery['diagnostics']
    old_pairs = chosen_pairs(first_context)
    new_pairs = chosen_pairs(recovery)
    values = dict(
        exec_ok=float(obs['status'] == 'ok'),
        empty_result=float(obs['status'] == 'ok' and len(obs.get('rows', [])) == 0),
        output_tokens=float(first_record['generation_tokens']),
        sql_chars=float(len(first_record['raw_text'])),
        repair_columns=float(len(new_pairs)),
        added_columns=float(len(new_pairs - old_pairs)),
        reused_columns=float(len(new_pairs & old_pairs)),
        removed_columns=float(len(old_pairs - new_pairs)),
        repair_schema_chars=float(len(recovery['schema_text'])),
    )
    for name in ('parse_ok', 'scope_count', 'required_table_count', 'seed_table_count',
                 'bridge_table_count', 'missing_fk_path_count', 'ambiguous_fk_path_count',
                 'ambiguous_binding_count', 'column_binding_count', 'scope_binding_coverage',
                 'global_coverage_fallback', 'retained_mapped_reference_coverage'):
        values['structure_' + name] = float(diag.get(name, 0))
    for name in ('logprob_mean', 'logprob_min', 'logprob_q10', 'low_confidence_fraction'):
        val = first_record.get(name)
        values[name] = float(val) if val is not None and math.isfinite(val) else 0.
    return values
