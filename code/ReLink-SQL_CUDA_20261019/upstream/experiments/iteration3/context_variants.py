"""Explicit context ablations; frozen v1 index remains unchanged."""
import time
from iteration3.repair_index import build_repair_context as base_context

def build_repair_context(task, initial, mode='closure', token_budget=768):
    if mode == 'fault':
        from iteration3.fault_context import build_fault_context
        return build_fault_context(task, initial, token_budget=token_budget)
    if mode in ('full', 'flat', 'closure'):
        return base_context(task, initial, mode=mode, token_budget=token_budget)
    if mode in ('closure_columns', 'all_columns'):
        from iteration3.column_context import build_column_context
        return build_column_context(task, initial, mode=mode, token_budget=token_budget)
    if mode != 'seeds':
        raise ValueError(mode)
    started = time.perf_counter()
    context = base_context(task, initial, mode='closure', token_budget=token_budget)
    selection = context['index']['selection']
    names = {r['table'] for r in selection['ranked_seeds']}
    nodes = [n for n in context['index']['G']['nodes']
             if n['kind'] in ('table','view') and n['name'].casefold() in names]
    context['schema_text'] = '\n\n'.join(n['ddl'] for n in sorted(nodes,key=lambda n:n['name'].casefold()))
    context['selected_tables'] = [n['name'] for n in nodes]
    context['selected_columns'] = {k:v for k,v in context['selected_columns'].items() if k.casefold() in names}
    selection['selected_tables'] = sorted(names)
    selection['omitted_seeds'] = []
    context['diagnostics'].update(mode='seeds', bridge_table_count=0,
        selected_table_count=len(names), schema_chars=len(context['schema_text']),
        estimated_schema_tokens=(len(context['schema_text'])+3)//4,
        overflow=len(context['schema_text']) > token_budget*4,
        ablation='Exactly the closure seeds; omit only FK bridge tables; no budget deletion')
    context['seconds'] = time.perf_counter()-started
    context['cost']['total_seconds'] = context['seconds']
    return context
