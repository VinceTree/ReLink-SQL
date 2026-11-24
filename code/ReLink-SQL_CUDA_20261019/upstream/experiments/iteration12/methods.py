"""Public-only feedback adapters; original method modules remain unchanged."""
from pathlib import Path

from iteration2 import share_adapter as share
from iteration4 import collect_paper_baselines as paper
from iteration4.collect_confirmation import initial_messages, MODELS
from iteration8 import input_selectors as selectors
from iteration10 import methods as previous
from iteration11.contexts import recovery_context

METHODS = ['RETRY7', 'RETRY14', 'EXEC14', 'RELINK14', 'DIN_GENTLE',
           'SELF_REFINE', 'CRITIC', 'CHESS_C', 'RSL_LINK', 'SHARE_META']
TASK_FIELDS = {'id', 'dataset', 'group', 'split', 'kind', 'question', 'evidence', 'db_path', 'schema'}
STATE_FIELDS = {'task_id', 'method', 'round', 'sql', 'observation', 'rejected_sqls',
                'feedback', 'critiques'}
NEGATIVE = 'The result is incorrect. Please correct it.'


def validate(task, state):
    if set(task) - TASK_FIELDS or set(state) != STATE_FIELDS:
        raise ValueError('Public task/state fields only')
    if state['task_id'] != task['id'] or state['method'] not in METHODS:
        raise ValueError('Mismatched task/method')
    if state['feedback'] != NEGATIVE or not state['rejected_sqls']:
        raise ValueError('Only rejected answers may enter repair')
    if state['rejected_sqls'][-1] != state['sql']:
        raise ValueError('Current SQL must be the latest rejected answer')
    allowed = {'status', 'error', 'columns', 'rows', 'seconds', 'sql'}
    if set(state['observation']) - allowed:
        raise ValueError('Unexpected observation information')


def attach(messages, state, current_present=True):
    messages = [dict(m) for m in messages]
    prefix = 'User feedback on the current SQL: ' + state['feedback'] + '\n'
    older = state['rejected_sqls'][:-1]
    if older:
        prefix += '\nEarlier rejected SQL attempts:\n' + '\n\n'.join(older) + '\n'
    if not current_present:
        prefix += '\nCurrent rejected SQL:\n' + state['sql'] + '\n'
    user = next(m for m in messages if m['role'] == 'user')
    user['content'] = prefix + '\n' + user['content']
    return messages


def initial_stage(method):
    if method in ('SELF_REFINE', 'CRITIC'):
        return 'feedback'
    if method in ('CHESS_C', 'RSL_LINK'):
        return 'selection'
    return 'answer'


def model(method):
    return MODELS['strong' if method == 'RETRY7' else 'strong14']


def build(task, state, full, stage, auxiliary=None):
    validate(task, state)
    method = state['method']
    current = dict(text=state['sql'], raw_text=state['sql'], observation=state['observation'])
    context = full
    metadata = {}
    if method in ('RETRY7', 'RETRY14', 'EXEC14', 'RELINK14'):
        if method == 'RELINK14':
            context = recovery_context(task, current)
            metadata = context
        messages = initial_messages(dict(task, schema=context['schema_text']))
        messages[-1]['content'] += '\n\nCurrent SQL to correct:\n' + state['sql']
        if method in ('EXEC14', 'RELINK14'):
            messages[-1]['content'] += '\n\nExecution report:\n' + paper.public_observation(state['observation'])
    elif method == 'DIN_GENTLE':
        messages = previous.messages(task, current, full, 'din_gentle')
    elif method in ('SELF_REFINE', 'CRITIC'):
        name = method.lower()
        values = dict(schema=full['schema_text'], question=task['question'], evidence=task.get('evidence', ''),
                      sql=state['sql'], observation=paper.public_observation(state['observation']),
                      history='\n\n'.join(state['critiques']))
        if stage == 'answer':
            if not isinstance(auxiliary, str):
                raise ValueError('Missing paid critique')
            values['feedback'] = auxiliary
        template = name + '_sql_' + ('feedback' if stage == 'feedback' else 'refine') + '_adapted.txt'
        messages = [dict(role='user', content=paper.render(template, values))]
    elif method in ('CHESS_C', 'RSL_LINK'):
        if stage == 'selection':
            messages = (selectors.chess_messages(task, full['schema_text'], 'columns')
                        if method == 'CHESS_C' else previous.rsl_messages(task, full))
            return attach(messages, state, current_present=False), metadata
        if not isinstance(auxiliary, str):
            raise ValueError('Missing paid schema selection')
        context = (selectors.chess_selection(task, auxiliary, 'columns', full)
                   if method == 'CHESS_C' else previous.rsl_select(task, current, auxiliary, full))
        metadata = context
        messages = initial_messages(dict(task, schema=context['schema_text']))
        return attach(messages, state, current_present=False), metadata
    else:
        raise ValueError(method)
    return attach(messages, state), metadata


def decode(raw, method):
    return paper.decode_sql(raw, 'din_gentle' if method == 'DIN_GENTLE' else 'plain')


def share_messages(task, state, stage, info, previous_outputs):
    validate(task, state)
    messages, metadata = share.stage_messages(stage, dict(task, db_id=task['group']), state['sql'],
                                             info, previous_outputs, {}, share.read_templates())
    if stage in share.STAGES[2:]:
        messages = attach(messages, state, current_present=False)
    return messages, metadata
