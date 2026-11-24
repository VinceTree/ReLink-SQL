"""Pinned paper components and SQL-domain adaptations, public information only."""
import ast, json
from pathlib import Path
from iteration4 import collect_paper_baselines as paper
from iteration8 import input_selectors as selectors
from iteration3.repair_index import _schema_index
from iteration4.collect_confirmation import initial_messages

HERE = Path(__file__).resolve().parent


def rsl_instruction():
    tree = ast.parse((HERE/'sources/rsl_Instruction.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='TABLE_AUG_INSTRUCTION' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing pinned RSL instruction')


def rsl_messages(task, full):
    user = ('### Sqlite SQL tables, with their properties:\n'+full['schema_text']+
            '\n\n### definition: '+task.get('evidence','').strip()+
            '\n### Question: '+task['question'].strip())
    return [dict(role='system',content=rsl_instruction()),dict(role='user',content=user)]


def rsl_select(task, weak, raw, full):
    decoder = json.JSONDecoder(); obj = None
    for i,c in enumerate(raw):
        if c != '{':continue
        try:
            value,_ = decoder.raw_decode(raw[i:])
            if isinstance(value,dict) and isinstance(value.get('columns'),list):obj=value;break
        except json.JSONDecodeError:pass
    tables = _schema_index(task['db_path'])[0]['tables']
    physical = {t['name'].lower()+'.'+c['name'].lower():(t['name'],c['name'])
                for t in tables.values() for c in t['columns'].values()}
    selected = {v.replace('`','').lower() for v in obj['columns'] if isinstance(v,str)} if obj else set()
    valid = selected & set(physical)
    if not valid:
        return dict(full, selection_fallback='invalid_or_empty_model_columns',
                    invalid_model_columns=sorted(selected-set(physical)), model_columns=0, union_columns=0)
    # Original RSL finds each physical column-name substring globally, without
    # resolving SQL aliases. Keep this mechanism rather than our own binder.
    sql = weak['text'].lower(); hint = task.get('evidence','').lower()
    combined = valid | {k for k,(t,c) in physical.items() if c.lower() in sql or c.lower() in hint}
    chosen = {}
    for k in sorted(combined):
        t,c = physical[k];chosen.setdefault(t,[]).append(c)
    return dict(selectors.render(task,chosen,preserve_keys=True),
                selection_fallback=None,invalid_model_columns=sorted(selected-set(physical)),
                model_columns=len(valid),union_columns=len(combined))


def messages(task, weak, full, method, stage='answer', rnd=1, previous=None, feedback=None):
    sql,obs,history = weak['text'],weak['observation'],[]
    previous = previous or {}
    for j in range(1,rnd):
        prior=previous[j]
        history.append('Previous SQL:\n'+sql+'\nFeedback:\n'+prior['feedback'])
        if method!='critic' or prior['observation']['status']=='ok':
            sql,obs=prior['text'],prior['observation']
    values=dict(schema=full['schema_text'],question=task['question'],evidence=task.get('evidence',''),
                sql=sql,observation=paper.public_observation(obs),history='\n\n'.join(history))
    if method.startswith('din_'):
        values.update(paper.din_schema(task))
        values['question']=task['question']+('\nEvidence: '+task['evidence'] if task.get('evidence') else '')
        name='din_sql_gentle.txt' if method=='din_gentle' else 'din_sql_generic_figure6.txt'
    else:
        assert method in ['self_refine','critic']
        if stage=='answer':
            assert isinstance(feedback,str);values['feedback']=feedback
        name=method+'_sql_'+('feedback' if stage=='feedback' else 'refine')+'_adapted.txt'
    return [dict(role='user',content=paper.render(name,values))]


def sql_messages(task, context):
    return initial_messages(dict(task,schema=context['schema_text']))
