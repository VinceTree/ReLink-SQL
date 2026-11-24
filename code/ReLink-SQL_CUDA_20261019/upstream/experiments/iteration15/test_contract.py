import json
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from iteration15 import contract as m


def task(tmp_path):
    p=tmp_path/'contract.sqlite'
    with sqlite3.connect(p) as db:
        db.executescript('CREATE TABLE t(id INTEGER, name TEXT); CREATE TABLE u(id INTEGER); INSERT INTO t VALUES (1,"Silverstone Circuit"),(2,"Other");')
    return dict(db_path=str(p),question='Find Silverstone',evidence='')


def test_plan_validation_and_unsupported_value(tmp_path):
    t=task(tmp_path)
    p=m.parse(json.dumps(dict(meaning='names',outputs=[['t','name'],['fake','id']],grain=[],
        inspect=[dict(field=['t','name'],target='invented')])),t)
    assert p['fields']==[['t','name']] and p['grain']=='unknown'
    assert not p['valid'] and not p['inspect']
    assert not m.parse('invalid',t)['aggregates_specified']


def test_scope_bindings_and_window_grain(tmp_path):
    t=task(tmp_path)
    a=m.profile(t,'SELECT a.name, COUNT(*) OVER () AS n FROM t a')
    assert a['grain']=='rows' and ('t','name') in a['fields']
    assert m.profile(t,'SELECT name, COUNT(*) FROM t GROUP BY name')['grain']=='grouped'
    assert m.profile(t,'SELECT COUNT(*) FROM t')['grain']=='single'
    ambiguous=m.profile(t,'SELECT id FROM t JOIN u ON t.id=u.id')
    assert 'id' in ambiguous['unresolved']


def test_anchor_uses_requirements_and_value_probe(tmp_path):
    t=task(tmp_path)
    p=m.parse(json.dumps(dict(meaning='names',outputs=[['t','name']],aggregates=[],grain='rows',
                             inspect=[dict(field=['t','name'],target='Silverstone')])),t)
    history=[dict(event='initial',sql='SELECT id FROM t',observation=dict(status='ok')),
             dict(event='repair',sql='SELECT name FROM t',observation=dict(status='ok')),
             dict(event='latest',sql='SELECT COUNT(*) FROM t',observation=dict(status='ok'))]
    selected,_=m.align(t,history,p)
    assert selected['event']=='repair'
    out=m.inspect_value(t['db_path'],'t','name','Silverstone')
    assert out['observation']['rows']==[('Silverstone Circuit',)]
    assert m.profile(t,'SELECT * FROM t')['output_fields']==[('t','id'),('t','name')]
    history[2]['sql']='SELECT id,name FROM t'
    selected,candidates=m.align(t,history,p)
    assert selected['event']=='repair' and candidates[-1]['extra_output_fields']==[('t','id')]


def test_requirement_prompt_is_independent_of_old_sql(tmp_path):
    t=dict(task(tmp_path),id='local')
    state=dict(task_id='local',method='EXEC14',round=1,sql='SELECT id FROM t',
        observation=dict(status='ok',columns=['id'],rows=[[1]]),rejected_sqls=['SELECT id FROM t'],
        feedback=m.r.methods.NEGATIVE,critiques=[])
    before=m.plan_messages(t,state)
    changed=dict(state,sql='SELECT name FROM t',rejected_sqls=['SELECT name FROM t'],
                 observation=dict(status='ok',columns=['name'],rows=[['Other']]))
    assert m.plan_messages(t,changed)==before
