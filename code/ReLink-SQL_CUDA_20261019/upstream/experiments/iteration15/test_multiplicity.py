import json
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.iteration15 import multiplicity as m


def setup(tmp_path,sql):
    db=tmp_path/'toy.sqlite'
    with sqlite3.connect(db) as c:
        c.executescript('CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT, n INTEGER);'
                        "INSERT INTO t VALUES(1,'a',2),(2,'a',3),(3,'b',4);")
    task=dict(id='toy',db_path=str(db),question='List names.',evidence='')
    observation=m.r.execute(str(db),sql)
    history=[dict(event='old',sql=sql,observation=observation)]
    return task,history


def plan(**extra):
    return json.dumps(dict(meaning='List names',outputs=[['t','name']],filters=[],
                          aggregates=[],grain='rows',distinct='no',**extra))


def test_result_multiplicity_and_rejection_skip(tmp_path):
    task,h=setup(tmp_path,'SELECT name FROM t')
    sql,meta=m.propose(task,h,plan())
    assert meta['operations']==['result_multiplicity']
    assert m.r.execute(task['db_path'],sql)['rows']==[('a',),('b',)]
    h.append(dict(event='also_rejected',sql=sql,observation=m.r.execute(task['db_path'],sql)))
    sql,meta=m.propose(task,h,plan())
    assert sql is None
    assert any(x['status']=='known_rejected_result' for x in meta['search'])


def test_count_changes_when_result_distinct_does_not(tmp_path):
    task,h=setup(tmp_path,'SELECT COUNT(name) FROM t')
    text=json.dumps(dict(meaning='Count different names',outputs=[['t','name']],
                         aggregates=['count'],grain='single',distinct='no'))
    sql,meta=m.propose(task,h,text)
    assert meta['operations']==['count_multiplicity']
    assert m.r.execute(task['db_path'],sql)['rows']==[(2,)]
    assert len(meta['search'])==2


def test_projection_fallback_and_ambiguous_entity_count(tmp_path):
    task,h=setup(tmp_path,'SELECT * FROM t')
    sql,meta=m.propose(task,h,plan())
    assert meta['operations']==['bound_requirements']
    assert m.r.execute(task['db_path'],sql)['rows']==[('a',),('a',),('b',)]
    parsed=m.contract.parse(plan(),task)
    assert m.count_alternative(task,'SELECT COUNT(*) FROM t a JOIN t b ON a.id=b.id',parsed) is None


def test_malformed_query_skips_count_rewrite(tmp_path):
    from experiments.iteration15 import multiplicity_safe
    task,h=setup(tmp_path,'SELECT name FROM t')
    bad="SELECT COUNT(*) FROM t WHERE name = 'Women\\'s Soccer'"
    assert multiplicity_safe.count_alternative(task,bad,m.contract.parse(plan(),task)) is None
