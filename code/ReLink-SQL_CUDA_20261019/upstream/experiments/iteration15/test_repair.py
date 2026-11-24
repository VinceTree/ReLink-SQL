"""Checks for result evidence and SQL intervention boundaries."""
import sqlite3
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from iteration15 import repair as model


def observed(rows,columns=('x',)):
    return dict(status='ok',columns=list(columns),rows=rows)


def test_equivalence_uses_full_typed_ordered_result():
    a=observed([[1],[2],[3],[4]])
    assert model.result_key(a)!=model.result_key(observed([[1],[2],[3],[5]]))
    assert model.result_key(a)!=model.result_key(observed([[4],[3],[2],[1]]))
    assert model.result_key(observed([[1]]))!=model.result_key(observed([[1.0]]))
    assert model.result_key(observed([[1],[1]]))!=model.result_key(observed([[1]]))
    assert model.result_key(observed([],('a',)))!=model.result_key(observed([],('a','b')))
    assert model.result_key(dict(status='error',error='row limit')) is None
    assert model.result_key(observed([[1]],('alias',)))==model.result_key(observed([[1]],('x',)))


def test_history_does_not_condemn_unchanged_parts():
    history=[dict(event='first',sql='SELECT x FROM t WHERE x>0',observation=observed([[1]])),
             dict(event='second',sql='SELECT x FROM t WHERE x>=1',observation=observed([[1]]))]
    facts=model.history_facts(history)
    assert facts['changes'][0]['same_result']
    assert facts['changes'][0]['changed_clauses']==['s0.where']
    assert model.route(history)=='reground'
    history[-1]['observation']=observed([[2]])
    assert model.route(history)=='edit'


def test_interventions_preserve_other_clauses_and_scope():
    sql='SELECT DISTINCT x FROM t WHERE x IN (SELECT x FROM u WHERE x>1) LIMIT 3'
    original=model.query(sql)
    variants=model.interventions(sql)
    assert len(variants)<=3
    assert {x['removed'] for x in variants}=={'distinct','where','limit'}
    for item in variants:
        actual=model.query(item['sql'])
        assert actual.args['from_'].sql()==original.args['from_'].sql()
        assert actual.args.get(item['removed']) is None
        assert [x.sql() for x in actual.expressions]==[x.sql() for x in original.expressions]
    assert not model.interventions('DELETE FROM t')


def test_literal_binding_and_real_database_checks(tmp_path):
    database=tmp_path/'test.sqlite'
    with sqlite3.connect(database) as c:
        c.executescript('CREATE TABLE t(x INTEGER, label TEXT); CREATE TABLE u(x INTEGER); INSERT INTO t VALUES (1,"a"),(1,"b"),(2,"c");')
    task=dict(db_path=str(database))
    assert model.predicate_field(task,'SELECT a.x FROM t a WHERE a.x=1')==('t','x')
    assert model.predicate_field(task,'SELECT x FROM t JOIN u ON t.x=u.x WHERE x=1') is None
    assert model.predicate_field(task,'SELECT t.x FROM t WHERE x IN (SELECT u.x FROM u WHERE u.x=1)')==('u','x')
    out=model.count_check(str(database),'SELECT DISTINCT x FROM t')
    assert out['observation']['rows']==[(2,)]
    after=model.count_check(str(database),'SELECT x FROM t')
    assert after['observation']['rows']==[(3,)]
