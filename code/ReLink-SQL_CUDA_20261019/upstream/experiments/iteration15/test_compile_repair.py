import json
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from iteration15 import compile_repair as c


def setup(tmp_path):
    path=tmp_path/'compile.sqlite'
    with sqlite3.connect(path) as db:
        db.executescript('CREATE TABLE t(id INTEGER, name TEXT, amount INTEGER, other_amount INTEGER); INSERT INTO t VALUES (1,"Silverstone Circuit",8,9);')
    return dict(db_path=str(path),question='Return names for Silverstone with amount less than 9',evidence='')


def history(sql):
    return [dict(event='earlier',sql=sql,observation=dict(status='ok'))]


def test_projection_and_type_compatible_binding(tmp_path):
    task=setup(tmp_path)
    plan=json.dumps(dict(meaning='names',outputs=[['t','name']],filters=[['t','amount']],aggregates=[],grain='rows'))
    sql,meta=c.propose(task,history('SELECT * FROM t WHERE other_amount<9'),plan)
    result=c.r.execute(task['db_path'],sql,timeout=3,max_rows=10)
    assert result['columns']==['name'] and result['rows']==[('Silverstone Circuit',)]
    assert 'project_requested_fields' in meta['operations']
    plan=json.dumps(dict(meaning='names',outputs=[['t','name']],filters=[['t','name']],aggregates=[],grain='rows'))
    sql,meta=c.propose(task,history('SELECT name FROM t WHERE other_amount<9'),plan)
    assert not any(isinstance(x,dict) and 'rebind_filter' in x for x in meta['operations'])


def test_unique_value_and_ambiguous_aliases(tmp_path):
    task=setup(tmp_path)
    plan=json.dumps(dict(meaning='names',outputs=[['t','name']],filters=[['t','name']],aggregates=[],grain='rows',inspect=[['t','name']]))
    sql,meta=c.propose(task,history("SELECT name FROM t WHERE name='Silverstone'"),plan)
    assert c.r.execute(task['db_path'],sql,timeout=3,max_rows=10)['rows']==[('Silverstone Circuit',)]
    sql,meta=c.propose(task,history('SELECT a.id,b.name FROM t a JOIN t b ON a.id=b.id'),plan)
    assert 'project_requested_fields' not in meta['operations']


def test_no_rewrite_of_derived_output_and_bounded_toggle(tmp_path):
    task=setup(tmp_path)
    plan=json.dumps(dict(meaning='mean',outputs=[['t','amount']],aggregates=['avg'],grain='single'))
    sql,meta=c.propose(task,history('SELECT SUM(amount) / COUNT(*) AS average FROM t'),plan)
    assert 'project_requested_fields' not in meta['operations']
    sql,meta=c.propose(task,history('SELECT id FROM t'),plan,'toggle')
    assert meta['operations']==['toggle_distinct'] and 'DISTINCT' in sql
