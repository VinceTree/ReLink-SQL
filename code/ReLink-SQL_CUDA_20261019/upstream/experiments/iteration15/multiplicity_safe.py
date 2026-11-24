"""V4.1: malformed SQL skips COUNT rewriting as the protocol intended.

V4 completed development but stopped on an unparsable SQL in the 400 cohort,
before any program candidate was delivered. Keep both original runs unchanged.
Only the parser exception boundary changes; replay both cohorts in new folders.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
from sqlglot.errors import SqlglotError
from . import run,compile_run,compile_repair,multiplicity

VERSION='observable_relational_repair_v4_1_parser_guard'


def count_alternative(*args):
    try:return multiplicity.count_alternative(*args)
    except SqlglotError:return None


def propose(task,history,plan_text,policy='multiplicity_safe'):
    assert policy=='multiplicity_safe'
    function=run.r.methods.selectors.clone(multiplicity.propose,
        count_alternative=count_alternative,VERSION=VERSION)
    sql,metadata=function(task,history,plan_text,'multiplicity')
    metadata['policy']=policy
    return sql,metadata


def main(population):
    def protect(dest,config,inputs):
        return run.r.protect(dest,config,inputs+[Path(multiplicity.__file__),Path(compile_repair.__file__),
                                               run.HERE/'MULTIPLICITY_PROTOCOL.md'])
    runtime=SimpleNamespace(**vars(run.r));runtime.protect=protect
    compiler=SimpleNamespace(__file__=__file__,VERSION=VERSION,propose=propose)
    execute=run.r.methods.selectors.clone(compile_run.main,r=runtime,compiler=compiler)
    execute(population,'multiplicity_safe')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('population',choices=['development','exploration400'])
    main(parser.parse_args().population)
