"""Import the pinned engine without modifying it.

The only relaxation: the code archive was packed without data/ folders, so the
upstream manifest check tolerates exactly one missing smoke-test data file that
no API experiment reads. Every other upstream file is still hash-checked.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / 'code' / 'ReLink-SQL_CUDA_20261019'
TOLERATED_MISSING = {'experiments/iteration13/data/confirmation400/public.jsonl'}

sys.path.insert(0, str(CODE))
from cuda_repro import common  # noqa: E402

_original = common.verify_upstream


def verify_upstream():
    marker = common.read(common.ROOT / 'upstream_manifest.json')
    actual = {str(p.relative_to(common.ROOT / 'upstream')): common.sha(p)
              for p in sorted((common.ROOT / 'upstream').rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    expected = {k: v for k, v in marker['files'].items() if k not in TOLERATED_MISSING}
    if actual != expected:
        raise ValueError('Upstream source differs from its manifest')
    return len(actual)


common.verify_upstream = verify_upstream
from cuda_repro import runner  # noqa: E402

engine = runner.engine
r = runner.r
