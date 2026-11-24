"""Expose two CPU definitions without importing a historical MLX collector.

collect_paper_baselines imports only MODELS and seed from iteration2.collect.
The collector imports MLX at module scope, although neither definition uses it.
Extract exactly these existing definitions from the verified archived source.
No MLX module or generation API is stubbed; old collector entry points are not exposed.
"""
import ast
import hashlib
import importlib
import sys
from types import ModuleType

from .common import ROOT, verify_upstream


def install():
    name = 'iteration2.collect'
    if name in sys.modules:
        if not getattr(sys.modules[name], '__cuda_repro_bridge__', False):
            raise ValueError('Refusing to mix a pre-imported historical collector with the CUDA entry point')
        return
    verify_upstream()
    sys.path.insert(0, str(ROOT / 'upstream/experiments'))
    parent = importlib.import_module('iteration2')
    source = ROOT / 'upstream/experiments/iteration2/collect.py'
    tree = ast.parse(source.read_text())
    models = [node for node in tree.body if isinstance(node, ast.Assign) and
              len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'MODELS']
    seeds = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'seed']
    if len(models) != 1 or len(seeds) != 1:
        raise ValueError('Unexpected archived collector definitions')
    module = ModuleType(name, __doc__)
    module.__file__ = str(source)
    module.__package__ = 'iteration2'
    module.__all__ = ['MODELS', 'seed']
    module.__cuda_repro_bridge__ = True
    module.MODELS = ast.literal_eval(models[0].value)
    module.hashlib = hashlib
    exec(compile(ast.Module(body=seeds, type_ignores=[]), str(source), 'exec'), module.__dict__)
    sys.modules[name] = module
    parent.collect = module
