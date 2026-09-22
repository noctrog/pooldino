"""Packaging checks that do not import models or require accelerator hardware."""

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def module_exists(name):
    path = ROOT.joinpath(*name.split('.'))
    return path.with_suffix('.py').is_file() or (path / '__init__.py').is_file()


def test_all_internal_imports_resolve_to_shipped_modules():
    missing = []
    for path in (ROOT / 'pooldino').rglob('*.py'):
        parts = path.relative_to(ROOT).with_suffix('').parts
        package = '.'.join(parts[:-1])
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ''
                if node.level:
                    name = importlib.util.resolve_name('.' * node.level + name, package)
                names = [name]
            for name in names:
                if name.startswith('pooldino') and not module_exists(name):
                    missing.append((str(path.relative_to(ROOT)), name))
    assert not missing, missing


def test_single_project_layout():
    assert not (ROOT / 'pooldino' / 'proj').exists()
    for module in ['train_decoder', 'train_generator', 'train_segmentation', 'train_depth', 'compute_stats']:
        assert module_exists('pooldino.' + module)
