"""Link existing data/SHARE weights into this new checkout, or check prerequisites.

The current experiment's outputs and model code are never modified.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-project', type=Path, help='Existing pipeline project containing experiments/data and experiments/models')
    args = parser.parse_args()
    if args.from_project:
        source = args.from_project.expanduser().resolve()
        for name in ('data', 'models'):
            target = source / 'experiments' / name
            dest = ROOT / 'experiments' / name
            if not target.is_dir():
                raise SystemExit(f'Missing source directory: {target}')
            if dest.exists() or dest.is_symlink():
                if dest.resolve() != target:
                    raise SystemExit(f'Existing resource path differs; not replacing: {dest}')
            else:
                dest.symlink_to(target, target_is_directory=True)
    resources = json.loads((ROOT / 'resources.json').read_text())
    missing = []
    for name in resources['external_data']:
        if not (ROOT / name).exists():
            missing.append(name)
    for role in ('bam', 'sam', 'lom'):
        folder = ROOT / 'experiments/models' / f'share-{role}-4bit'
        expected = resources['share_' + role]
        for name in ('config.json', 'share_provenance.json'):
            if not (folder / name).is_file():
                missing.append(str(folder.relative_to(ROOT) / name))
        if (folder / 'config.json').exists():
            actual = hashlib.sha256((folder / 'config.json').read_bytes()).hexdigest()
            if actual != expected['converted_config_sha256']:
                missing.append(f'{role}: converted config does not match the recorded configuration')
        if not list(folder.glob('*.safetensors')):
            missing.append(f'{role}: no converted model weights')
    # Use exactly the cache layout required by the preserved inference backend.
    for repo, revision in resources['generator_models'].values():
        folder = Path.home() / '.cache/huggingface/hub' / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
        if not (folder / 'config.json').is_file() or not list(folder.glob('*.safetensors')):
            missing.append(f'Model cache: {repo}@{revision}')
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        missing.append('The supplied inference backend requires macOS on Apple Silicon')
    print(json.dumps({'ready': not missing, 'missing': missing, 'writes_to_existing_project': False}, ensure_ascii=False, indent=2))
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
