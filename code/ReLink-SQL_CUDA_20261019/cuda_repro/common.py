"""Small, durable artifact helpers shared by the standalone entry points."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), default=str).encode()).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def save(path, value):
    """Commit immutable artifacts, allowing only byte-identical resume writes."""
    path = Path(path)
    if path.suffix == '.jsonl':
        text = ''.join(json.dumps(row, ensure_ascii=False, default=str) + '\n' for row in value)
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise ValueError('Refusing changed committed artifact: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def output_path(value):
    path = Path(value).expanduser().resolve()
    if path == ROOT / 'upstream' or (ROOT / 'upstream') in path.parents:
        raise ValueError('Outputs must be outside the preserved upstream copy')
    return path


def verify_upstream():
    marker = read(ROOT / 'upstream_manifest.json')
    actual = {str(p.relative_to(ROOT / 'upstream')): sha(p)
              for p in sorted((ROOT / 'upstream').rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if actual != marker['files']:
        raise ValueError('The preserved upstream source differs from its manifest')
    return len(actual)
