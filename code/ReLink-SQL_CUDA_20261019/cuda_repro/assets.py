"""Explicit model download/configuration, separate from offline inference."""
import argparse
import json
from pathlib import Path

from .common import output_path, save

MODELS = {
    'strong': {'repo': 'Qwen/Qwen2.5-7B-Instruct',
               'revision': 'a09a35458c702b33eeacc393d103063234e8bc28'},
    'strong14': {'repo': 'Qwen/Qwen2.5-14B-Instruct',
                 'revision': 'cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'},
}


def configure(cache_dir, config_path, download=False, device=0):
    from huggingface_hub import snapshot_download
    from .backend import validate_config
    cache_dir, config_path = output_path(cache_dir), output_path(config_path)
    models = {}
    for key, identity in MODELS.items():
        path = snapshot_download(**dict(repo_id=identity['repo'], revision=identity['revision']),
                                 cache_dir=str(cache_dir), local_files_only=not download,
                                 allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja'])
        models[key] = dict(identity, path=str(Path(path).absolute()))
    config = dict(models=models, device=device, batch_size=1, max_context_tokens=24000,
                  quantization=dict(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                    bnb_4bit_compute_dtype='bfloat16', bnb_4bit_use_double_quant=False))
    validate_config(config)
    save(config_path, config)
    return dict(config=str(config_path), models=models, download=download)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--download', action='store_true', help='Explicitly permit network/model download')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(configure(args.cache_dir, args.config, args.download, args.device), indent=2))


if __name__ == '__main__':
    main()
