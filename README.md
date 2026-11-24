# ReLink-SQL

Code and experiment results for **ReLink-SQL**, a post-generation correction framework for Text-to-SQL. When an initial SQL query is rejected, ReLink-SQL selects the relevant schema, repairs the query programmatically where possible, and regenerates on the selected schema. Candidates whose results were already rejected are filtered out.

## Repository layout

| Path | Contents |
|---|---|
| `code/ReLink-SQL_CUDA_20261019/` | Core engine. `cuda_repro/` is the CUDA runner (Qwen2.5 Instruct, bitsandbytes NF4). `upstream/` is the pinned original implementation, hash-checked against `upstream_manifest.json`. `PROTOCOL_CUDA.md` describes the run protocol. |
| `code/relink_sql_cuda_20261110/` | BIRD entry point (`bird_cuda.py`) and the ablation scripts: full-schema retry, without schema selection, without programmatic repair, without rejected-result filtering. Each script comes with a test. |
| `relink_api/boot.py` | Engine bootstrap that verifies the upstream hashes. Saved API-model experiment results are in `results/`. |
| `analysis/` | Latency statistics for the CUDA runs. |
| `results/` | Supplementary experiments on the 1,529 scorable BIRD dev questions: backbone models (E1, E2), CRITIC comparisons (E1b, E2b), parameter sensitivity (E3), and stage-level token and outcome analysis (E4, E5). Contains summary tables, bootstrap intervals and per-question results; see `results/README.md`. |
| `data/` | Instructions for obtaining the benchmark datasets. |

## Environment

- Python 3.11
- `pip install -r code/ReLink-SQL_CUDA_20261019/requirements.cuda.txt`
  - torch 2.7.1 (CUDA 12.8), transformers 4.51.3, accelerate 1.6.0, bitsandbytes 0.46.1, sqlglot 30.18.0, numpy 2.4.6
- One CUDA GPU. The model repositories and revisions are pinned in `cuda_repro/assets.py`.

## Running

```bash
# Spider (CUDA runner); see PROTOCOL_CUDA.md for the full protocol
(cd code/ReLink-SQL_CUDA_20261019 && python -m cuda_repro.runner --help)

# BIRD and ablations; RELINK_ROOT is the directory that contains ReLink-SQL_CUDA_20261019/
export RELINK_ROOT=$(pwd)/code
python code/relink_sql_cuda_20261110/bird_cuda.py --help
python code/relink_sql_cuda_20261110/bird_full_schema_retry.py --help
```

`bird_cuda.py prepare` checks every BIRD database file against the SHA-256 values in `DATABASES` before any model call.

Tests:

```bash
cd code/ReLink-SQL_CUDA_20261019 && python -m pytest -q tests
```

## Evaluation

Accuracy is single-database execution matching on the public dev sets (Spider 1.0 dev, BIRD dev). Statistical tests use paired, database-clustered bootstrap with 10,000 resamples.

## Third-party code

The paths below are relative to `code/ReLink-SQL_CUDA_20261019/upstream/`.

- [SHARE](https://github.com/quge2023/SHARE) prompt source and its Apache-2.0 license are included in `experiments/vendor/share/`.
- [Spider test-suite evaluator](https://github.com/taoyds/test-suite-sql-eval) and its Apache-2.0 license are included in `experiments/iteration8/sources/sql_eval/`. The [BIRD evaluator](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird) in the same directory has its own MIT license in `LICENSE-BIRD`.
- The [DIN-SQL](https://github.com/MohammadrezaPourreza/Few-shot-NL2SQL-with-prompting) prompt adaptations and upstream MIT license are included under `experiments/iteration4/`.
- The [CHESS](https://github.com/ShayanTalaei/CHESS) revision prompt in `experiments/iteration4/prompts/chess_revise_one.txt` is accompanied by the upstream Apache-2.0 license in `experiments/iteration4/source_materials/chess/`.

## Reproduction scope

The standalone CUDA entry points prepare inputs from the official benchmark downloads described in `data/README.md`. The archived upstream MLX full-development runner also expects historical pilot inputs. The public package contains the runnable CUDA entry points and saved supplementary BIRD results. The supplementary API experiment runner and its original checkpoints are separate from the engine bootstrap included here.

## License

This code is released under the Apache License 2.0; see `LICENSE`. The third-party components listed above keep their own licenses.
