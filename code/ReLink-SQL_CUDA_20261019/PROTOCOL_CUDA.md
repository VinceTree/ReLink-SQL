# Isolated CUDA Spider subset protocol

This is a new inference configuration derived from `upstream/experiments/iteration17/PROTOCOL.md`.
The public upstream snapshot is verified against `upstream_manifest.json`.
The original `iteration12/PROTOCOL.md` and `iteration13/PROTOCOL.md` are missing from the supplied archive.
They are explicitly recorded as missing historical provenance, never fabricated or silently skipped by the old runner.
This standalone runner freezes all existing upstream files, its own source, this protocol, requirements, dependency
versions, input JSON, selected database files, model identities and inventory in a new manifest instead of invoking
the historical full-dev preparation/protection functions.

## Scope and selection

Default: Spider 1.0 official development JSON (1,034 rows), 10 tasks, CURRENT/ReLink-SQL only.
Before reference execution or inference, sort original row indices by SHA-256 of
`['relink-cuda-spider-subset-v1', row_index, db_id, question]` using common.digest and take the first N.
Save original indices and IDs. No outcome filtering, replacement or difficulty-based tuning.
The source JSON must include `query`, `question`, `db_id`; gold SQL stays in private references.
Only selected database files are opened. Gold execution uses the original read-only SQLite executor,
10 second timeout, 50,000 row limit. Unscorable selected references are recorded without replacement.
Generate initial answers for all selected requests; run repair and paired quality/cost on scorable requests only.
If none is scorable, fail before any inference. A subset result is an interface/systems check, not a paper accuracy claim.

## CUDA differences

Python 3.11, accepted torch 2.7.1 CUDA 12.8, Transformers 4.51.3, Accelerate 1.6.0, bitsandbytes 0.46.1.
Preserve upstream SQLGlot 30.18.0 and NumPy 2.4.6. Model repositories/revisions are fixed in cuda_repro/assets.py.
Load official non-MLX Qwen2.5 Instruct checkpoints with bitsandbytes NF4, no double quantization,
BF16 compute, one CUDA device, batch size 1. This differs from original MLX 4-bit affine/batch4.
No bitwise or historical SQL/accuracy equivalence is claimed. No training, SHARE assistants or extra baselines.
New runs never load original MLX response caches. Models must be fully present before offline inference.
Use the original chat-template encoding policy, greedy decoding, actual terminal EOS counting,
SQL cap 1,024 and requirement extraction cap 384. Refuse context overflow instead of truncating.

## Method and feedback

Reuse the original iteration17.run.run_round and imported iteration15 implementation:
7B shared initial generation; 14B structure-guided re-answer; up to two direct repairs;
independent requirements plus bounded program modification; conditional final full generation.
No more than five repair stages. Accept/reject uses the pinned official single-database Spider result matcher,
not multi-database test-suite accuracy. Accepted requests stop immediately, including initial successes.
Commit initial SQL observations and each stage's deliveries before obtaining acceptance.
Generation only receives validated public task/state fields; gold SQL/results never enter prompt construction.
Keep original r4 proposal and r5 historical-context rules and duplicate-request memo behavior.

## Accounting, resume and checks

Save prompt/generated token IDs, include terminal EOS in usage but exclude it from text.
Path token totals include the initial generation and every attached actual/auxiliary call.
An exact same-run cached result remains fully charged to each logical path attachment.
Keep interrupted attempts and observable partial usage in append-only events; partial usage is a lower bound,
never zero/free work or an exact end-to-end service cost. Report completed calls separately from logical path cost.
Write immutable per-call checkpoints atomically. Lock each run directory against concurrent writers.
Freeze source, dependencies, inputs and model inventory; refuse resume after changes.
Changing sample size, model, quantization, dependencies or code requires a different run directory.
Audit verifies saved artifacts/counts/transitions/stopping only; it does not rescore SQL or rerun a model.
CUDA kernels, actual generation, memory peaks and throughput must be verified on the target server.
