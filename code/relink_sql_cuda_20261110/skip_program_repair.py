"""Run the first CUDA ablation: keep r4 planning, skip program repair, run r5.

This script clones a completed Spider CURRENT run into a new directory, reuses
the immutable r0-r3 artifacts and CUDA call ledger, and writes a new r4/r5
trajectory. It never changes the source run.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def read(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def skip_program_deliveries(previous, plans, account):
    deliveries = []
    for row in previous:
        current = {key: value for key, value in row.items()
                   if key not in ("accepted", "feedback")}
        current.update(round=4, carried=True)
        if not row["accepted"]:
            current["call_ids"] = row["call_ids"] + plans[row["task_id"]]["request_ids"]
            current = account(current)
        deliveries.append(current)
    return deliveries


def run(base, output):
    base = Path(base).resolve()
    output = Path(output).resolve()
    root = base.parents[1]
    expected_base = root / "runs/spider_dev_CURRENT_20261020"
    expected_parent = root / "runs/ablations"
    if base != expected_base or output.parent != expected_parent:
        raise SystemExit("STOP: use the confirmed CURRENT source and a new runs/ablations child")
    if not (base / "complete.json").is_file():
        raise SystemExit("STOP: source CURRENT run is incomplete")
    if output.exists():
        raise SystemExit("STOP: output already exists; choose a new directory")
    if any(path.is_symlink() for path in base.rglob("*")):
        raise SystemExit("STOP: source run contains a symlink; refusing to clone it")
    import sys
    code = root / "ReLink-SQL_CUDA_20261019"
    if not (code / "cuda_repro/runner.py").is_file():
        raise SystemExit("STOP: confirmed CUDA runner is missing")
    sys.path.insert(0, str(code))
    from cuda_repro import runner
    from cuda_repro.backend import CudaCallBank

    source_marker = runner.verify(base)
    source_audit = runner.audit(base)
    if not source_audit.get("passed") or source_marker.get("sample_size") != 1034:
        raise SystemExit("STOP: source run failed audit or is not the 1034-row Spider run")
    if source_marker["config"].get("device") != 0 or source_marker["config"].get("batch_size") != 1:
        raise SystemExit("STOP: source config is not the approved GPU 0 / batch 1 setup")
    source_complete = read(base / "complete.json")
    if source_complete.get("final_round") != 5:
        raise SystemExit("STOP: source run is not the complete five-stage baseline")
    required_free = int(sum(path.stat().st_size for path in base.rglob("*") if path.is_file()) * 1.25)
    required_free += 1024**3
    if shutil.disk_usage(root).free < required_free:
        raise SystemExit("STOP: insufficient free space to clone the source run safely")
    if not output.parent.is_dir():
        output.parent.mkdir(parents=True, mode=0o700)
    lock_path = output.parent / "gpu0_ablation.lock"
    if lock_path.is_symlink():
        raise SystemExit("STOP: GPU lock path is a symlink")
    lock_handle = lock_path.open("a")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_handle.close()
        raise SystemExit("STOP: another managed GPU 0 ablation is running")
    try:
        processes = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        lock_handle.close()
        raise SystemExit("STOP: cannot verify GPU 0 availability: " + str(exc))
    if processes:
        lock_handle.close()
        raise SystemExit("STOP: GPU 0 has an existing compute process: " + processes)
    shutil.copytree(base, output)
    for path in (output / "CURRENT/r4", output / "CURRENT/r5"):
        if path.exists():
            shutil.rmtree(path)
    for path in (output / "complete.json", output / "final_paths.jsonl", output / "summary.json"):
        if path.exists():
            path.unlink()
    variant_path = output / "variant.json"
    variant_path.write_text(json.dumps({
        "variant": "without_programmatic_repair",
        "source_run": str(base),
        "source_run_fingerprint": source_marker["run_fingerprint"],
        "source_run_audit": source_audit,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "kept": ["initial", "r1", "r2", "r3", "r4_requirement_extraction", "r5"],
        "removed": ["r4_programmatic_candidates", "r4_candidate_submissions"],
        "new_model_calls": "only requests absent from the copied immutable ledger",
    }, ensure_ascii=False, indent=2) + "\n")
    os.chmod(variant_path, 0o600)

    marker = runner.verify(output)
    config = dict(marker["config"], run_fingerprint=marker["run_fingerprint"])
    runner.configure_models(config)
    bank = CudaCallBank(config, output)
    tasks = read(output / "public.jsonl")
    tasks_by_id = {task["id"]: task for task in tasks}
    full = read(output / "full_contexts.json")
    evaluator = runner.r.Evaluator(output / "private/references.json")
    dest = output / "CURRENT"
    previous = read(dest / "r3/paths.jsonl")
    folder = dest / "r4"
    folder.mkdir(parents=True)

    # Reuse the original r4 requirement-extraction requests, but do not create,
    # execute, filter, or submit any program candidate.
    from experiments.iteration15 import contract
    specs = []
    for row in previous:
        if row["accepted"]:
            continue
        state = runner.engine.r.public_state(row, 4, method="EXEC14")
        messages, metadata = contract.plan_messages(tasks_by_id[row["task_id"]], state)
        spec = runner.engine.r.spec(dict(state, method="CURRENT"), "contract",
                                    messages, metadata, tag="full_dev_17")
        spec["request"]["max_tokens"] = contract.PLAN_CAP
        specs.append(spec)

    plans = {row["task_id"]: row for row in bank.invoke(folder / "planning", specs)}
    deliveries = skip_program_deliveries(
        previous, plans, lambda row: runner.engine.r.account(row, bank))

    runner.engine.r.save(folder / "proposals.jsonl", [
        {"task_id": row["task_id"], "deliver": False,
         "reason": "programmatic_repair_disabled"}
        for row in previous if not row["accepted"]
    ])
    previous = runner.engine.commit(folder, previous, deliveries, tasks_by_id,
                                    evaluator, bank)
    runner.seal(folder)

    previous = runner.engine.current_tail(dest, 5, previous, tasks_by_id,
                                          bank, evaluator)
    runner.seal(dest / "r5")
    runner.save(output / "final_paths.jsonl", previous)
    summary = dict(
        selected_n=len(tasks),
        scorable_n=len(previous),
        excluded_n=len(marker["unscorable"]),
        initial_correct=sum(row["initial_accepted"] for row in previous),
        final_correct=sum(row["accepted"] for row in previous),
        final_round=5,
        path_tokens=sum(row["total_tokens"] for row in previous),
        all_initial_tokens=bank.tokens([row["request_id"] for row in read(output / "initial/observations.jsonl")]),
        feedback="pinned official single-database execution matching",
        independent_test=False,
        ablation_variant="without_programmatic_repair",
    )
    runner.save(output / "summary.json", summary)
    runner.save(output / "complete.json", dict(
        run_fingerprint=marker["run_fingerprint"],
        variant="without_programmatic_repair",
        variant_sha256=hashlib.sha256(variant_path.read_bytes()).hexdigest(),
        final_paths_sha256=runner.sha(output / "final_paths.jsonl"),
        summary_sha256=runner.sha(output / "summary.json"),
        initial_paths_sha256=runner.sha(dest / "initial_paths.jsonl"),
        ledger_events_sha256=runner.sha(output / "ledger/events.jsonl"),
        final_round=5,
        stop_reason="fixed_five_stage_budget",
    ))
    audit = runner.audit(output)
    runner.save(output / "audit.json", audit)
    print(json.dumps({"summary": summary, "audit": audit, "output": str(output)},
                     ensure_ascii=False, indent=2))


def check_source(base):
    base = Path(base).resolve()
    root = base.parents[1]
    code = root / "ReLink-SQL_CUDA_20261019"
    if base != root / "runs/spider_dev_CURRENT_20261020":
        raise SystemExit("STOP: source path is not the confirmed CURRENT run")
    import sys
    sys.path.insert(0, str(code))
    from cuda_repro import runner
    marker = runner.verify(base)
    audit = runner.audit(base)
    complete = read(base / "complete.json")
    if not audit.get("passed") or complete.get("final_round") != 5:
        raise SystemExit("STOP: source run is not a verified five-stage run")
    print(json.dumps({"source": str(base),
                      "run_fingerprint": marker["run_fingerprint"],
                      "sample_size": marker["sample_size"],
                      "audit": audit,
                      "final_round": complete.get("final_round"),
                      "stop_reason": complete.get("stop_reason")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        check_source(args.base)
    elif args.output is None:
        parser.error("--output is required unless --check-only is used")
    else:
        run(args.base, args.output)
