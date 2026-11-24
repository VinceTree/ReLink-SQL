"""Run the Spider CUDA ablation without r4 rejected-result filtering.

The source CURRENT run is cloned. r0-r3 are reused exactly; r4 requirement
extraction remains enabled. Candidate construction and completeness checks are
retained, but the first complete candidate is submitted even when its strict
result key matches a previously rejected result. The original run is untouched.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def read(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def deliver_without_result_filter(result_key):
    """Submit any candidate with a complete result, including known rejections."""
    return result_key is not None


def propose_without_filter(task, history, plan_text, contract, multiplicity,
                           multiplicity_safe, execute):
    plan = contract.parse(plan_text, task)
    anchor, _ = contract.align(task, history, plan)
    anchor_sql = history[anchor["index"]]["sql"]
    rejected = {contract.repair.result_key(row["observation"]) for row in history} - {None}
    seen = set()
    metadata = dict(version="no_rejected_result_filter_v1", plan=plan, anchor=anchor,
                    operations=[], probes=[], search=[])

    for operation in ("result_multiplicity", "count_multiplicity", "bound_requirements"):
        try:
            if operation == "result_multiplicity":
                candidate, detail = multiplicity.local.propose(task, history, plan_text, "toggle")
            elif operation == "count_multiplicity":
                candidate = multiplicity_safe.count_alternative(task, anchor_sql, plan)
                detail = {}
            else:
                candidate, detail = multiplicity.local.propose(task, history, plan_text, "requirements")
        except (ValueError, TypeError, IndexError, KeyError):
            candidate, detail = None, {}

        metadata["probes"].extend(detail.get("probes", []))
        if candidate is None or candidate in seen:
            metadata["search"].append(dict(operation=operation, status="unavailable_or_identical_sql"))
            continue
        seen.add(candidate)
        observation = execute(task["db_path"], candidate)
        key = contract.repair.result_key(observation)
        status = "incomplete" if key is None else (
            "known_rejected_result" if key in rejected else "new_result")
        metadata["probes"].append(dict(sql=candidate, observation=contract.r.stable(observation)))
        metadata["search"].append(dict(operation=operation, status=status))

        # This is the ablated decision: completeness is still required, but a
        # previously rejected complete result is no longer filtered out.
        if deliver_without_result_filter(key):
            metadata["operations"] = [operation]
            metadata["local_operations"] = detail.get("operations", [])
            metadata["selected_status"] = status
            return candidate, metadata

    return None, metadata


def source_check(base):
    base = Path(base).resolve()
    root = base.parents[1]
    if base != root / "runs/spider_dev_CURRENT_20261020":
        raise SystemExit("STOP: source path is not the confirmed CURRENT run")
    sys.path.insert(0, str(root / "ReLink-SQL_CUDA_20261019"))
    from cuda_repro import runner
    marker = runner.verify(base)
    audit = runner.audit(base)
    complete = read(base / "complete.json")
    if (not audit.get("passed") or marker.get("sample_size") != 1034
            or complete.get("final_round") != 5):
        raise SystemExit("STOP: source run is not the audited five-stage Spider run")
    return marker, audit, complete


def run(base, output):
    base = Path(base).resolve()
    output = Path(output).resolve()
    root = base.parents[1]
    expected_parent = root / "runs/ablations"
    if base != root / "runs/spider_dev_CURRENT_20261020" or output.parent != expected_parent:
        raise SystemExit("STOP: use the confirmed CURRENT source and a new runs/ablations child")
    if output.exists():
        raise SystemExit("STOP: output already exists; choose a new directory")
    if any(path.is_symlink() for path in base.rglob("*")):
        raise SystemExit("STOP: source run contains a symlink; refusing to clone it")

    import sys
    code = root / "ReLink-SQL_CUDA_20261019"
    sys.path.insert(0, str(code))
    from cuda_repro import runner
    from cuda_repro.backend import CudaCallBank

    source_marker, source_audit, _ = source_check(base)
    required_free = int(sum(p.stat().st_size for p in base.rglob("*") if p.is_file()) * 1.25) + 1024**3
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
    current = output / "CURRENT"
    for path in (current / "r4", current / "r5"):
        if path.exists():
            shutil.rmtree(path)
    for path in (output / "complete.json", output / "final_paths.jsonl", output / "summary.json"):
        if path.exists():
            path.unlink()

    variant_path = output / "variant.json"
    variant_path.write_text(json.dumps({
        "variant": "without_rejected_result_filter",
        "source_run": str(base),
        "source_run_fingerprint": source_marker["run_fingerprint"],
        "source_run_audit": source_audit,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "kept": ["initial", "r1", "r2", "r3", "r4_requirement_extraction", "r4_local_candidates", "r5"],
        "removed": ["r4_known_rejected_result_filter"],
        "candidate_rule": "first complete candidate is submitted even if its strict result was previously rejected",
    }, ensure_ascii=False, indent=2) + "\n")
    os.chmod(variant_path, 0o600)

    marker = runner.verify(output)
    config = dict(marker["config"], run_fingerprint=marker["run_fingerprint"])
    runner.configure_models(config)
    bank = CudaCallBank(config, output)
    tasks = read(output / "public.jsonl")
    tasks_by_id = {task["id"]: task for task in tasks}
    evaluator = runner.r.Evaluator(output / "private/references.json")
    dest = output / "CURRENT"
    previous = read(dest / "r3/paths.jsonl")
    histories = runner.engine.history(dest)
    folder = dest / "r4"
    folder.mkdir(parents=True)

    from experiments.iteration15 import contract, multiplicity, multiplicity_safe
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

    proposals = []
    deliveries = []
    for row in previous:
        tid = row["task_id"]
        delivery = {key: value for key, value in row.items()
                    if key not in ("accepted", "feedback")}
        delivery.update(round=4, carried=True)
        if not row["accepted"]:
            plan = plans[tid]
            delivery["call_ids"] = row["call_ids"] + plan["request_ids"]
            sql, metadata = propose_without_filter(
                tasks_by_id[tid], histories[tid], plan["text"], contract,
                multiplicity, multiplicity_safe,
                lambda db, query: runner.r.execute(db, query, timeout=10, max_rows=50000),
            )
            observation = runner.r.execute(tasks_by_id[tid]["db_path"], sql,
                                           timeout=10, max_rows=50000) if sql else None
            result = contract.repair.result_key(observation) if observation else None
            delivered = result is not None
            proposals.append(dict(
                task_id=tid, sql=sql,
                observation=runner.r.stable(observation) if observation else None,
                metadata=metadata, deliver=delivered,
                delivered_rejected_result=bool(
                    delivered and metadata.get("selected_status") == "known_rejected_result"),
            ))
            if delivered:
                delivery.update(sql=sql, observation=observation, carried=False,
                                rejected_sqls=row["rejected_sqls"] + [row["sql"]])
            delivery = runner.engine.r.account(delivery, bank)
        deliveries.append(delivery)

    runner.engine.r.save(folder / "proposals.jsonl", proposals)
    previous = runner.engine.commit(folder, previous, deliveries, tasks_by_id,
                                    evaluator, bank)
    runner.seal(folder)
    previous = runner.engine.current_tail(dest, 5, previous, tasks_by_id, bank, evaluator)
    runner.seal(dest / "r5")

    proposal_rows = read(folder / "proposals.jsonl")
    submitted = [row for row in proposal_rows if row["deliver"]]
    summary = dict(
        selected_n=len(tasks), scorable_n=len(previous), excluded_n=len(marker["unscorable"]),
        initial_correct=sum(row["initial_accepted"] for row in previous),
        final_correct=sum(row["accepted"] for row in previous), final_round=5,
        path_tokens=sum(row["total_tokens"] for row in previous),
        all_initial_tokens=bank.tokens([row["request_id"] for row in read(output / "initial/observations.jsonl")]),
        r4_program_candidates_tested=sum(len(row["metadata"].get("search", [])) for row in proposal_rows),
        r4_program_candidates_submitted=len(submitted),
        r4_known_rejected_result_submissions=sum(row["delivered_rejected_result"] for row in proposal_rows),
        feedback="pinned official single-database execution matching",
        independent_test=False, ablation_variant="without_rejected_result_filter",
    )
    runner.save(output / "final_paths.jsonl", previous)
    runner.save(output / "summary.json", summary)
    runner.save(output / "complete.json", dict(
        run_fingerprint=marker["run_fingerprint"], variant="without_rejected_result_filter",
        variant_sha256=hashlib.sha256(variant_path.read_bytes()).hexdigest(),
        final_paths_sha256=runner.sha(output / "final_paths.jsonl"),
        summary_sha256=runner.sha(output / "summary.json"),
        initial_paths_sha256=runner.sha(dest / "initial_paths.jsonl"),
        ledger_events_sha256=runner.sha(output / "ledger/events.jsonl"),
        final_round=5, stop_reason="fixed_five_stage_budget"))
    audit = runner.audit(output)
    runner.save(output / "audit.json", audit)
    print(json.dumps({"summary": summary, "audit": audit, "output": str(output)},
                     ensure_ascii=False, indent=2))


def check_source(base):
    marker, audit, complete = source_check(base)
    print(json.dumps({"source": str(Path(base).resolve()),
                      "run_fingerprint": marker["run_fingerprint"],
                      "sample_size": marker["sample_size"], "audit": audit,
                      "final_round": complete.get("final_round"),
                      "stop_reason": complete.get("stop_reason")}, ensure_ascii=False, indent=2))


def source_check(base):
    base = Path(base).resolve()
    root = base.parents[1]
    code = root / "ReLink-SQL_CUDA_20261019"
    if base != root / "runs/spider_dev_CURRENT_20261020":
        raise SystemExit("STOP: source path is not the confirmed CURRENT run")
    sys.path.insert(0, str(code))
    from cuda_repro import runner
    marker = runner.verify(base)
    audit = runner.audit(base)
    complete = read(base / "complete.json")
    if not audit.get("passed") or complete.get("final_round") != 5:
        raise SystemExit("STOP: source run is not the audited five-stage baseline")
    return marker, audit, complete


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
