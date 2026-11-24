"""BIRD CUDA ablation: submit the first complete r4 result, even if previously rejected."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


VARIANT = "bird_without_rejected_result_filter_v1"
SUFFIX = "|BIRD_NO_RESULT_FILTER_20261111"


def read(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def deliver_without_result_filter(result_key):
    return result_key is not None


class VariantBank:
    def __init__(self, bank):
        self.bank = bank

    def __getattr__(self, key):
        return getattr(self.bank, key)

    def invoke(self, directory, specs, stopped=False):
        namespaced = copy.deepcopy(specs)
        for spec in namespaced:
            spec["request"]["id"] += SUFFIX
        return self.bank.invoke(directory, namespaced, stopped)


def propose_without_filter(task, history, plan_text, contract, multiplicity,
                           multiplicity_safe, execute):
    plan = contract.parse(plan_text, task)
    anchor, _ = contract.align(task, history, plan)
    anchor_sql = history[anchor["index"]]["sql"]
    rejected = {contract.repair.result_key(row["observation"]) for row in history} - {None}
    seen = set()
    metadata = dict(version=VARIANT, plan=plan, anchor=anchor,
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

        if deliver_without_result_filter(key):
            metadata["operations"] = [operation]
            metadata["local_operations"] = detail.get("operations", [])
            metadata["selected_status"] = status
            return candidate, metadata

    return None, metadata


def source_check(base):
    base = Path(base).resolve()
    root = base.parents[1]
    if base != root / "runs/bird_dev_CURRENT_20261110":
        raise SystemExit("STOP: source path is not the audited BIRD CURRENT run")
    sys.path.insert(0, str(root / "ReLink-SQL_CUDA_20261019"))
    from cuda_repro import runner
    marker = runner.verify(base)
    audit = runner.audit(base)
    complete = read(base / "complete.json")
    if (not audit.get("passed") or marker.get("sample_size") != 1534
            or marker.get("protocol") != "isolated-cuda-bird-dev-v1"
            or complete.get("final_round") != 5):
        raise SystemExit("STOP: BIRD source is not the audited 1534-row five-stage run")
    if marker["config"].get("device") != 0 or marker["config"].get("batch_size") != 1:
        raise SystemExit("STOP: source configuration is not GPU 0 / batch 1")
    selected = read(base / "selection.json")["rows"]
    excluded = {row["source_row"] for row in selected if row["id"] in marker["unscorable"]}
    if excluded != {384, 518, 693, 701, 1131}:
        raise SystemExit("STOP: BIRD exclusion set differs from the reviewed baseline")
    print("SOURCE BASELINE AUDIT PASSED", flush=True)
    return root, runner, marker, audit


def shared_prefix(base):
    names = ["initial/observations.jsonl", "CURRENT/initial_paths.jsonl"] + [
        f"CURRENT/r{round_number}/paths.jsonl" for round_number in range(1, 4)]
    return {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name in names}


def assert_shared_prefix(base, output, variant):
    actual = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
              for name in variant["shared_prefix_sha256"]}
    if actual != variant["shared_prefix_sha256"]:
        raise ValueError("Copied r0-r3 prefix differs from the BIRD baseline")


def variant_manifest(base, marker, audit):
    return dict(
        variant=VARIANT,
        source_run=str(base),
        source_run_fingerprint=marker["run_fingerprint"],
        source_run_audit=audit,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        request_suffix=SUFFIX,
        common_initial_answers=True,
        kept=["r0-r3", "r4 requirement extraction", "r4 candidate generation", "r5"],
        removed=["r4 rejected-result filter"],
        candidate_rule="submit the first candidate with a complete strict result key, including known rejections",
        shared_prefix_sha256=shared_prefix(base),
        max_repair_submissions=5,
    )


def paired_comparison(baseline_paths, variant_paths):
    baseline_by_id = {row["task_id"]: row for row in baseline_paths}
    variant_by_id = {row["task_id"]: row for row in variant_paths}
    if baseline_by_id.keys() != variant_by_id.keys():
        raise ValueError("Baseline and ablation scorable task sets differ")
    ids = sorted(baseline_by_id)
    full_only = sum(baseline_by_id[tid]["accepted"] and not variant_by_id[tid]["accepted"] for tid in ids)
    variant_only = sum(variant_by_id[tid]["accepted"] and not baseline_by_id[tid]["accepted"] for tid in ids)
    baseline_tokens = sum(row["total_tokens"] for row in baseline_paths)
    variant_tokens = sum(row["total_tokens"] for row in variant_paths)
    return dict(
        paired_scorable_n=len(ids),
        baseline_final_correct=sum(row["accepted"] for row in baseline_paths),
        variant_final_correct=sum(row["accepted"] for row in variant_paths),
        full_only_successes=full_only,
        ablation_only_successes=variant_only,
        net_correct_delta_variant_minus_baseline=variant_only - full_only,
        baseline_path_tokens=baseline_tokens,
        variant_path_tokens=variant_tokens,
        path_token_delta_variant_minus_baseline=variant_tokens - baseline_tokens,
    )


def completion_record(output, run_fingerprint, paired, sha):
    return dict(
        run_fingerprint=run_fingerprint,
        variant=VARIANT,
        variant_sha256=sha(output / "variant.json"),
        final_paths_sha256=sha(output / "final_paths.jsonl"),
        summary_sha256=sha(output / "summary.json"),
        initial_paths_sha256=sha(output / "CURRENT/initial_paths.jsonl"),
        ledger_events_sha256=sha(output / "ledger/events.jsonl"),
        final_round=5,
        stop_reason="fixed_five_stage_budget",
        paired_comparison=paired,
    )


def summarize(output, base, marker, paths, proposals, bank, runner):
    tasks = read(output / "public.jsonl")
    baseline_paths = read(base / "final_paths.jsonl")
    paired = paired_comparison(baseline_paths, paths)
    submitted = [row for row in proposals if row["deliver"]]
    summary = dict(
        selected_n=len(tasks),
        scorable_n=len(paths),
        excluded_n=len(marker["unscorable"]),
        initial_correct=sum(row["initial_accepted"] for row in paths),
        final_correct=sum(row["accepted"] for row in paths),
        final_round=5,
        path_tokens=sum(row["total_tokens"] for row in paths),
        all_initial_tokens=bank.tokens([row["request_id"] for row in
                                        read(output / "initial/observations.jsonl")]),
        r4_candidate_search_operations=sum(len(row["metadata"].get("search", []))
                                           for row in proposals),
        r4_candidate_check_executions=sum(
            sum(item["status"] not in ("unavailable_or_identical_sql",)
                for item in row["metadata"].get("search", [])) for row in proposals),
        r4_program_candidates_submitted=len(submitted),
        r4_submitted_candidate_sql_reexecutions=len(submitted),
        r4_known_rejected_result_submissions=sum(
            row["delivered_rejected_result"] for row in submitted),
        paired_comparison=paired,
        feedback="pinned official single-database execution matching",
        independent_test=False,
        ablation_variant=VARIANT,
    )
    return summary, paired


def finalize(base, output):
    base, output = Path(base).resolve(), Path(output).resolve()
    root, runner, source_marker, source_audit = source_check(base)
    expected = root / "runs/ablations/bird_without_rejected_result_filter_20261111"
    if output != expected:
        raise SystemExit("STOP: finalization path is not the designated BIRD ablation")
    variant = read(output / "variant.json")
    if variant != variant_manifest(base, source_marker, source_audit):
        raise SystemExit("STOP: saved variant manifest differs from this protocol")
    assert_shared_prefix(base, output, variant)
    marker = runner.verify(output)
    dest = output / "CURRENT"
    for stage in range(1, 6):
        if not (dest / f"r{stage}/complete.json").is_file():
            raise SystemExit(f"STOP: missing completed r{stage} stage")
    paths = read(output / "final_paths.jsonl")
    if paths != read(dest / "r5/paths.jsonl"):
        raise SystemExit("STOP: final paths differ from committed r5 paths")
    proposals = read(dest / "r4/proposals.jsonl")
    from cuda_repro.backend import CudaCallBank
    runner.configure_models(dict(marker["config"], run_fingerprint=marker["run_fingerprint"]))
    bank = CudaCallBank(dict(marker["config"], run_fingerprint=marker["run_fingerprint"]),
                        output, readonly=True)
    summary, paired = summarize(output, base, marker, paths, proposals, bank, runner)
    if read(output / "summary.json") != summary:
        raise SystemExit("STOP: saved summary differs from recomputed saved-record summary")
    if len(paths) != 1529:
        raise SystemExit("STOP: unexpected BIRD scorable count")
    if not (output / "complete.json").exists():
        runner.save(output / "complete.json", completion_record(
            output, marker["run_fingerprint"], paired, runner.sha))
    audit = runner.audit(output)
    runner.save(output / "audit.json", audit)
    print(json.dumps({"finalized_saved_run": True, "summary": summary,
                      "audit": audit, "output": str(output)}, ensure_ascii=False, indent=2))


def run(base, output):
    base, output = Path(base).resolve(), Path(output).resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        raise SystemExit("STOP: CUDA_VISIBLE_DEVICES must be unset")
    root, runner, source_marker, source_audit = source_check(base)
    expected_parent = root / "runs/ablations"
    if output.parent != expected_parent:
        raise SystemExit("STOP: output must be a child of runs/ablations")
    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        raise SystemExit("STOP: staging exists; preserve it for inspection")
    if any(path.is_symlink() for path in base.rglob("*")):
        raise SystemExit("STOP: source contains a symlink")
    needed = int(sum(path.stat().st_size for path in base.rglob("*") if path.is_file()) * 1.25)
    if shutil.disk_usage(root).free < needed:
        raise SystemExit("STOP: insufficient free space to clone baseline")
    expected_parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    lock_path = expected_parent / "gpu0_ablation.lock"
    if lock_path.is_symlink():
        raise SystemExit("STOP: GPU lock path is a symlink")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("STOP: another managed GPU 0 experiment is running")
        gpu = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=uuid",
                                       "--format=csv,noheader"], text=True).strip()
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                        "--format=csv,noheader"], text=True)
        if any(line.split(",")[0].strip() == gpu for line in apps.splitlines()):
            raise SystemExit("STOP: GPU 0 is occupied")

        variant = variant_manifest(base, source_marker, source_audit)
        if output.exists():
            if read(output / "variant.json") != variant:
                raise SystemExit("STOP: existing output has a different variant manifest")
        else:
            print("CLONE BASELINE TO VARIANT STAGING", flush=True)
            shutil.copytree(base, staging)
            for path in (staging / "CURRENT/r4", staging / "CURRENT/r5"):
                if path.exists():
                    shutil.rmtree(path)
            for name in ("complete.json", "final_paths.jsonl", "summary.json", "audit.json"):
                path = staging / name
                if path.exists():
                    path.unlink()
            runner.save(staging / "variant.json", variant)
            os.rename(staging, output)
            print("VARIANT CLONE COMPLETE", flush=True)

        assert_shared_prefix(base, output, variant)
        marker = runner.verify(output)
        if (output / "complete.json").exists():
            audit = runner.audit(output)
            print(json.dumps({"already_complete": True, "summary": read(output / "summary.json"),
                              "audit": audit, "output": str(output)}, ensure_ascii=False, indent=2))
            return
        config = dict(marker["config"], run_fingerprint=marker["run_fingerprint"])
        runner.configure_models(config)
        from cuda_repro.backend import CudaCallBank
        bank = VariantBank(CudaCallBank(config, output))
        tasks = read(output / "public.jsonl")
        tasks_by_id = {task["id"]: task for task in tasks}
        evaluator = runner.r.Evaluator(output / "private/references.json")
        dest = output / "CURRENT"
        previous = read(dest / "r3/paths.jsonl")
        folder = dest / "r4"
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "complete.json").exists():
            previous = read(folder / "paths.jsonl")
            proposals = read(folder / "proposals.jsonl")
        else:
            print("R4 REQUIREMENT EXTRACTION START", flush=True)
            from experiments.iteration15 import contract, multiplicity, multiplicity_safe
            histories = runner.engine.history(dest)
            specs = []
            for row in previous:
                if row["accepted"]:
                    continue
                state = runner.engine.r.public_state(row, 4, method="EXEC14")
                messages, metadata = contract.plan_messages(tasks_by_id[row["task_id"]], state)
                spec = runner.engine.r.spec(dict(state, method="CURRENT"), "contract",
                                            messages, metadata, tag="bird_dev_17")
                spec["request"]["max_tokens"] = contract.PLAN_CAP
                specs.append(spec)
            plans = {row["task_id"]: row for row in bank.invoke(folder / "planning", specs)}
            item_dir = folder / "proposal_items"
            item_dir.mkdir(parents=True, exist_ok=True)
            proposals_by_id = {}
            for index, row in enumerate(previous, 1):
                if row["accepted"]:
                    continue
                tid = row["task_id"]
                item_path = item_dir / (hashlib.sha256(tid.encode()).hexdigest() + ".json")
                if item_path.exists():
                    proposal = read(item_path)
                else:
                    plan = plans[tid]
                    sql, metadata = propose_without_filter(
                        tasks_by_id[tid], histories[tid], plan["text"], contract,
                        multiplicity, multiplicity_safe,
                        lambda db, query: runner.r.execute(db, query, timeout=10, max_rows=50000))
                    observation = runner.r.execute(tasks_by_id[tid]["db_path"], sql,
                        timeout=10, max_rows=50000) if sql else None
                    proposal = dict(
                        task_id=tid,
                        sql=sql,
                        observation=runner.r.stable(observation) if observation else None,
                        metadata=metadata,
                        deliver=deliver_without_result_filter(
                            contract.repair.result_key(observation) if observation else None),
                        delivered_rejected_result=bool(
                            sql and metadata.get("selected_status") == "known_rejected_result"),
                    )
                    runner.save(item_path, proposal)
                proposals_by_id[tid] = proposal
                if index % 100 == 0:
                    print("R4 CANDIDATE SEARCH", index, "/", len(previous), flush=True)

            proposals = [proposals_by_id[row["task_id"]] for row in previous
                         if not row["accepted"]]
            runner.save(folder / "proposals.jsonl", proposals)
            deliveries = []
            for row in previous:
                current = {key: value for key, value in row.items()
                           if key not in ("accepted", "feedback")}
                current.update(round=4, carried=True)
                if not row["accepted"]:
                    plan = plans[row["task_id"]]
                    proposal = proposals_by_id[row["task_id"]]
                    current["call_ids"] = row["call_ids"] + plan["request_ids"]
                    if proposal["deliver"]:
                        current.update(sql=proposal["sql"], observation=proposal["observation"],
                                       carried=False,
                                       rejected_sqls=row["rejected_sqls"] + [row["sql"]])
                    current = runner.engine.r.account(current, bank)
                deliveries.append(current)
            previous = runner.engine.commit(folder, previous, deliveries, tasks_by_id,
                                            evaluator, bank)
            runner.seal(folder)
            print("R4 COMPLETE", "accepted", sum(row["accepted"] for row in previous),
                  "submitted", sum(row["deliver"] for row in proposals), flush=True)

        print("R5 CONDITIONAL GENERATION START", flush=True)
        previous = runner.engine.current_tail(dest, 5, previous, tasks_by_id, bank, evaluator)
        if not (dest / "r5/cuda_artifacts.json").exists():
            runner.seal(dest / "r5")
        print("R5 COMPLETE", "accepted", sum(row["accepted"] for row in previous), flush=True)
        runner.save(output / "final_paths.jsonl", previous)
        summary, paired = summarize(output, base, marker, previous, proposals, bank, runner)
        runner.save(output / "summary.json", summary)
        runner.save(output / "complete.json", completion_record(
            output, marker["run_fingerprint"], paired, runner.sha))
        audit = runner.audit(output)
        runner.save(output / "audit.json", audit)
        print(json.dumps({"summary": summary, "audit": audit, "output": str(output)},
                         ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.finalize_only:
        finalize(args.base, args.output)
    else:
        run(args.base, args.output)


if __name__ == "__main__":
    main()
