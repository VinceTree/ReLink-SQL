"""BIRD CUDA ablation: remove r4 programmatic repair, keep r5 conditional generation."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


VARIANT = "bird_without_programmatic_repair_v1"
SUFFIX = "|BIRD_NO_PROGRAM_REPAIR_20261111"
RECORDED_RUN_DRIVER_SHA = "2df58e291d9a758730564bc252611ac6045db31daf946c9308f5de18612eaa18"


def read(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


class VariantBank:
    """Namespace new logical requests while reusing the copied immutable ledger."""
    def __init__(self, bank):
        self.bank = bank

    def __getattr__(self, key):
        return getattr(self.bank, key)

    def invoke(self, directory, specs, stopped=False):
        namespaced = copy.deepcopy(specs)
        for spec in namespaced:
            spec["request"]["id"] += SUFFIX
        return self.bank.invoke(directory, namespaced, stopped)


def source_check(base):
    base = Path(base).resolve()
    root = base.parents[1]
    expected = root / "runs/bird_dev_CURRENT_20261110"
    if base != expected:
        raise SystemExit("STOP: source path is not the verified BIRD CURRENT run")
    code = root / "ReLink-SQL_CUDA_20261019"
    sys.path.insert(0, str(code))
    from cuda_repro import runner

    marker = runner.verify(base)
    audit = runner.audit(base)
    complete = read(base / "complete.json")
    if not audit.get("passed") or marker.get("sample_size") != 1534:
        raise SystemExit("STOP: BIRD source failed audit or has unexpected sample size")
    if complete.get("final_round") != 5:
        raise SystemExit("STOP: BIRD source is not a complete five-stage run")
    if marker.get("protocol") != "isolated-cuda-bird-dev-v1":
        raise SystemExit("STOP: source protocol is not the BIRD CUDA protocol")
    if marker["config"].get("device") != 0 or marker["config"].get("batch_size") != 1:
        raise SystemExit("STOP: source configuration is not GPU 0 / batch 1")
    selected = read(base / "selection.json")["rows"]
    source_rows = {row["source_row"] for row in selected if row["id"] in marker["unscorable"]}
    if source_rows != {384, 518, 693, 701, 1131}:
        raise SystemExit("STOP: source exclusion set differs from reviewed preparation")
    print("SOURCE BASELINE AUDIT PASSED", flush=True)
    return root, code, runner, marker, audit


def copy_without_repair_stages(base, output):
    shutil.copytree(base, output)
    for path in (output / "CURRENT/r4", output / "CURRENT/r5"):
        if path.exists():
            shutil.rmtree(path)
    for path in (output / "complete.json", output / "final_paths.jsonl",
                 output / "summary.json", output / "audit.json"):
        if path.exists():
            path.unlink()


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


def variant_manifest(base, marker, source_audit):
    shared = ["initial/observations.jsonl", "CURRENT/initial_paths.jsonl"] + [
        f"CURRENT/r{round_number}/paths.jsonl" for round_number in range(1, 4)]
    return dict(
        variant=VARIANT,
        source_run=str(base),
        source_run_fingerprint=marker["run_fingerprint"],
        source_run_audit=source_audit,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        request_suffix=SUFFIX,
        kept=["initial", "r1", "r2", "r3", "r4_requirement_extraction", "r5"],
        removed=["r4_programmatic_candidates", "r4_candidate_execution", "r4_candidate_submissions"],
        common_initial_answers=True,
        shared_prefix_sha256={name: hashlib.sha256((base / name).read_bytes()).hexdigest()
                              for name in shared},
        max_repair_submissions=5,
    )


def assert_shared_prefix(base, output, variant):
    actual = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
              for name in variant["shared_prefix_sha256"]}
    if actual != variant["shared_prefix_sha256"]:
        raise ValueError("Copied r0-r3 prefix differs from the BIRD baseline")


def paired_comparison(baseline_paths, variant_paths):
    baseline_by_id = {row["task_id"]: row for row in baseline_paths}
    variant_by_id = {row["task_id"]: row for row in variant_paths}
    if baseline_by_id.keys() != variant_by_id.keys():
        raise ValueError("Baseline and ablation scorable task sets differ")
    task_ids = sorted(baseline_by_id)
    full_only = sum(baseline_by_id[tid]["accepted"] and not variant_by_id[tid]["accepted"]
                    for tid in task_ids)
    variant_only = sum(variant_by_id[tid]["accepted"] and not baseline_by_id[tid]["accepted"]
                       for tid in task_ids)
    baseline_tokens = sum(row["total_tokens"] for row in baseline_paths)
    variant_tokens = sum(row["total_tokens"] for row in variant_paths)
    return dict(
        paired_scorable_n=len(task_ids),
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
    variant_path = output / "variant.json"
    return dict(
        run_fingerprint=run_fingerprint,
        variant=VARIANT,
        variant_sha256=sha(variant_path),
        final_paths_sha256=sha(output / "final_paths.jsonl"),
        summary_sha256=sha(output / "summary.json"),
        initial_paths_sha256=sha(output / "CURRENT/initial_paths.jsonl"),
        ledger_events_sha256=sha(output / "ledger/events.jsonl"),
        final_round=5,
        stop_reason="fixed_five_stage_budget",
        paired_comparison=paired,
    )


def finalize(base, output):
    """Validate completed stage artifacts and write only the missing final receipt."""
    base, output = Path(base).resolve(), Path(output).resolve()
    root, _, runner, source_marker, source_audit = source_check(base)
    if output != root / "runs/ablations/bird_without_programmatic_repair_20261111_retry01":
        raise SystemExit("STOP: finalization path is not the interrupted BIRD ablation")
    variant_path = output / "variant.json"
    if not variant_path.is_file():
        raise SystemExit("STOP: missing variant manifest")
    saved_variant = read(variant_path)
    expected_variant = variant_manifest(base, source_marker, source_audit)
    expected_variant["driver_sha256"] = RECORDED_RUN_DRIVER_SHA
    if saved_variant != expected_variant:
        raise SystemExit("STOP: saved variant manifest does not match the interrupted run")
    assert_shared_prefix(base, output, saved_variant)
    marker = runner.verify(output)
    dest = output / "CURRENT"
    for stage in range(1, 6):
        if not (dest / f"r{stage}/complete.json").is_file():
            raise SystemExit(f"STOP: missing completed r{stage} stage")
    baseline_paths = read(base / "final_paths.jsonl")
    variant_paths = read(output / "final_paths.jsonl")
    if variant_paths != read(dest / "r5/paths.jsonl"):
        raise SystemExit("STOP: final paths differ from the completed r5 paths")
    paired = paired_comparison(baseline_paths, variant_paths)
    summary = read(output / "summary.json")
    expected_summary = dict(
        selected_n=1534,
        scorable_n=len(variant_paths),
        excluded_n=5,
        initial_correct=sum(row["initial_accepted"] for row in variant_paths),
        final_correct=sum(row["accepted"] for row in variant_paths),
        final_round=5,
        path_tokens=sum(row["total_tokens"] for row in variant_paths),
        r4_program_candidates_tested=0,
        r4_program_candidates_submitted=0,
        paired_comparison=paired,
        feedback="pinned official single-database execution matching",
        independent_test=False,
        ablation_variant=VARIANT,
    )
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            raise SystemExit("STOP: summary mismatch at " + key)
    if len(variant_paths) != 1529:
        raise SystemExit("STOP: unexpected BIRD scorable count")
    if not (output / "complete.json").exists():
        runner.save(output / "complete.json", completion_record(
            output, marker["run_fingerprint"], paired, runner.sha))
    audit = runner.audit(output)
    runner.save(output / "audit.json", audit)
    print(json.dumps({"finalized_saved_run": True, "summary": summary,
                      "audit": audit, "output": str(output)}, ensure_ascii=False, indent=2))


def run(base, output):
    base = Path(base).resolve()
    output = Path(output).resolve()
    print("VERIFY BIRD BASELINE", flush=True)
    root, code, runner, marker, source_audit = source_check(base)
    expected_parent = root / "runs/ablations"
    if output.parent != expected_parent:
        raise SystemExit("STOP: output must be a child of runs/ablations")
    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        raise SystemExit("STOP: staging directory exists; preserve it for inspection")
    if any(path.is_symlink() for path in base.rglob("*")):
        raise SystemExit("STOP: source contains a symlink")
    required_free = int(sum(path.stat().st_size for path in base.rglob("*") if path.is_file()) * 1.25)
    if shutil.disk_usage(root).free < required_free:
        raise SystemExit("STOP: insufficient free space for an isolated clone")
    expected_parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    lock_path = expected_parent / "gpu0_ablation.lock"
    if lock_path.is_symlink():
        raise SystemExit("STOP: GPU lock path is a symlink")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("STOP: another managed GPU 0 experiment is running")
        try:
            gpu = subprocess.check_output(
                ["nvidia-smi", "--id=0", "--query-gpu=uuid", "--format=csv,noheader"],
                text=True).strip()
            apps = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SystemExit("STOP: cannot verify GPU 0 availability: " + str(exc))
        if any(line.split(",")[0].strip() == gpu for line in apps.splitlines()):
            raise SystemExit("STOP: GPU 0 is occupied")

        variant = variant_manifest(base, marker, source_audit)
        if output.exists():
            if not (output / "variant.json").is_file() or read(output / "variant.json") != variant:
                raise SystemExit("STOP: existing output has a different or missing variant manifest")
        else:
            print("CLONE BASELINE TO VARIANT STAGING", flush=True)
            copy_without_repair_stages(base, staging)
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
        valid = {task["id"]: task for task in tasks if task["id"] not in marker["unscorable"]}
        evaluator = runner.r.Evaluator(output / "private/references.json")
        dest = output / "CURRENT"
        previous = read(dest / "r3/paths.jsonl")
        folder = dest / "r4"
        folder.mkdir(parents=True, exist_ok=True)

        print("R4 REQUIREMENT EXTRACTION START", flush=True)
        from experiments.iteration15 import contract
        specs = []
        for row in previous:
            if row["accepted"]:
                continue
            state = runner.engine.r.public_state(row, 4, method="EXEC14")
            messages, metadata = contract.plan_messages(valid[row["task_id"]], state)
            spec = runner.engine.r.spec(dict(state, method="CURRENT"), "contract",
                                        messages, metadata, tag="bird_dev_17")
            spec["request"]["max_tokens"] = contract.PLAN_CAP
            specs.append(spec)
        plans = {row["task_id"]: row for row in bank.invoke(folder / "planning", specs)}

        proposals = []
        for row in previous:
            if not row["accepted"]:
                proposals.append(dict(task_id=row["task_id"], deliver=False,
                                      reason="programmatic_repair_disabled"))
        deliveries = skip_program_deliveries(
            previous, plans, lambda row: runner.engine.r.account(row, bank))
        runner.engine.r.save(folder / "proposals.jsonl", proposals)
        previous = runner.engine.commit(folder, previous, deliveries, valid, evaluator, bank)
        runner.seal(folder)
        print("R4 COMPLETE", "accepted", sum(row["accepted"] for row in previous), flush=True)

        print("R5 CONDITIONAL GENERATION START", flush=True)
        previous = runner.engine.current_tail(dest, 5, previous, valid, bank, evaluator)
        runner.seal(dest / "r5")
        print("R5 COMPLETE", "accepted", sum(row["accepted"] for row in previous), flush=True)
        runner.save(output / "final_paths.jsonl", previous)
        baseline_paths = read(base / "final_paths.jsonl")
        paired = paired_comparison(baseline_paths, previous)
        summary = dict(
            selected_n=len(tasks),
            scorable_n=len(previous),
            excluded_n=len(marker["unscorable"]),
            initial_correct=sum(row["initial_accepted"] for row in previous),
            final_correct=sum(row["accepted"] for row in previous),
            final_round=5,
            path_tokens=sum(row["total_tokens"] for row in previous),
            all_initial_tokens=bank.tokens([row["request_id"] for row in read(output / "initial/observations.jsonl")]),
            r4_program_candidates_tested=0,
            r4_program_candidates_submitted=0,
            paired_comparison=paired,
            feedback="pinned official single-database execution matching",
            independent_test=False,
            ablation_variant=VARIANT,
        )
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
    parser.add_argument("--finalize-only", action="store_true",
                        help="verify completed saved stages and write a missing final receipt")
    args = parser.parse_args()
    finalize(args.base, args.output) if args.finalize_only else run(args.base, args.output)


if __name__ == "__main__":
    main()
