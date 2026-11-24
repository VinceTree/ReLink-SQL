"""BIRD CUDA ablation: use the full schema for r1, then retain original r2-r5."""
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
from types import SimpleNamespace


VARIANT = "bird_without_schema_selection_v1"
SUFFIX = "|BIRD_NO_SCHEMA_SELECTION_20261111"


def read(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


class VariantBank:
    def __init__(self, bank):
        self.bank = bank

    def __getattr__(self, name):
        return getattr(self.bank, name)

    def invoke(self, directory, specs, stopped=False):
        specs = copy.deepcopy(specs)
        for spec in specs:
            spec["request"]["id"] += SUFFIX
        return self.bank.invoke(directory, specs, stopped)


def full_schema_round(engine, dest, previous, tasks, full, bank, evaluator):
    r = engine.r

    def build(task, state, context, stage="answer", auxiliary=None):
        if stage != "answer":
            raise ValueError("Unexpected r1 generation stage")
        return r.build(task, dict(state, method="FRESH14"), context, stage, auxiliary)

    methods = SimpleNamespace(**vars(r.methods))
    methods.initial_stage = lambda method: "answer" if method == "CURRENT" else r.methods.initial_stage(method)
    execute = r.methods.selectors.clone(r.round_run, methods=methods, build=build)
    result = execute(dest, 1, previous, tasks, full, {}, bank, evaluator, engine.GATE)
    engine.validate_transition(previous, result, bank)
    return result


def preflight_summary(lengths, context_limit):
    ordered = sorted(lengths, key=lambda row: row["prompt_tokens"])
    overflow = [row for row in lengths if row["prompt_tokens"] + row["max_new_tokens"] > context_limit]
    if ordered:
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] ["prompt_tokens"]
        maximum = ordered[-1]["prompt_tokens"]
    else:
        p95 = maximum = 0
    frozen_rows = sorted(lengths, key=lambda row: row["task_id"])
    return dict(
        pending_r1=len(lengths),
        max_prompt_tokens=maximum,
        p95_prompt_tokens=p95,
        context_limit=context_limit,
        generation_cap=lengths[0]["max_new_tokens"] if lengths else 1024,
        overflow_n=len(overflow),
        overflow_task_ids=[row["task_id"] for row in overflow],
        prompt_lengths_sha256=hashlib.sha256(json.dumps(
            frozen_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    )


def source_check(base):
    base = Path(base).resolve()
    root = base.parents[1]
    if base != root / "runs/bird_dev_CURRENT_20261110":
        raise SystemExit("STOP: source is not the audited BIRD CURRENT run")
    sys.path.insert(0, str(root / "ReLink-SQL_CUDA_20261019"))
    from cuda_repro import runner
    marker = runner.verify(base)
    audit = runner.audit(base)
    complete = read(base / "complete.json")
    if (not audit.get("passed") or marker.get("protocol") != "isolated-cuda-bird-dev-v1"
            or marker.get("sample_size") != 1534 or complete.get("final_round") != 5):
        raise SystemExit("STOP: BIRD source is not the audited full five-stage run")
    if marker["config"].get("device") != 0 or marker["config"].get("batch_size") != 1:
        raise SystemExit("STOP: source configuration is not GPU 0 / batch 1")
    selected = read(base / "selection.json")["rows"]
    excluded = {row["source_row"] for row in selected if row["id"] in marker["unscorable"]}
    if excluded != {384, 518, 693, 701, 1131}:
        raise SystemExit("STOP: BIRD exclusion set differs from the audited baseline")
    return root, runner, marker, audit


def shared_prefix_files():
    # r1 is the intervention; only the shared r0 artifacts belong in the prefix.
    return ["initial/observations.jsonl", "CURRENT/initial_paths.jsonl"]


def measure_r1_prompts(base, runner, marker):
    tasks = {row["id"]: row for row in read(base / "public.jsonl")}
    full = read(base / "full_contexts.json")
    initial_paths = read(base / "CURRENT/initial_paths.jsonl")
    pending = [row for row in initial_paths if not row["accepted"]]
    config = marker["config"]
    model_info = marker["model_inventory"]["strong14"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_info["path"], local_files_only=True, trust_remote_code=False)
    lengths = []
    for index, row in enumerate(pending, 1):
        task = tasks[row["task_id"]]
        state = runner.engine.r.public_state(row, 1)
        messages, metadata = runner.r.build(
            task, dict(state, method="FRESH14"), full[row["task_id"]], "answer")
        spec = runner.r.spec(state, "answer", messages, metadata, tag="bird_dev_17")
        prompt = tokenizer.apply_chat_template(
            spec["request"]["messages"], tokenize=False, add_generation_prompt=True)
        special = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
        token_count = len(tokenizer.encode(prompt, add_special_tokens=special))
        lengths.append(dict(task_id=row["task_id"], prompt_tokens=token_count,
                            max_new_tokens=spec["request"]["max_tokens"]))
        if index % 100 == 0 or index == len(pending):
            print("PREFLIGHT R1 PROMPTS", index, "/", len(pending), flush=True)
    return preflight_summary(lengths, config.get("max_context_tokens", 24000))


def variant_manifest(base, marker, audit, preflight):
    names = shared_prefix_files()
    return dict(
        variant=VARIANT,
        source_run=str(base),
        source_run_fingerprint=marker["run_fingerprint"],
        source_run_audit=audit,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        request_suffix=SUFFIX,
        intervention="Full database schema in the original FRESH14 r1 prompt; unchanged conditional r2-r5",
        preflight=preflight,
        shared_prefix_sha256={name: hashlib.sha256((base / name).read_bytes()).hexdigest()
                              for name in names},
        common_initial_answers=True,
        max_repair_submissions=5,
    )


def assert_shared_prefix(base, output, manifest):
    actual = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
              for name in manifest["shared_prefix_sha256"]}
    if actual != manifest["shared_prefix_sha256"]:
        raise ValueError("Copied r0-r3 prefix differs from the BIRD baseline")


def copy_without_repair_stages(base, staging):
    def ignore(directory, names):
        current = Path(directory)
        if current == base:
            return {name for name in names if name in {"complete.json", "final_paths.jsonl", "summary.json", "audit.json"}}
        if current == base / "CURRENT":
            return {name for name in names if name in {"r1", "r2", "r3", "r4", "r5"}}
        return set()
    shutil.copytree(base, staging, ignore=ignore)


def paired_comparison(baseline_paths, variant_paths):
    base = {row["task_id"]: row for row in baseline_paths}
    variant = {row["task_id"]: row for row in variant_paths}
    if base.keys() != variant.keys():
        raise ValueError("Baseline and ablation scorable task sets differ")
    ids = sorted(base)
    full_only = sum(base[tid]["accepted"] and not variant[tid]["accepted"] for tid in ids)
    variant_only = sum(variant[tid]["accepted"] and not base[tid]["accepted"] for tid in ids)
    base_tokens = sum(row["total_tokens"] for row in baseline_paths)
    variant_tokens = sum(row["total_tokens"] for row in variant_paths)
    return dict(
        paired_scorable_n=len(ids),
        baseline_final_correct=sum(row["accepted"] for row in baseline_paths),
        variant_final_correct=sum(row["accepted"] for row in variant_paths),
        full_only_successes=full_only,
        ablation_only_successes=variant_only,
        net_correct_delta_variant_minus_baseline=variant_only-full_only,
        baseline_path_tokens=base_tokens,
        variant_path_tokens=variant_tokens,
        path_token_delta_variant_minus_baseline=variant_tokens-base_tokens,
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


def summarize(output, base, marker, paths, bank):
    tasks = read(output / "public.jsonl")
    paired = paired_comparison(read(base / "final_paths.jsonl"), paths)
    return dict(
        selected_n=len(tasks),
        scorable_n=len(paths),
        excluded_n=len(marker["unscorable"]),
        initial_correct=sum(row["initial_accepted"] for row in paths),
        final_correct=sum(row["accepted"] for row in paths),
        final_round=5,
        path_tokens=sum(row["total_tokens"] for row in paths),
        all_initial_tokens=bank.tokens([row["request_id"] for row in
                                        read(output / "initial/observations.jsonl")]),
        paired_comparison=paired,
        feedback="pinned official single-database execution matching",
        independent_test=False,
        ablation_variant=VARIANT,
    ), paired


def finalize(base, output):
    base, output = Path(base).resolve(), Path(output).resolve()
    root, runner, source_marker, source_audit = source_check(base)
    expected = root / "runs/ablations/bird_without_schema_selection_20261111"
    if output != expected:
        raise SystemExit("STOP: finalization path is not the designated BIRD schema ablation")
    preflight = measure_r1_prompts(base, runner, source_marker)
    if preflight["overflow_n"]:
        raise SystemExit("STOP: r1 full-schema prompts exceed the frozen context limit")
    manifest = variant_manifest(base, source_marker, source_audit, preflight)
    if read(output / "variant.json") != manifest:
        raise SystemExit("STOP: saved variant manifest differs from this protocol/preflight")
    assert_shared_prefix(base, output, manifest)
    marker = runner.verify(output)
    dest = output / "CURRENT"
    for stage in range(1, 6):
        if not (dest / f"r{stage}/complete.json").is_file():
            raise SystemExit(f"STOP: missing completed r{stage} stage")
    paths = read(output / "final_paths.jsonl")
    if paths != read(dest / "r5/paths.jsonl"):
        raise SystemExit("STOP: final paths differ from committed r5 paths")
    from cuda_repro.backend import CudaCallBank
    config = dict(marker["config"], run_fingerprint=marker["run_fingerprint"])
    runner.configure_models(config)
    bank = CudaCallBank(config, output, readonly=True)
    summary, paired = summarize(output, base, marker, paths, bank)
    if read(output / "summary.json") != summary or len(paths) != 1529:
        raise SystemExit("STOP: saved summary or scorable count mismatch")
    if not (output / "complete.json").exists():
        runner.save(output / "complete.json", completion_record(
            output, marker["run_fingerprint"], paired, runner.sha))
    audit = runner.audit(output)
    runner.save(output / "audit.json", audit)
    print(json.dumps({"finalized_saved_run": True, "preflight": preflight,
                      "summary": summary, "audit": audit, "output": str(output)},
                     ensure_ascii=False, indent=2))


def run(base, output):
    base, output = Path(base).resolve(), Path(output).resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        raise SystemExit("STOP: CUDA_VISIBLE_DEVICES must be unset")
    root, runner, source_marker, source_audit = source_check(base)
    expected_parent = root / "runs/ablations"
    if output.parent != expected_parent:
        raise SystemExit("STOP: output must be a child of runs/ablations")
    preflight = measure_r1_prompts(base, runner, source_marker)
    print("R1 FULL-SCHEMA PREFLIGHT", json.dumps(preflight, ensure_ascii=False), flush=True)
    if preflight["overflow_n"]:
        raise SystemExit("STOP: one or more full-schema r1 prompts overflow; no run was cloned or inferred")
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
        manifest = variant_manifest(base, source_marker, source_audit, preflight)
        staging = output.with_name(output.name + ".staging")
        if output.exists():
            if read(output / "variant.json") != manifest:
                raise SystemExit("STOP: existing output identity differs")
        else:
            if staging.exists():
                raise SystemExit("STOP: staging exists; preserve it for inspection")
            print("CLONE BASELINE TO VARIANT STAGING", flush=True)
            copy_without_repair_stages(base, staging)
            runner.save(staging / "variant.json", manifest)
            os.rename(staging, output)
            print("VARIANT CLONE COMPLETE", flush=True)

        assert_shared_prefix(base, output, manifest)
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
        full = read(output / "full_contexts.json")
        evaluator = runner.r.Evaluator(output / "private/references.json")
        dest = output / "CURRENT"
        previous = read(dest / "initial_paths.jsonl")
        for stage in range(1, 6):
            print("STAGE START", stage, flush=True)
            if stage == 1:
                previous = full_schema_round(runner.engine, dest, previous, valid, full, bank, evaluator)
            else:
                previous = runner.engine.run_round(dest, stage, previous, valid, full, {}, bank, evaluator)
            if not (dest / f"r{stage}/cuda_artifacts.json").exists():
                runner.seal(dest / f"r{stage}")
            print("STAGE COMPLETE", stage, "accepted", sum(row["accepted"] for row in previous), flush=True)
        runner.save(output / "final_paths.jsonl", previous)
        summary, paired = summarize(output, base, marker, previous, bank)
        runner.save(output / "summary.json", summary)
        runner.save(output / "complete.json", completion_record(
            output, marker["run_fingerprint"], paired, runner.sha))
        audit = runner.audit(output)
        runner.save(output / "audit.json", audit)
        print(json.dumps({"summary": summary, "audit": audit, "output": str(output)},
                         ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("preflight")
    prep.add_argument("--base", type=Path, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--base", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    finish = sub.add_parser("finalize-only")
    finish.add_argument("--base", type=Path, required=True)
    finish.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        root, runner, marker, audit = source_check(args.base)
        print(json.dumps(measure_r1_prompts(args.base, runner, marker), ensure_ascii=False, indent=2))
    elif args.command == "run":
        run(args.base, args.output)
    else:
        finalize(args.base, args.output)


if __name__ == "__main__":
    main()
