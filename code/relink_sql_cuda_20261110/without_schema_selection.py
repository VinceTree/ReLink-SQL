"""Spider CUDA r1 full-schema ablation, with immutable and resumable records."""
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


VARIANT = "without_schema_selection_v1"


def read(path):
    with Path(path).open(encoding="utf-8") as stream:
        if Path(path).suffix == ".jsonl":
            return [json.loads(line) for line in stream if line.strip()]
        return json.load(stream)


class VariantBank:
    """Namespace changed requests while preserving semantic cache lookup."""
    def __init__(self, bank):
        self.bank = bank

    def __getattr__(self, key):
        return getattr(self.bank, key)

    def invoke(self, directory, specs, stopped=False):
        namespaced = copy.deepcopy(specs)
        for spec in namespaced:
            spec["request"]["id"] += "|" + VARIANT
        return self.bank.invoke(directory, namespaced, stopped)


def full_schema_round(engine, dest, previous, tasks, full, bank, evaluator):
    r = engine.r
    def build(task, state, context, stage="answer", auxiliary=None):
        if stage != "answer":
            raise ValueError("Unexpected r1 generation stage")
        return r.build(task, dict(state, method="FRESH14"), context, stage, auxiliary)
    proxy = SimpleNamespace(**vars(r.methods))
    proxy.initial_stage = lambda method: "answer" if method == "CURRENT" else r.methods.initial_stage(method)
    execute = r.methods.selectors.clone(r.round_run, methods=proxy, build=build)
    result = execute(dest, 1, previous, tasks, full, {}, bank, evaluator, engine.GATE)
    engine.validate_transition(previous, result, bank)
    return result


def copy_without_repair_stages(base, staging):
    def ignore(directory, names):
        current = Path(directory)
        if current == base:
            return {name for name in names if name in {"complete.json", "final_paths.jsonl", "summary.json"}}
        if current == base / "CURRENT":
            return {name for name in names if name in {"r1", "r2", "r3", "r4", "r5"}}
        return set()
    shutil.copytree(base, staging, ignore=ignore)


def main(base, output):
    base = Path(base).resolve()
    output = Path(output).resolve()
    root = base.parents[1]
    if base != root / "runs/spider_dev_CURRENT_20261020" or output.parent != root / "runs/ablations":
        raise ValueError("Unexpected source or output path")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        raise ValueError("CUDA_VISIBLE_DEVICES must be unset")
    code = root / "ReLink-SQL_CUDA_20261019"
    sys.path.insert(0, str(code))
    from cuda_repro import runner
    from cuda_repro.backend import CudaCallBank
    marker = runner.verify(base)
    audit = runner.audit(base)
    if (not audit.get("passed") or marker.get("sample_size") != 1034
            or marker["config"].get("device") != 0 or marker["config"].get("batch_size") != 1
            or read(base / "complete.json").get("final_round") != 5):
        raise ValueError("Source audit or fixed configuration failed")
    if any(p.is_symlink() for p in base.rglob("*")):
        raise ValueError("Source contains a symlink")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lockfile = output.parent / "gpu0_ablation.lock"
    if lockfile.is_symlink():
        raise ValueError("GPU lock is a symlink")
    with lockfile.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=uuid",
                                       "--format=csv,noheader"], text=True).strip()
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                        "--format=csv,noheader"], text=True)
        if any(line.split(",")[0].strip() == gpu for line in apps.splitlines()):
            raise ValueError("GPU 0 is occupied")
        script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        variant = dict(variant=VARIANT, source_run=str(base),
                       source_fingerprint=marker["run_fingerprint"], driver_sha256=script_sha,
                       r1="full schema, same question and prompt, no failed SQL or feedback",
                       later="original conditional r2-r5", max_repair_submissions=5)
        staging = output.with_name(output.name + ".staging")
        if not output.exists():
            if staging.exists():
                raise ValueError("Staging directory exists; preserve it for inspection")
            needed = sum(p.stat().st_size for p in base.rglob("*") if p.is_file()) + 1024**3
            if shutil.disk_usage(root).free < needed:
                raise ValueError("Insufficient disk space")
            copy_without_repair_stages(base, staging)
            runner.save(staging / "variant.json", variant)
            os.rename(staging, output)
        elif read(output / "variant.json") != variant:
            raise ValueError("Existing output belongs to another variant or driver")

        runner.verify(output)
        if (output / "complete.json").exists():
            result = runner.audit(output)
            runner.save(output / "audit.json", result)
            print(json.dumps({"already_complete": True, "audit": result,
                              "summary": read(output / "summary.json")}, ensure_ascii=False, indent=2))
            return
        cfg = dict(marker["config"], run_fingerprint=marker["run_fingerprint"])
        runner.configure_models(cfg)
        raw_bank = CudaCallBank(cfg, output)
        bank = VariantBank(raw_bank)
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
            seal = dest / f"r{stage}/cuda_artifacts.json"
            if not seal.exists():
                runner.seal(dest / f"r{stage}")
            print("STAGE COMPLETE", stage, "accepted", sum(row["accepted"] for row in previous), flush=True)
        runner.save(output / "final_paths.jsonl", previous)
        summary = dict(selected_n=len(tasks), scorable_n=len(previous), excluded_n=len(marker["unscorable"]),
                       initial_correct=sum(row["initial_accepted"] for row in previous),
                       final_correct=sum(row["accepted"] for row in previous), final_round=5,
                       path_tokens=sum(row["total_tokens"] for row in previous),
                       all_initial_tokens=bank.tokens([row["request_id"] for row in read(output / "initial/observations.jsonl")]),
                       feedback="pinned official single-database execution matching", independent_test=False,
                       ablation_variant=VARIANT)
        runner.save(output / "summary.json", summary)
        runner.save(output / "complete.json", dict(run_fingerprint=marker["run_fingerprint"], variant=VARIANT,
                    variant_sha256=runner.sha(output / "variant.json"),
                    final_paths_sha256=runner.sha(output / "final_paths.jsonl"),
                    summary_sha256=runner.sha(output / "summary.json"),
                    initial_paths_sha256=runner.sha(dest / "initial_paths.jsonl"),
                    ledger_events_sha256=runner.sha(output / "ledger/events.jsonl"),
                    final_round=5, stop_reason="fixed_five_stage_budget"))
        result = runner.audit(output)
        runner.save(output / "audit.json", result)
        print(json.dumps({"summary": summary, "audit": result, "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.base, args.output)
