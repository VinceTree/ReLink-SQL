#!/usr/bin/env python3
"""Public offline batch generation via local mlx_lm BatchGenerator, default 4.

`prepare` only reads files. `run` is the explicit GPU entry point. No downloads,
SQL parsing, tools, correctness labels, shared KV cache, or output postprocessing.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid
import numpy as np

FIELDS = {"id", "messages", "model", "revision", "max_tokens"}
ENCODINGS = ("stream_string", "explicit_no_special_tokens")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as handle:
        handle.write(text); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def append_event(path, row):
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def validate_requests(rows):
    result, ids = [], set()
    for row in rows:
        if set(row) != FIELDS:
            raise ValueError("Request fields must be exactly id/messages/model/revision/max_tokens")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in ids:
            raise ValueError("Missing/duplicate request ID")
        if not isinstance(row["model"], str) or not row["model"]:
            raise ValueError("Model must name an existing pinned snapshot or converted directory")
        if not isinstance(row["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", row["revision"]):
            raise ValueError("A pinned 40-hex revision is required")
        if type(row["max_tokens"]) is not int or row["max_tokens"] < 1:
            raise ValueError("max_tokens must be positive")
        if not isinstance(row["messages"], list) or not row["messages"]:
            raise ValueError("Nonempty public messages required")
        for message in row["messages"]:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in ("system", "user", "assistant")
                    or not isinstance(message["content"], str)):
                raise ValueError("Only text system/user/assistant messages are supported")
        ids.add(row["id"]); result.append(row)
    if not result:
        raise ValueError("Empty request file")
    return result


def batch_plan(requests, batch_size=4):
    if type(batch_size) is not int or not 1 <= batch_size <= 4:
        raise ValueError("batch_size must be between 1 and 4")
    groups = OrderedDict()
    for request in requests:
        groups.setdefault((request["model"], request["revision"]), []).append(request["id"])
    batches = []
    for (model, revision), ids in groups.items():
        for start in range(0, len(ids), batch_size):
            members = ids[start:start + batch_size]
            batches.append(dict(batch_id=digest([model, revision, members])[:20], model=model,
                                revision=revision, planned_members=members))
    return batches


def backend_metadata():
    """Discover local package sources without importing mlx or mlx_lm."""
    dist = importlib.metadata.distribution("mlx-lm")
    package = Path(dist.locate_file("mlx_lm"))
    files = {str(path.relative_to(package)): sha(path) for path in sorted(package.rglob("*.py"))}
    if not all(name in files for name in ("generate.py", "tokenizer_utils.py", "sample_utils.py", "models/cache.py")):
        raise ValueError("Incomplete local mlx_lm source tree")
    return dict(mlx_lm_version=dist.version, mlx_version=importlib.metadata.version("mlx"),
        source_directory=str(package), python_sources_sha256=files, code_sha256=digest(files),
        api="BatchGenerator.insert/next/close; GenerationBatch.Response", audited_version="0.31.3")


def model_metadata(model, revision):
    path = Path(model).expanduser()
    if not path.is_dir():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model):
            raise ValueError("Local model directory does not exist")
        path = Path.home() / ".cache/huggingface/hub" / ("models--" + model.replace("/", "--")) / "snapshots" / revision
    if not path.is_dir():
        raise ValueError("Pinned snapshot unavailable locally; this tool never downloads")
    path = path.absolute()  # Keep snapshot name instead of following cache file symlinks.
    if path.parent.name == "snapshots" and path.name == revision:
        source = "pinned_local_snapshot_path"
    elif (path / "share_provenance.json").exists():
        provenance = json.loads((path / "share_provenance.json").read_text())
        if provenance["revision"] != revision:
            raise ValueError("Converted model revision mismatch")
        source = "converted_share_provenance"
    else:
        raise ValueError("Unverified local model revision: use a pinned snapshot or SHARE conversion provenance")
    files = {name: sha(path / name) for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "generation_config.json", "model.safetensors.index.json", "share_provenance.json") if (path / name).is_file()}
    weights = {weight.name: dict(bytes=weight.stat().st_size, mtime_ns=weight.stat().st_mtime_ns,
                                 resolved_path=str(weight.resolve())) for weight in sorted(path.glob("*.safetensors"))}
    if "config.json" not in files or not weights:
        raise ValueError("Incomplete local model")
    return dict(local_path=str(path), revision=revision, revision_evidence=source, files_sha256=files,
                weight_inventory=weights, weights_rehashed=False)


def prepare(requests_path, output, batch_size=4, max_context_tokens=24000, prompt_encoding="stream_string"):
    if output.exists():
        raise ValueError("Use a new prepared output directory")
    if prompt_encoding not in ENCODINGS or max_context_tokens < 2:
        raise ValueError("Invalid prompt encoding/context limit")
    requests = validate_requests(read_lines(requests_path))
    if any(request["max_tokens"] >= max_context_tokens for request in requests):
        raise ValueError("Output cap leaves no room for a nonempty prompt")
    batches = batch_plan(requests, batch_size)
    models = {}
    for batch in batches:
        key = digest([batch["model"], batch["revision"]])
        if key not in models:
            models[key] = model_metadata(batch["model"], batch["revision"])
    backend = backend_metadata()
    if backend["mlx_lm_version"] != "0.31.3":
        raise ValueError("This adapter requires re-audit for a different mlx_lm version")
    manifest = dict(protocol="public-batch4-greedy-v1", requests_sha256=digest(requests),
        input_file_sha256=sha(requests_path), source=str(requests_path.resolve()), batches=batches,
        batch_size=batch_size, max_context_tokens=max_context_tokens, prompt_encoding=prompt_encoding,
        models=models, backend=backend, runner_sha256=sha(Path(__file__)), temperature=0,
        stop_policy="Only tokenizer EOS token IDs; no custom stop strings",
        accounting="Per completed item: full actual prompt token count; every returned generated token INCLUDING terminal EOS; EOS excluded from text. Batch wall times are shared, not per-item service cost.",
        restart="Completed item files are immutable. Incomplete members rerun in a new attempt with actual batch membership recorded. Interrupted work remains in events; no free-cost claim.",
        numerical_equivalence="Greedy batching can differ from scalar kernels; no bitwise/output equivalence claim or silent merge into scalar runs.")
    atomic_write(output / "requests.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in requests))
    atomic_write(output / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return dict(prepared=str(output), requests=len(requests), batches=len(batches), models_loaded=False)


def prepare_prompt(request, tokenizer, encoding, max_context_tokens):
    prompt = tokenizer.apply_chat_template(request["messages"], tokenize=False, add_generation_prompt=True)
    if encoding == "stream_string":
        # Exact stream_generate(str) BOS inference in the local 0.31.3 source.
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
    elif encoding == "explicit_no_special_tokens":
        add_special_tokens = False  # Matches SHARE's explicit integer prompt path.
    else:
        raise ValueError("Unknown prompt encoding")
    token_ids = list(tokenizer.encode(prompt, add_special_tokens=add_special_tokens))
    if not token_ids or len(token_ids) + request["max_tokens"] > max_context_tokens:
        raise ValueError("Empty/overflow prompt; no truncation")
    return dict(request=request, prompt_tokens=len(token_ids), prompt_token_ids=token_ids,
        rendered_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        messages_sha256=digest(request["messages"]), add_special_tokens=add_special_tokens)


def drain_batch(generator, uid_items, tokenizer, on_complete, progress=None):
    """Consume public backend response API; pure fake-compatible orchestration."""
    eos_ids = set(tokenizer.eos_token_ids)
    states = {uid: dict(tokens=[], selected_logprobs=[], pieces=[], detokenizer=tokenizer.detokenizer,
                        prompt_processed=0, completed=False) for uid in uid_items}
    if progress is not None:
        progress.update(states)
    remaining = set(uid_items)
    while remaining:
        prompt_responses, generated = generator.next()
        if not prompt_responses and not generated:
            raise RuntimeError("Backend stopped before all batch members completed")
        for response in prompt_responses:
            if response.uid not in states:
                raise ValueError("Unexpected prompt UID")
            states[response.uid]["prompt_processed"] = max(states[response.uid]["prompt_processed"], response.progress[0])
        for response in generated:
            if response.uid not in remaining:
                raise ValueError("Unknown or already completed generated UID")
            state, item = states[response.uid], uid_items[response.uid]
            state["tokens"].append(int(response.token))
            # Same observed-token sequence as scalar stream_generate, including
            # its terminal EOS response. Confidence is needed by frozen routers.
            lp=float(response.logprobs[int(response.token)])
            if not np.isfinite(lp): raise ValueError("Nonfinite generated-token log probability")
            state["selected_logprobs"].append(lp)
            count, cap = len(state["tokens"]), item["request"]["max_tokens"]
            finish = response.finish_reason
            if count > cap or finish not in (None, "stop", "length"):
                raise ValueError("Invalid backend token count/finish reason")
            is_eos = response.token in eos_ids
            if (finish == "stop") != is_eos or (finish == "length" and count != cap):
                raise ValueError("Backend EOS/length accounting differs from stream_generate")
            if not is_eos:
                state["detokenizer"].add_token(int(response.token))
            if finish is None:
                state["pieces"].append(state["detokenizer"].last_segment)
                continue
            state["detokenizer"].finalize()
            state["pieces"].append(state["detokenizer"].last_segment)
            text = "".join(state["pieces"])
            lps=np.asarray(state["selected_logprobs"],dtype=np.float32)
            result = dict(id=item["request"]["id"], text=text, raw_text=text,
                prompt_tokens=item["prompt_tokens"], generation_tokens=count,
                text_generation_tokens=count - int(is_eos), terminal_eos_count=int(is_eos),
                generated_token_ids=list(state["tokens"]), prompt_token_ids=item["prompt_token_ids"],
                finish_reason=finish, eos_token_ids=sorted(eos_ids),
                rendered_prompt_sha256=item["rendered_prompt_sha256"], messages_sha256=item["messages_sha256"],
                prompt_add_special_tokens=item["add_special_tokens"],
                selected_logprobs=list(state["selected_logprobs"]),
                logprob_mean=float(lps.mean()),logprob_min=float(lps.min()),
                logprob_q10=float(np.quantile(lps,.1)),low_confidence_fraction=float((lps<-.7).mean()),
                confidence_aggregation="selected-token logprobs including EOS; float32 numpy mean/min/quantile and lp<-.7 fraction")
            # Checkpoint callback runs as soon as an item completes, not after
            # the slowest member. Mark complete only after durable callback.
            on_complete(item, result)
            state["completed"] = True; remaining.remove(response.uid)
    return states


def item_path(output, ident):
    return output / "items" / (hashlib.sha256(ident.encode()).hexdigest() + ".json")


def load_items(output, requests, manifest_hash):
    by_request = {row["id"]: row for row in requests}
    result = {}
    for path in sorted((output / "items").glob("*.json")) if (output / "items").exists() else []:
        row = json.loads(path.read_text()); ident = row["id"]
        if ident not in by_request or ident in result or path != item_path(output, ident):
            raise ValueError("Unexpected/duplicate item checkpoint")
        if row["request_sha256"] != digest(by_request[ident]) or row["manifest_sha256"] != manifest_hash:
            raise ValueError("Stale item checkpoint")
        if row["generation_tokens"] != len(row["generated_token_ids"]) or row["prompt_tokens"] != len(row["prompt_token_ids"]):
            raise ValueError("Checkpoint token accounting mismatch")
        result[ident] = row
    return result


def export_items(output, requests, completed):
    atomic_write(output / "results.jsonl", "".join(json.dumps(completed[row["id"]], ensure_ascii=False) + "\n"
        for row in requests if row["id"] in completed))


def run(output):
    manifest = json.loads((output / "manifest.json").read_text())
    requests = validate_requests(read_lines(output / "requests.jsonl"))
    if digest(requests) != manifest["requests_sha256"] or sha(Path(__file__)) != manifest["runner_sha256"]:
        raise ValueError("Frozen request/runner changed")
    if (manifest["protocol"] != "public-batch4-greedy-v1" or manifest["temperature"] != 0
            or manifest["batches"] != batch_plan(requests, manifest["batch_size"])):
        raise ValueError("Frozen batch protocol/complete request membership changed")
    if backend_metadata() != manifest["backend"]:
        raise ValueError("Local MLX backend changed; prepare a new version")
    checked_models = set()
    for batch in manifest["batches"]:
        key = digest([batch["model"], batch["revision"]])
        if key in checked_models:
            continue
        if model_metadata(batch["model"], batch["revision"]) != manifest["models"][key]:
            raise ValueError("Pinned model metadata/weight inventory changed")
        checked_models.add(key)
    manifest_hash = sha(output / "manifest.json")
    completed = load_items(output, requests, manifest_hash)
    by_id = {row["id"]: row for row in requests}
    export_items(output, requests, completed)
    pending_batches = [batch for batch in manifest["batches"] if any(ident not in completed for ident in batch["planned_members"])]
    if not pending_batches:
        if not (output / "complete.json").exists():
            atomic_write(output / "complete.json", json.dumps(dict(items=len(completed), results_sha256=sha(output / "results.jsonl"),
                manifest_sha256=manifest_hash, recovered_after_all_item_checkpoints=True,
                incomplete_attempt_costs="See events.jsonl; completed results do not erase interrupted work"), indent=2) + "\n")
        return dict(completed=len(completed), new_calls=0, models_loaded=False)
    # Only this explicit `run` branch imports a model/GPU dependency.
    import gc
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler
    current_model_key = None; model = tokenizer = None; new_calls = 0
    events = output / "events.jsonl"
    try:
        for batch in pending_batches:
            members = [ident for ident in batch["planned_members"] if ident not in completed]
            model_key = digest([batch["model"], batch["revision"]])
            if current_model_key != model_key:
                model = tokenizer = None; gc.collect(); mx.clear_cache()
                started = time.perf_counter()
                model, tokenizer = load(manifest["models"][model_key]["local_path"],
                    tokenizer_config={"trust_remote_code": False, "local_files_only": True})
                mx.eval(model.parameters())
                append_event(events, dict(event="model_loaded", model=batch["model"], revision=batch["revision"],
                    seconds=time.perf_counter() - started, excluded_from_item_latency=True))
                current_model_key = model_key
            prepared = [prepare_prompt(by_id[ident], tokenizer, manifest["prompt_encoding"], manifest["max_context_tokens"]) for ident in members]
            attempt_id = uuid.uuid4().hex
            shared = dict(batch_id=batch["batch_id"], attempt_id=attempt_id, batch_members=members,
                planned_batch_members=batch["planned_members"], model=batch["model"], revision=batch["revision"],
                backend_code_sha256=manifest["backend"]["code_sha256"], manifest_sha256=manifest_hash)
            append_event(events, dict(shared, event="batch_started", prompt_tokens_requested={item["request"]["id"]: item["prompt_tokens"] for item in prepared},
                                      time_ns=time.time_ns()))
            generator = None; progress = {}; uid_items = {}; started = time.perf_counter()
            try:
                generator = BatchGenerator(model, sampler=make_sampler(temp=0),
                    stop_tokens=[[int(token)] for token in tokenizer.eos_token_ids],
                    completion_batch_size=manifest["batch_size"], prefill_batch_size=manifest["batch_size"],
                    prefill_step_size=2048)
                uids = generator.insert([item["prompt_token_ids"] for item in prepared],
                                        max_tokens=[item["request"]["max_tokens"] for item in prepared])
                if len(uids) != len(prepared) or len(set(uids)) != len(uids):
                    raise ValueError("Backend returned invalid request UIDs")
                uid_items = dict(zip(uids, prepared))
                def checkpoint(item, result):
                    nonlocal new_calls
                    ident = result["id"]
                    if ident in completed or item_path(output, ident).exists():
                        raise ValueError("Refusing to overwrite completed item")
                    row = dict(result, **shared, request_sha256=digest(item["request"]),
                        max_tokens=item["request"]["max_tokens"],
                        usage_scope="This completed attempt only; prior interrupted attempt usage remains in events.jsonl",
                        batch_elapsed_to_completion_seconds=time.perf_counter() - started,
                        time_note="Shared batch elapsed time including prefill/checkpoints; not an additive per-item service cost")
                    atomic_write(item_path(output, ident), json.dumps(row, ensure_ascii=False) + "\n")
                    completed[ident] = row; new_calls += 1
                drain_batch(generator, uid_items, tokenizer, checkpoint, progress)
                mx.synchronize()
                append_event(events, dict(shared, event="batch_completed", seconds=time.perf_counter() - started,
                    prompt_tokens=sum(completed[ident]["prompt_tokens"] for ident in members),
                    generation_tokens=sum(completed[ident]["generation_tokens"] for ident in members),
                    time_note="Shared wall time, includes prefill/token processing/checkpoint IO; model loading excluded"))
            except BaseException as exc:
                partial = {uid_items[uid]["request"]["id"]: dict(prompt_tokens_processed_observed=state["prompt_processed"],
                    generated_tokens_observed=len(state["tokens"]), generated_token_ids_observed=state["tokens"],
                    checkpoint_completed=uid_items[uid]["request"]["id"] in completed) for uid, state in progress.items()}
                append_event(events, dict(shared, event="batch_interrupted", exception_type=type(exc).__name__,
                    error=str(exc), seconds=time.perf_counter() - started, observed_partial_usage=partial,
                    incomplete_members=[ident for ident in members if ident not in completed],
                    cost_note="Incomplete observed usage is a lower bound, not zero. A later retry is an additional attempt."))
                raise
            finally:
                if generator is not None:
                    generator.close()
                export_items(output, requests, completed)
    finally:
        export_items(output, requests, completed)
    if set(completed) != set(by_id):
        raise RuntimeError("Not every request has a completed checkpoint")
    atomic_write(output / "complete.json", json.dumps(dict(items=len(completed), results_sha256=sha(output / "results.jsonl"),
        manifest_sha256=manifest_hash, events_sha256=sha(events), new_calls_this_invocation=new_calls,
        incomplete_attempt_costs="See events; completed results alone do not erase interrupted work"), indent=2) + "\n")
    return dict(completed=len(completed), new_calls=new_calls, models_loaded=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--requests", dest="requests_path", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--batch-size", type=int, default=4)
    prep.add_argument("--max-context-tokens", type=int, default=24000)
    prep.add_argument("--prompt-encoding", choices=ENCODINGS, default="stream_string")
    execute_parser = sub.add_parser("run")
    execute_parser.add_argument("--output", type=Path, required=True)
    arguments = vars(parser.parse_args()); command = arguments.pop("command")
    print(json.dumps((prepare if command == "prepare" else run)(**arguments), ensure_ascii=False))
