"""Isolated, local-only CUDA generation with durable per-call accounting.

Importing this module, validating assets and reopening a ledger need only Python's
standard library. No historical MLX cache is consulted. Failed attempt usage in
events.jsonl is a lower bound, separate from completed logical-call accounting.
"""
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import uuid


REPOS = {"strong": "Qwen/Qwen2.5-7B-Instruct", "strong14": "Qwen/Qwen2.5-14B-Instruct"}
QUANTIZATION = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_use_double_quant": False}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _snapshot(info, repo):
    revision = info.get("revision", "")
    if info.get("repo") != repo or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Expected the official Qwen repo and a pinned 40-hex revision")
    path = Path(info.get("path", "")).expanduser().absolute()
    if (path.name != revision or path.parent.name != "snapshots" or
            path.parent.parent.name != "models--" + repo.replace("/", "--") or not path.is_dir()):
        raise ValueError("Expected a local Hugging Face models--repo/snapshots/revision directory")
    small = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        file = path / name
        if not file.is_file() or not file.stat().st_size:
            raise ValueError("Missing local model asset: " + str(file))
        content = file.read_bytes()
        json.loads(content)
        small[name] = hashlib.sha256(content).hexdigest()
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "qwen2" or config.get("quantization") or config.get("quantization_config"):
        raise ValueError("Expected unquantized official Qwen2 weights; MLX/other prequantized weights are unsupported")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        contents = index.read_bytes()
        small[index.name] = hashlib.sha256(contents).hexdigest()
        weight_map = json.loads(contents).get("weight_map", {})
        if not weight_map:
            raise ValueError("Empty safetensors weight index")
        names = sorted(set(weight_map.values()))
    else:
        names = ["model.safetensors"]
    weights = {}
    for name in names:
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Invalid safetensors shard name")
        weight = path / name
        if not weight.is_file() or not weight.stat().st_size:
            raise ValueError("Missing safetensors shard: " + str(weight))
        stat = weight.stat()
        weights[name] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return dict(repo=repo, revision=revision, path=str(path), small_files_sha256=small,
                weights=weights, weights_rehashed=False)


def validate_config(config):
    """Validate configuration/local assets without importing any GPU dependency."""
    if type(config.get("batch_size", 1)) is not int or config.get("batch_size", 1) != 1:
        raise ValueError("This CUDA adapter supports batch_size=1 only")
    if type(config.get("device", 0)) is not int or config.get("device", 0) < 0:
        raise ValueError("device must be a nonnegative CUDA device index")
    limit = config.get("max_context_tokens", 24000)
    if type(limit) is not int or limit < 2:
        raise ValueError("max_context_tokens must be an integer >=2")
    if config.get("quantization") != QUANTIZATION:
        raise ValueError("Unsupported quantization configuration")
    if set(config.get("models", {})) != set(REPOS):
        raise ValueError("Exactly strong and strong14 model configurations are required")
    return {key: _snapshot(config["models"][key], repo) for key, repo in REPOS.items()}


def _immutable(path, value, lines=False):
    """Atomically publish a new immutable file; never replace an existing file."""
    path = Path(path)
    content = ("".join(_json(row) + "\n" for row in value) if lines else _json(value) + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise ValueError("Refusing changed committed artifact: " + str(path))
        return
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text() != content:
                raise ValueError("Conflicting committed artifact: " + str(path))
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.unlink(temporary)


def _ids(value):
    values = value.tolist() if hasattr(value, "tolist") else value
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("Only one sequence may be generated")
        values = values[0]
    if any(type(token) is not int or token < 0 for token in values):
        raise ValueError("Invalid token IDs")
    return list(values)


class _TokenObserver:
    def __init__(self, prompt):
        self.prompt = prompt
        self.prompt_submitted = False
        self.generated = []
        self.ended = False

    def put(self, value):
        tokens = _ids(value)
        if not self.prompt_submitted:
            if tokens != self.prompt:
                raise ValueError("Streamer prompt differs from the encoded prompt")
            self.prompt_submitted = True
        else:
            self.generated.extend(tokens)

    def end(self):
        self.ended = True


class CudaCallBank:
    def __init__(self, config, run_dir, readonly=False):
        self.config = json.loads(_json(config))
        self.models = validate_config(self.config)
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.ledger = self.run_dir / "ledger"
        self.readonly = readonly
        self.fingerprint = _digest(dict(config=self.config, models=self.models, backend_protocol="cuda-greedy-single-v1"))
        manifest = dict(bank_fingerprint=self.fingerprint, run_fingerprint=self.config.get("run_fingerprint"),
                        config=self.config, models=self.models)
        if readonly:
            path = self.ledger / "manifest.json"
            if not path.is_file() or json.loads(path.read_text()) != manifest:
                raise ValueError("Missing or changed CUDA ledger manifest")
        else:
            _immutable(self.ledger / "manifest.json", manifest)
        self.raw, self.index, self.sources = {}, {}, {}
        self._current_key = None
        self._model = self._tokenizer = None
        self._refresh()

    def _refresh(self):
        found = set()
        for path in sorted((self.ledger / "items").glob("*.json")):
            row = json.loads(path.read_text())
            self._validate_row(row, path)
            found.add(row["id"])
            self._register(row, path)
        if set(self.raw) - found:
            raise ValueError("A completed CUDA checkpoint disappeared")

    def key(self, task_id, request, stopped=False):
        return _digest([self.fingerprint, task_id, request["model"], request["revision"],
                        request["messages"], request["max_tokens"], stopped])

    def _model_key(self, request):
        for key, info in self.models.items():
            if (request.get("model"), request.get("revision")) == (info["repo"], info["revision"]):
                return key
        raise ValueError("Unknown model/revision; only configured pinned models may be used")

    def _validate_request(self, request):
        if set(request) != {"id", "messages", "model", "revision", "max_tokens"}:
            raise ValueError("Unexpected generation request fields")
        if not isinstance(request["id"], str) or not request["id"]:
            raise ValueError("A nonempty request ID is required")
        self._model_key(request)
        cap = request["max_tokens"]
        if type(cap) is not int or not 0 < cap < self.config.get("max_context_tokens", 24000):
            raise ValueError("Invalid generation token cap")
        if not request["messages"]:
            raise ValueError("An empty chat is unsupported")
        for message in request["messages"]:
            if (set(message) != {"role", "content"} or message["role"] not in ("system", "user", "assistant")
                    or not isinstance(message["content"], str)):
                raise ValueError("Only plain system/user/assistant chat messages are supported")

    def _item_path(self, request_id):
        return self.ledger / "items" / (hashlib.sha256(request_id.encode()).hexdigest() + ".json")

    def _validate_row(self, row, path):
        request = row["request"]
        self._validate_request(request)
        if (row.get("record_sha256") != _digest({k: v for k, v in row.items() if k != "record_sha256"}) or
                row["id"] != request["id"] or path != self._item_path(row["id"]) or
                row["bank_fingerprint"] != self.fingerprint or row["request_sha256"] != _digest(request) or
                row["messages_sha256"] != _digest(request["messages"]) or
                row["prompt_tokens"] != len(row["prompt_token_ids"]) or
                row["generation_tokens"] != len(row["generated_token_ids"])):
            raise ValueError("Stale or inconsistent CUDA call checkpoint")
        self._check_completion(row["generated_token_ids"], row["eos_token_ids"], request["max_tokens"])
        terminal = int(row["generated_token_ids"][-1] in row["eos_token_ids"])
        if (row["terminal_eos_count"] != terminal or row["text_generation_tokens"] != row["generation_tokens"] - terminal
                or row["finish_reason"] != ("stop" if terminal else "length")):
            raise ValueError("Invalid EOS/token accounting in CUDA checkpoint")

    def _register(self, row, path):
        self.raw[row["id"]] = row
        self.index.setdefault(self.key(row["task_id"], row["request"]), row["id"])
        self.sources[row["id"]] = str(path)

    def _event(self, event, **values):
        self.ledger.mkdir(parents=True, exist_ok=True)
        with (self.ledger / "events.jsonl").open("a") as stream:
            stream.write(_json(dict(event=event, time_ns=time.time_ns(), **values)) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @contextmanager
    def _lock(self):
        import fcntl
        with (self.ledger / "inference.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def _load_model(self, key):
        """The only model-loading path. All downloads and remote code are disabled."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        device = self.config.get("device", 0)
        if not torch.cuda.is_available() or device >= torch.cuda.device_count():
            raise RuntimeError("Configured CUDA device is unavailable")
        if torch.cuda.get_device_capability(device) < (8, 0):
            raise RuntimeError("This configuration requires native CUDA bfloat16 support")
        path = self.models[key]["path"]
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        quantization = dict(self.config["quantization"], bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                device_map={"": device}, torch_dtype=torch.bfloat16,
                quantization_config=BitsAndBytesConfig(**quantization), attn_implementation="sdpa")
        model.eval()
        return model, tokenizer

    def _get_model(self, key):
        if _snapshot(self.config["models"][key], REPOS[key]) != self.models[key]:
            raise ValueError("Local model assets changed after ledger preparation")
        if self._current_key != key:
            self._model = self._tokenizer = None
            self._current_key = None
            gc.collect()
            # This branch is reached only on an actual inference path.
            import sys
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            print('CUDA MODEL LOAD', self.models[key]['repo'], flush=True)
            self._model, self._tokenizer = self._load_model(key)
            self._current_key = key
        return self._model, self._tokenizer

    @staticmethod
    def _eos_ids(model, tokenizer):
        result = set()
        for value in (tokenizer.eos_token_id, getattr(model.generation_config, "eos_token_id", None)):
            if value is not None:
                result.update(value if isinstance(value, (tuple, list)) else [value])
        if not result or any(type(token) is not int or token < 0 for token in result):
            raise ValueError("Model/tokenizer must define valid EOS token IDs")
        return sorted(result)

    def _generate(self, model, tokenizer, prompt_ids, cap, observer, eos_ids):
        import torch
        from transformers import GenerationConfig
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device="cuda:" + str(self.config.get("device", 0)))
        generation = GenerationConfig(max_new_tokens=cap, do_sample=False, num_beams=1,
            num_return_sequences=1, repetition_penalty=1.0, eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0],
            bos_token_id=tokenizer.bos_token_id, use_cache=True, output_scores=False,
            output_logits=False, return_dict_in_generate=False)
        with torch.inference_mode():
            result = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                    generation_config=generation, streamer=observer)
        torch.cuda.synchronize(self.config.get("device", 0))
        return _ids(result)

    @staticmethod
    def _check_completion(tokens, eos_ids, cap):
        if not tokens or len(tokens) > cap or any(token in eos_ids for token in tokens[:-1]):
            raise ValueError("Invalid emitted token sequence/EOS position")
        if tokens[-1] not in eos_ids and len(tokens) != cap:
            raise ValueError("Generation stopped before EOS or the exact requested token cap")

    def _invoke_one(self, task_id, request):
        attempt_id = uuid.uuid4().hex
        shared = dict(attempt_id=attempt_id, request_id=request["id"], task_id=task_id,
                      request_sha256=_digest(request), bank_fingerprint=self.fingerprint)
        self._event("attempt_started", **shared)
        print('CUDA CALL', request['id'], flush=True)
        prompt_ids, observer = [], None
        try:
            model, tokenizer = self._get_model(self._model_key(request))
            prompt = tokenizer.apply_chat_template(request["messages"], tokenize=False, add_generation_prompt=True)
            special = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
            prompt_ids = _ids(tokenizer.encode(prompt, add_special_tokens=special))
            cap = request["max_tokens"]
            if not prompt_ids or len(prompt_ids) + cap > self.config.get("max_context_tokens", 24000):
                raise ValueError("Empty/overflow prompt; no truncation is permitted")
            observer = _TokenObserver(prompt_ids)
            eos_ids = self._eos_ids(model, tokenizer)
            self._event("generation_requested", **shared, prompt_tokens_requested=len(prompt_ids),
                        prompt_token_ids=prompt_ids, max_new_tokens=cap,
                        note="Requested prompt is not a measurement of processed prefill tokens")
            sequence = self._generate(model, tokenizer, prompt_ids, cap, observer, eos_ids)
            if (not observer.prompt_submitted or not observer.ended or
                    sequence != prompt_ids + observer.generated):
                raise ValueError("Streamer tokens and returned generation disagree")
            self._check_completion(observer.generated, eos_ids, cap)
            terminal = int(observer.generated[-1] in eos_ids)
            text_ids = observer.generated[:-1] if terminal else observer.generated
            text = tokenizer.decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            row = dict(id=request["id"], task_id=task_id, request=request, model=request["model"],
                revision=request["revision"], prompt_token_ids=prompt_ids,
                generated_token_ids=observer.generated, prompt_tokens=len(prompt_ids),
                generation_tokens=len(observer.generated), terminal_eos_count=terminal,
                text_generation_tokens=len(text_ids), finish_reason="stop" if terminal else "length",
                eos_token_ids=eos_ids, text=text, raw_text=text, prompt_add_special_tokens=special,
                rendered_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                messages_sha256=_digest(request["messages"]), request_sha256=_digest(request),
                bank_fingerprint=self.fingerprint, run_fingerprint=self.config.get("run_fingerprint"),
                attempt_id=attempt_id, usage_scope="Completed attempt only; failed attempts remain in events.jsonl")
            row["record_sha256"] = _digest(row)
            path = self._item_path(request["id"])
            _immutable(path, row)
            self._register(row, path)
            self._event("attempt_completed", **shared, prompt_tokens=row["prompt_tokens"],
                        generation_tokens=row["generation_tokens"])
            print('CUDA CALL COMPLETE', request['id'], 'tokens', row['prompt_tokens'],
                  row['generation_tokens'], row['finish_reason'], flush=True)
            return row
        except BaseException as exc:
            self._event("attempt_failed", **shared, error_type=type(exc).__name__, error=str(exc),
                prompt_tokens_requested=len(prompt_ids), prompt_tokens_processed_observed=None,
                generated_token_ids_observed=observer.generated if observer else [],
                generation_tokens_observed=len(observer.generated) if observer else 0,
                usage_note="Observed lower bound, not zero/free usage; prefill is unmeasured. Retrying is a new attempt.")
            # Do not retain a potentially unhealthy model after CUDA/generation failure.
            self._model = self._tokenizer = None
            self._current_key = None
            raise

    def invoke(self, directory, specs, stopped=False):
        if self.readonly:
            raise ValueError("A readonly CUDA ledger cannot invoke generation")
        if stopped:
            raise ValueError("SHARE string-stop requests are not supported by this CUDA pilot")
        directory = Path(directory).resolve()
        if not directory.is_relative_to(self.run_dir):
            raise ValueError("Generation records must remain inside this run directory")
        for spec in specs:
            self._validate_request(spec["request"])
            if not isinstance(spec["task_id"], str) or not spec["task_id"]:
                raise ValueError("A nonempty task ID is required")
        with self._lock():
            # A bank constructed before another process completed must refresh
            # under the lock before deciding whether a new model call is needed.
            self._refresh()
            _immutable(directory / "logical_requests.jsonl", specs, lines=True)
            record_path = directory / "records.jsonl"
            if record_path.exists():
                saved = [json.loads(line) for line in record_path.read_text().splitlines() if line.strip()]
                if any(ident not in self.raw for row in saved for ident in row["request_ids"]):
                    raise ValueError("Stage records refer to missing completed CUDA checkpoints")
            rows = []
            for spec in specs:
                request = spec["request"]
                if request["id"] in self.raw and (self.raw[request["id"]]["request_sha256"] != _digest(request)
                        or self.raw[request["id"]]["task_id"] != spec["task_id"]):
                    raise ValueError("A completed request ID was reused with changed inputs")
                semantic_key = self.key(spec["task_id"], request)
                if semantic_key in self.index:
                    raw = self.raw[self.index[semantic_key]]
                else:
                    raw = self._invoke_one(spec["task_id"], request)
                row = dict(task_id=spec["task_id"], method=spec["method"], round=spec["round"],
                    stage=spec["stage"], metadata=spec.get("metadata", {}), request_id=raw["id"],
                    request_ids=[raw["id"]], text=raw["text"], raw_text=raw["raw_text"],
                    finish_reason=raw["finish_reason"], logical_request_id=request["id"],
                    source_checkpoint=self.sources[raw["id"]])
                if raw["id"] != request["id"]:
                    row["reused_from"] = self.sources[raw["id"]]
                rows.append(row)
            _immutable(record_path, rows, lines=True)
            return rows

    def tokens(self, ids):
        return sum(self.raw[ident]["prompt_tokens"] + self.raw[ident]["generation_tokens"] for ident in ids)
