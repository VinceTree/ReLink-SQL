"""Fake backend + actual local stream function AST; no MLX import or GPU."""
import ast
import contextlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import types
from typing import List, Union

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_public_generate as module


def request(ident="a", model="local/model", cap=8):
    return dict(id=ident, messages=[dict(role="user", content="Public")], model=model, revision="a" * 40, max_tokens=cap)


class Detokenizer:
    def __init__(self):
        self.text, self.offset = "", 0
    def add_token(self, token):
        self.text += {1: "A", 2: "中", 3: "B"}[token]
    def finalize(self):
        pass
    @property
    def last_segment(self):
        segment = self.text[self.offset:]; self.offset = len(self.text)
        return segment


class Tokenizer:
    eos_token_ids = {0}
    bos_token = "<BOS>"
    rendered = "<BOS>prompt"
    def __init__(self):
        self.encodings = []
    def apply_chat_template(self, messages, **kwargs):
        return self.rendered
    def encode(self, text, **kwargs):
        self.encodings.append(kwargs)
        return [8, 9, 10]
    @property
    def detokenizer(self):
        return Detokenizer()


class Backend:
    def __init__(self, steps):
        self.steps = iter(steps)
    def next(self):
        step = next(self.steps)
        if isinstance(step, BaseException):
            raise step
        return step


def response(uid, token, finish=None):
    return types.SimpleNamespace(uid=uid, token=token, finish_reason=finish,logprobs=[-.2,-1.,-.6,-.8])


def local_stream_function():
    """Run the installed stream_generate Python control flow with fake arrays/model."""
    path = Path(importlib.metadata.distribution("mlx-lm").locate_file("mlx_lm/generate.py"))
    node = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "stream_generate")
    class Array:
        def __init__(self, values):
            self.values = values; self.size = len(values)
    namespace = dict(nn=types.SimpleNamespace(Module=object), Union=Union, List=List,
        PreTrainedTokenizer=Tokenizer, TokenizerWrapper=Tokenizer, Optional=__import__("typing").Optional,
        Generator=__import__("typing").Generator,
        mx=types.SimpleNamespace(array=Array, get_peak_memory=lambda: 0), time=time,
        wired_limit=lambda *args: contextlib.nullcontext(), generation_stream=None,
        generate_step=lambda prompt, model, **kwargs: iter((token, None) for token in model[:kwargs["max_tokens"]]),
        GenerationResponse=types.SimpleNamespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["stream_generate"]


def test_per_item_eos_length_and_text_match_actual_stream_control_flow():
    tokenizer = Tokenizer()
    uid_items = {uid: module.prepare_prompt(request(str(uid), cap=cap), tokenizer, "stream_string", 50)
                 for uid, cap in ((0, 8), (1, 2), (2, 1))}
    backend = Backend([
        ([types.SimpleNamespace(uid=0, progress=(3, 3))], []),
        ([], [response(0, 1), response(1, 2), response(2, 0, "stop")]),
        ([], [response(0, 0, "stop"), response(1, 3, "length")])])
    completed = []
    module.drain_batch(backend, uid_items, tokenizer, lambda item, row: completed.append(row))
    assert [row["id"] for row in completed] == ["2", "0", "1"]  # early EOS checkpoint before peers
    stream = local_stream_function()
    for row in completed:
        scalar = list(stream(row["generated_token_ids"], Tokenizer(), [8, 9, 10], max_tokens=uid_items[int(row["id"])]["request"]["max_tokens"]))
        assert row["text"] == "".join(part.text for part in scalar)
        assert row["generation_tokens"] == scalar[-1].generation_tokens
        assert row["prompt_tokens"] == scalar[-1].prompt_tokens == 3
        assert row["finish_reason"] == scalar[-1].finish_reason
        assert row["text_generation_tokens"] + row["terminal_eos_count"] == row["generation_tokens"]
    eos_first = completed[0]
    assert eos_first["raw_text"] == "" and eos_first["generation_tokens"] == eos_first["terminal_eos_count"] == 1
    first_two=next(row for row in completed if row['id']=='0')
    assert first_two['selected_logprobs']==[-1.,-.2]
    assert first_two['logprob_mean']==pytest.approx(-.6)
    assert first_two['logprob_min']==-1. and first_two['low_confidence_fraction']==.5
    assert first_two['logprob_q10']==pytest.approx(-.92)
    assert eos_first['logprob_mean']==pytest.approx(-.2)


def test_prompt_encoding_matches_stream_bos_rule_and_share_explicit_ids():
    tokenizer = Tokenizer()
    module.prepare_prompt(request(), tokenizer, "stream_string", 50)
    assert tokenizer.encodings[-1] == {"add_special_tokens": False}
    tokenizer.rendered = "no BOS"
    module.prepare_prompt(request(), tokenizer, "stream_string", 50)
    assert tokenizer.encodings[-1] == {"add_special_tokens": True}
    module.prepare_prompt(request(), tokenizer, "explicit_no_special_tokens", 50)
    assert tokenizer.encodings[-1] == {"add_special_tokens": False}
    with pytest.raises(ValueError, match="overflow"):
        module.prepare_prompt(request(cap=8), tokenizer, "stream_string", 10)


def test_stable_batch_four_grouping_and_public_request_contract():
    rows = [request(str(index), model="m/a" if index % 2 == 0 else "m/b") for index in range(11)]
    batches = module.batch_plan(module.validate_requests(rows))
    assert [batch["planned_members"] for batch in batches] == [["0", "2", "4", "6"], ["8", "10"], ["1", "3", "5", "7"], ["9"]]
    assert batches == module.batch_plan(rows)
    with pytest.raises(ValueError, match="exactly"):
        module.validate_requests([dict(request(), correct=True)])
    with pytest.raises(ValueError, match="duplicate"):
        module.validate_requests([request(), request()])
    with pytest.raises(ValueError, match="between 1 and 4"):
        module.batch_plan(rows, 5)


def test_completed_item_checkpoint_survives_later_batch_failure(tmp_path):
    requests = [request("a"), request("b")]
    tokenizer = Tokenizer(); progress = {}
    items = {uid: module.prepare_prompt(row, tokenizer, "stream_string", 50) for uid, row in enumerate(requests)}
    backend = Backend([( [], [response(0, 0, "stop"), response(1, 1)]), RuntimeError("synthetic interruption")])
    def save(item, row):
        row.update(request_sha256=module.digest(item["request"]), manifest_sha256="fixture")
        module.atomic_write(module.item_path(tmp_path, row["id"]), json.dumps(row))
    with pytest.raises(RuntimeError, match="interruption"):
        module.drain_batch(backend, items, tokenizer, save, progress)
    completed = module.load_items(tmp_path, requests, "fixture")
    assert set(completed) == {"a"} and completed["a"]["generation_tokens"] == 1
    assert progress[1]["tokens"] == [1] and not progress[1]["completed"]
    assert not module.item_path(tmp_path, "b").exists()
    with pytest.raises(ValueError, match="Stale"):
        module.load_items(tmp_path, requests, "different manifest")


def test_unexpected_stop_or_early_length_never_emits_completed_item():
    tokenizer = Tokenizer(); item = module.prepare_prompt(request(), tokenizer, "stream_string", 50)
    for bad in (response(0, 1, "stop"), response(0, 1, "length")):
        completed = []
        with pytest.raises(ValueError, match="EOS/length"):
            module.drain_batch(Backend([([], [bad])]), {0: item}, tokenizer, lambda *args: completed.append(args))
        assert not completed


def test_prepare_only_uses_files_and_frozen_backend_metadata(tmp_path, monkeypatch):
    source = tmp_path / "requests.jsonl"
    source.write_text(json.dumps(request()) + "\n")
    called = []
    monkeypatch.setattr(module, "model_metadata", lambda *args: called.append("model metadata") or {"local_path": "fixture"})
    monkeypatch.setattr(module, "backend_metadata", lambda: {"mlx_lm_version": "0.31.3", "code_sha256": "fixture"})
    result = module.prepare(source, tmp_path / "prepared")
    assert result["models_loaded"] is False and called == ["model metadata"]
    manifest = json.loads((tmp_path / "prepared/manifest.json").read_text())
    assert manifest["batch_size"] == 4 and manifest["temperature"] == 0
    assert not (tmp_path / "prepared/items").exists()
