"""CPU-only contract tests: no torch, Transformers, weights or GPU required."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cuda_repro.backend import CudaCallBank, QUANTIZATION, REPOS, _TokenObserver, validate_config


class FakeTokenizer:
    bos_token = None
    bos_token_id = None
    eos_token_id = 99
    pad_token_id = 98

    def __init__(self):
        self.special_flags = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "CHAT:" + messages[-1]["content"]

    def encode(self, prompt, add_special_tokens):
        self.special_flags.append(add_special_tokens)
        return [1, 2, 3]

    def decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False and clean_up_tokenization_spaces is False
        return " ".join(str(token) for token in ids)


class FakeBank(CudaCallBank):
    def __init__(self, *args):
        self.loads = []
        self.generate_calls = 0
        self.output = [10, 11, 99]
        self.fail_after = None
        self.return_mismatch = False
        super().__init__(*args)

    def _load_model(self, key):
        self.loads.append(key)
        return SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[98, 99])), FakeTokenizer()

    def _generate(self, model, tokenizer, prompt_ids, cap, observer, eos_ids):
        self.generate_calls += 1
        observer.put([prompt_ids])
        for i, token in enumerate(self.output):
            observer.put([token])
            if self.fail_after == i + 1:
                raise RuntimeError("Injected generation failure")
        observer.end()
        return prompt_ids + self.output + ([7] if self.return_mismatch else [])


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.config = dict(models={}, device=0, batch_size=1, max_context_tokens=24000,
                           quantization=dict(QUANTIZATION), run_fingerprint="fixture-run")
        for i, (key, repo) in enumerate(REPOS.items(), start=1):
            revision = str(i) * 40
            path = self.root / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
            path.mkdir(parents=True)
            (path / "config.json").write_text(json.dumps(dict(model_type="qwen2")))
            for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
                (path / name).write_text("{}")
            (path / "model.safetensors").write_bytes(b"fake weight fixture")
            self.config["models"][key] = dict(repo=repo, revision=revision, path=str(path))

    def spec(self, ident="task|answer", key="strong14", content="SELECT people", cap=1024):
        model = self.config["models"][key]
        return dict(task_id="task", method="CURRENT", round=1, stage="answer", metadata={},
                    request=dict(id=ident, messages=[dict(role="user", content=content)],
                                 model=model["repo"], revision=model["revision"], max_tokens=cap))

    def events(self):
        return [json.loads(line) for line in (self.run / "ledger/events.jsonl").read_text().splitlines()]

    def test_validate_and_constructor_are_cpu_only(self):
        metadata = validate_config(self.config)
        self.assertEqual(set(metadata), set(REPOS))
        bank = CudaCallBank(self.config, self.run)
        self.assertIsNone(bank._model)
        self.assertEqual(bank.raw, {})
        self.assertFalse((self.run / "ledger/events.jsonl").exists())

    def test_eos_is_charged_and_removed_only_from_text(self):
        bank = FakeBank(self.config, self.run)
        rows = bank.invoke(self.run / "stage", [self.spec()])
        raw = bank.raw[rows[0]["request_id"]]
        self.assertEqual(raw["prompt_token_ids"], [1, 2, 3])
        self.assertEqual(raw["generated_token_ids"], [10, 11, 99])
        self.assertEqual(raw["generation_tokens"], 3)
        self.assertEqual(raw["text_generation_tokens"], 2)
        self.assertEqual(raw["terminal_eos_count"], 1)
        self.assertEqual(raw["text"], "10 11")
        self.assertEqual(raw["finish_reason"], "stop")
        self.assertEqual(bank.tokens(rows[0]["request_ids"]), 6)
        self.assertEqual(bank._tokenizer.special_flags, [True])
        self.assertEqual([row["event"] for row in self.events()],
                         ["attempt_started", "generation_requested", "attempt_completed"])

    def test_same_stage_and_process_restart_never_regenerate(self):
        bank = FakeBank(self.config, self.run)
        expected = bank.invoke(self.run / "stage", [self.spec()])
        self.assertEqual(bank.invoke(self.run / "stage", [self.spec()]), expected)
        self.assertEqual(bank.generate_calls, 1)
        reopened = CudaCallBank(self.config, self.run)
        self.assertEqual(reopened.invoke(self.run / "stage", [self.spec()]), expected)
        self.assertEqual(reopened.tokens(expected[0]["request_ids"]), 6)
        self.assertIsNone(reopened._model)

    def test_bank_created_before_completion_refreshes_before_generation(self):
        first = FakeBank(self.config, self.run)
        second = FakeBank(self.config, self.run)
        expected = first.invoke(self.run / "stage", [self.spec()])
        self.assertEqual(second.invoke(self.run / "stage", [self.spec()]), expected)
        self.assertEqual(second.generate_calls, 0)

    def test_readonly_audit_cannot_create_or_modify_artifacts(self):
        bank = FakeBank(self.config, self.run)
        bank.invoke(self.run / "stage", [self.spec()])
        before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.run.rglob("*") if p.is_file()}
        readonly = CudaCallBank(self.config, self.run, readonly=True)
        self.assertEqual(readonly.tokens([self.spec()["request"]["id"]]), 6)
        with self.assertRaisesRegex(ValueError, "readonly"):
            readonly.invoke(self.run / "uncreated_stage", [])
        with self.assertRaisesRegex(ValueError, "Missing or changed"):
            CudaCallBank(self.config, self.root / "uncreated", readonly=True)
        self.assertFalse((self.root / "uncreated").exists())
        after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in self.run.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_same_length_checkpoint_token_or_text_tampering_is_rejected(self):
        for field, changed in (("generated_token_ids", [10, 12, 99]), ("raw_text", "changed SQL")):
            with self.subTest(field=field):
                run = self.root / field
                bank = FakeBank(self.config, run)
                bank.invoke(run / "stage", [self.spec()])
                path = bank._item_path(self.spec()["request"]["id"])
                row = json.loads(path.read_text())
                row[field] = changed
                path.write_text(json.dumps(row))
                with self.assertRaisesRegex(ValueError, "inconsistent"):
                    CudaCallBank(self.config, run, readonly=True)

    def test_real_generate_adapter_contract_with_fake_modules(self):
        class Tensor:
            def __init__(self, values):
                self.values = values

            def tolist(self):
                return self.values

        captured = {}

        def generate(**kwargs):
            captured.update(kwargs)
            kwargs["streamer"].put(kwargs["input_ids"])
            kwargs["streamer"].put(Tensor([10]))
            kwargs["streamer"].put(Tensor([99]))
            kwargs["streamer"].end()
            return Tensor([[1, 2, 3, 10, 99]])

        fake_torch = SimpleNamespace(long="long", tensor=lambda values, **kw: Tensor(values),
            ones_like=lambda tensor: Tensor([[1] * len(tensor.values[0])]), inference_mode=nullcontext,
            cuda=SimpleNamespace(synchronize=lambda device: captured.update(synchronized=device)))
        fake_transformers = SimpleNamespace(GenerationConfig=lambda **kwargs: SimpleNamespace(**kwargs))
        observer = _TokenObserver([1, 2, 3])
        bank = CudaCallBank(self.config, self.run)
        with patch.dict("sys.modules", torch=fake_torch, transformers=fake_transformers):
            result = bank._generate(SimpleNamespace(generate=generate), FakeTokenizer(), [1, 2, 3],
                                    1024, observer, [98, 99])
        self.assertEqual(result, [1, 2, 3, 10, 99])
        generation = captured["generation_config"]
        self.assertEqual(generation.max_new_tokens, 1024)
        self.assertFalse(generation.do_sample)
        self.assertFalse(generation.output_scores)
        self.assertFalse(generation.output_logits)
        self.assertEqual(generation.num_beams, 1)
        self.assertEqual(generation.repetition_penalty, 1.0)
        self.assertEqual(captured["synchronized"], 0)

    def test_identical_request_reuses_only_this_run_ledger(self):
        bank = FakeBank(self.config, self.run)
        first = bank.invoke(self.run / "stage1", [self.spec()])
        second = bank.invoke(self.run / "stage2", [self.spec(ident="task|repeat")])
        self.assertEqual(second[0]["request_id"], first[0]["request_id"])
        self.assertEqual(second[0]["logical_request_id"], "task|repeat")
        self.assertEqual(bank.generate_calls, 1)
        other = FakeBank(self.config, self.root / "other_run")
        other.invoke(self.root / "other_run/stage", [self.spec()])
        self.assertEqual(other.generate_calls, 1)

    def test_length_completion_requires_exact_cap(self):
        bank = FakeBank(self.config, self.run)
        bank.output = [10, 11]
        rows = bank.invoke(self.run / "stage", [self.spec(cap=2)])
        raw = bank.raw[rows[0]["request_id"]]
        self.assertEqual((raw["finish_reason"], raw["terminal_eos_count"]), ("length", 0))
        self.assertEqual(raw["text"], "10 11")

    def test_overflow_fails_without_truncating_or_generating(self):
        self.config["max_context_tokens"] = 6
        bank = FakeBank(self.config, self.run)
        with self.assertRaisesRegex(ValueError, "no truncation"):
            bank.invoke(self.run / "stage", [self.spec(cap=4)])
        self.assertEqual(bank.generate_calls, 0)
        failed = self.events()[-1]
        self.assertEqual(failed["prompt_tokens_requested"], 3)
        self.assertIsNone(failed["prompt_tokens_processed_observed"])
        self.assertEqual(bank.raw, {})

    def test_failed_partial_is_preserved_and_retry_is_new_attempt(self):
        bank = FakeBank(self.config, self.run)
        bank.fail_after = 2
        with self.assertRaisesRegex(RuntimeError, "Injected"):
            bank.invoke(self.run / "stage", [self.spec()])
        failed = self.events()[-1]
        self.assertEqual(failed["event"], "attempt_failed")
        self.assertEqual(failed["generated_token_ids_observed"], [10, 11])
        self.assertIn("lower bound", failed["usage_note"])
        self.assertIsNone(failed["prompt_tokens_processed_observed"])
        self.assertEqual(bank.raw, {})
        retry = FakeBank(self.config, self.run)
        retry.invoke(self.run / "stage", [self.spec()])
        starts = [row for row in self.events() if row["event"] == "attempt_started"]
        self.assertEqual(len(starts), 2)
        self.assertNotEqual(starts[0]["attempt_id"], starts[1]["attempt_id"])

    def test_model_switch_retains_only_current_model(self):
        bank = FakeBank(self.config, self.run)
        bank.invoke(self.run / "initial", [self.spec("task|initial", key="strong")])
        bank.invoke(self.run / "repair", [self.spec("task|repair", key="strong14")])
        self.assertEqual(bank.loads, ["strong", "strong14"])
        self.assertEqual(bank._current_key, "strong14")

    def test_changed_config_refuses_resume(self):
        FakeBank(self.config, self.run)
        changed = copy.deepcopy(self.config)
        changed["run_fingerprint"] = "new-config"
        with self.assertRaisesRegex(ValueError, "changed committed"):
            CudaCallBank(changed, self.run)

    def test_unknown_model_revision_and_share_fail_before_inference(self):
        bank = FakeBank(self.config, self.run)
        wrong = self.spec()
        wrong["request"]["revision"] = "f" * 40
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            bank.invoke(self.run / "stage", [wrong])
        with self.assertRaisesRegex(ValueError, "SHARE"):
            bank.invoke(self.run / "share", [self.spec()], stopped=True)
        self.assertEqual(bank.generate_calls, 0)

    def test_missing_shard_and_unpinned_path_are_rejected(self):
        info = self.config["models"]["strong"]
        path = Path(info["path"])
        (path / "model.safetensors.index.json").write_text(json.dumps(
            dict(weight_map={"one": "missing-00001.safetensors"})))
        with self.assertRaisesRegex(ValueError, "Missing safetensors shard"):
            validate_config(self.config)
        (path / "model.safetensors.index.json").unlink()
        info["path"] = str(path.parent)
        with self.assertRaisesRegex(ValueError, "snapshots/revision"):
            validate_config(self.config)

    def test_changed_request_or_stage_inputs_cannot_overwrite(self):
        bank = FakeBank(self.config, self.run)
        bank.invoke(self.run / "stage", [self.spec()])
        with self.assertRaisesRegex(ValueError, "changed committed"):
            bank.invoke(self.run / "stage", [self.spec(content="different SQL")])
        with self.assertRaisesRegex(ValueError, "changed inputs"):
            bank.invoke(self.run / "other", [self.spec(content="different SQL")])

    def test_early_stop_or_stream_mismatch_never_commits(self):
        for name, output, mismatch in (("early", [10], False), ("eos_middle", [99, 10], False),
                                       ("mismatch", [10, 99], True)):
            with self.subTest(name=name):
                run = self.root / name
                bank = FakeBank(self.config, run)
                bank.output, bank.return_mismatch = output, mismatch
                with self.assertRaises(ValueError):
                    bank.invoke(run / "stage", [self.spec()])
                self.assertEqual(bank.raw, {})

    def test_changed_model_asset_is_rejected(self):
        bank = FakeBank(self.config, self.run)
        path = Path(self.config["models"]["strong14"]["path"])
        (path / "model.safetensors").write_bytes(b"changed size and contents")
        with self.assertRaisesRegex(ValueError, "assets changed"):
            bank.invoke(self.run / "stage", [self.spec()])
        self.assertEqual(bank.generate_calls, 0)

    def test_key_is_stable_and_stage_cannot_escape_run_directory(self):
        bank = FakeBank(self.config, self.run)
        request = self.spec()["request"]
        key = bank.key("task", request)
        self.assertEqual(key, CudaCallBank(self.config, self.run).key("task", request))
        self.assertNotEqual(key, bank.key("other_task", request))
        with self.assertRaisesRegex(ValueError, "inside this run"):
            bank.invoke(self.root / "outside", [self.spec()])


if __name__ == "__main__":
    unittest.main()
