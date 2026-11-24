"""Tests for BIRD rejected-result-filter ablation decisions and accounting."""
import importlib.util
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "bird_without_result_filter", Path(__file__).with_name("bird_without_result_filter.py"))
variant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(variant)


class AblationTests(unittest.TestCase):
    def test_complete_known_rejected_result_is_delivered(self):
        class Repair:
            @staticmethod
            def result_key(observation):
                return observation.get("key") if observation.get("status") == "ok" else None

        class Contract:
            repair = Repair
            r = SimpleNamespace(stable=lambda value: value)

            @staticmethod
            def parse(_text, _task):
                return {"plan": True}

            @staticmethod
            def align(_task, _history, _plan):
                return ({"index": 0}, [])

        local = SimpleNamespace(propose=lambda *_args: (
            "SELECT repeated", {"operations": ["toggle"], "probes": []}))
        multiplicity = SimpleNamespace(local=local)
        multiplicity_safe = SimpleNamespace(count_alternative=lambda *_args: self.fail(
            "the first complete candidate must stop candidate search"))
        history = [dict(sql="SELECT original", observation={"status": "ok", "key": "same"})]
        candidate, metadata = variant.propose_without_filter(
            {"db_path": "fixture"}, history, "plan", Contract, multiplicity,
            multiplicity_safe,
            lambda _db, _sql: {"status": "ok", "key": "same", "rows": [[1]], "columns": ["x"]},
        )
        self.assertEqual(candidate, "SELECT repeated")
        self.assertEqual(metadata["selected_status"], "known_rejected_result")
        self.assertTrue(variant.deliver_without_result_filter("same"))

    def test_incomplete_execution_is_not_delivered(self):
        self.assertFalse(variant.deliver_without_result_filter(None))

    def test_jsonl_reader_and_paired_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            expected = [{"id": "a"}, {"id": "b"}]
            path.write_text("".join(json.dumps(row) + "\n" for row in expected), encoding="utf-8")
            self.assertEqual(variant.read(path), expected)

        result = variant.paired_comparison(
            [dict(task_id="a", accepted=True, total_tokens=10),
             dict(task_id="b", accepted=False, total_tokens=20)],
            [dict(task_id="a", accepted=False, total_tokens=14),
             dict(task_id="b", accepted=True, total_tokens=19)],
        )
        self.assertEqual(result["full_only_successes"], 1)
        self.assertEqual(result["ablation_only_successes"], 1)
        self.assertEqual(result["path_token_delta_variant_minus_baseline"], 3)
        with self.assertRaisesRegex(ValueError, "task sets differ"):
            variant.paired_comparison(
                [dict(task_id="a", accepted=True, total_tokens=1)],
                [dict(task_id="b", accepted=True, total_tokens=1)],
            )

    def test_completion_record_commits_variant_and_final_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rel in ("variant.json", "final_paths.jsonl", "summary.json",
                        "CURRENT/initial_paths.jsonl", "ledger/events.jsonl"):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(rel, encoding="utf-8")

            def sha(path):
                return hashlib.sha256(Path(path).read_bytes()).hexdigest()

            record = variant.completion_record(root, "fingerprint", {"paired": True}, sha)
            self.assertEqual(record["variant_sha256"], sha(root / "variant.json"))
            self.assertEqual(record["paired_comparison"], {"paired": True})


if __name__ == "__main__":
    unittest.main()
