"""Tests for BIRD no-program-repair delivery and request isolation."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "bird_skip_program_repair", Path(__file__).with_name("bird_skip_program_repair.py"))
variant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(variant)


class RecordingBank:
    def invoke(self, directory, specs, stopped=False):
        self.specs = specs
        return specs


class AblationContractTests(unittest.TestCase):
    def test_jsonl_reader_parses_complete_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            rows = [{"id": "one", "question": "first"},
                    {"id": "two", "question": "second"}]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            self.assertEqual(variant.read(path), rows)

    def test_r4_skips_candidate_delivery_but_accounts_requirement_call(self):
        previous = [
            dict(task_id="accepted", accepted=True, feedback="", call_ids=["a"],
                 total_tokens=5, input_tokens=3, output_tokens=2),
            dict(task_id="rejected", accepted=False, feedback="NO", call_ids=["b"],
                 total_tokens=7, input_tokens=4, output_tokens=3),
        ]
        plans = {"rejected": dict(request_ids=["plan"])}

        def account(row):
            row["total_tokens"] += 11
            row["input_tokens"] += 8
            row["output_tokens"] += 3
            return row

        result = variant.skip_program_deliveries(previous, plans, account)
        self.assertEqual(result[0]["call_ids"], ["a"])
        self.assertEqual(result[0]["total_tokens"], 5)
        self.assertTrue(result[0]["carried"])
        self.assertEqual(result[1]["call_ids"], ["b", "plan"])
        self.assertEqual(result[1]["total_tokens"], 18)
        self.assertNotIn("accepted", result[1])
        self.assertNotIn("feedback", result[1])

    def test_variant_requests_use_distinct_logical_ids(self):
        raw = RecordingBank()
        bank = variant.VariantBank(raw)
        specs = [dict(request=dict(id="bird:task:r4"))]
        records = bank.invoke("r4/planning", specs)
        self.assertEqual(specs[0]["request"]["id"], "bird:task:r4")
        self.assertEqual(records[0]["request"]["id"], "bird:task:r4" + variant.SUFFIX)

    def test_paired_comparison_counts_discordant_outcomes_and_tokens(self):
        baseline = [
            dict(task_id="a", accepted=True, total_tokens=10),
            dict(task_id="b", accepted=False, total_tokens=20),
            dict(task_id="c", accepted=True, total_tokens=30),
        ]
        ablation = [
            dict(task_id="a", accepted=False, total_tokens=12),
            dict(task_id="b", accepted=True, total_tokens=18),
            dict(task_id="c", accepted=True, total_tokens=31),
        ]
        result = variant.paired_comparison(baseline, ablation)
        self.assertEqual(result["full_only_successes"], 1)
        self.assertEqual(result["ablation_only_successes"], 1)
        self.assertEqual(result["net_correct_delta_variant_minus_baseline"], 0)
        self.assertEqual(result["path_token_delta_variant_minus_baseline"], 1)

    def test_completion_record_hashes_variant_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            files = {
                "variant.json": "variant",
                "final_paths.jsonl": "paths",
                "summary.json": "summary",
                "CURRENT/initial_paths.jsonl": "initial",
                "ledger/events.jsonl": "events",
            }
            for relative, content in files.items():
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

            def sha(path):
                return hashlib.sha256(Path(path).read_bytes()).hexdigest()

            record = variant.completion_record(output, "fingerprint", {"paired": True}, sha)
            self.assertEqual(record["variant_sha256"], sha(output / "variant.json"))
            self.assertEqual(record["final_paths_sha256"], sha(output / "final_paths.jsonl"))


if __name__ == "__main__":
    unittest.main()
