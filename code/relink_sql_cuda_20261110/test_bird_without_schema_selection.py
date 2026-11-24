"""Contract tests for the BIRD full-schema r1 ablation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "bird_without_schema_selection", Path(__file__).with_name("bird_without_schema_selection.py"))
variant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(variant)


class SchemaAblationTests(unittest.TestCase):
    def test_preflight_accepts_prompts_within_context_budget(self):
        result = variant.preflight_summary([
            dict(task_id="a", prompt_tokens=100, max_new_tokens=1024),
            dict(task_id="b", prompt_tokens=200, max_new_tokens=1024),
        ], 2048)
        self.assertEqual(result["pending_r1"], 2)
        self.assertEqual(result["overflow_n"], 0)
        self.assertEqual(result["max_prompt_tokens"], 200)

    def test_preflight_lists_overflow_without_truncation(self):
        result = variant.preflight_summary([
            dict(task_id="fits", prompt_tokens=900, max_new_tokens=100),
            dict(task_id="overflow", prompt_tokens=1900, max_new_tokens=200),
        ], 2048)
        self.assertEqual(result["overflow_n"], 1)
        self.assertEqual(result["overflow_task_ids"], ["overflow"])

    def test_full_schema_round_routes_r1_to_fresh14_and_preserves_transition(self):
        observed = []

        def build(task, state, context, stage, auxiliary):
            observed.append((state["method"], context["schema_text"], stage))
            return [], {}

        def clone(_fn, **replacement):
            def execute(_dest, round_number, previous, tasks, full, _tables, _bank, _evaluator, _gate):
                self.assertEqual(round_number, 1)
                self.assertEqual(replacement["methods"].initial_stage("CURRENT"), "answer")
                replacement["build"](tasks["t"], {"method": "CURRENT"}, full["t"], "answer")
                return previous
            return execute

        methods = SimpleNamespace(selectors=SimpleNamespace(clone=clone),
                                  initial_stage=lambda _name: "feedback")
        engine = SimpleNamespace(
            r=SimpleNamespace(methods=methods, build=build, round_run=object()),
            GATE="official",
            validate_transition=lambda before, after, _bank: self.assertEqual(before, after),
        )
        paths = [dict(task_id="t", accepted=False)]
        result = variant.full_schema_round(
            engine, "dest", paths, {"t": {"id": "t"}},
            {"t": {"schema_text": "all tables"}}, object(), object())
        self.assertIs(result, paths)
        self.assertEqual(observed, [("FRESH14", "all tables", "answer")])

    def test_paired_comparison_rejects_different_cohorts(self):
        with self.assertRaisesRegex(ValueError, "task sets differ"):
            variant.paired_comparison(
                [dict(task_id="a", accepted=True, total_tokens=1)],
                [dict(task_id="b", accepted=True, total_tokens=1)],
            )

    def test_shared_prefix_excludes_intervened_r1_artifacts(self):
        self.assertEqual(variant.shared_prefix_files(), [
            "initial/observations.jsonl", "CURRENT/initial_paths.jsonl"])


if __name__ == "__main__":
    unittest.main()
