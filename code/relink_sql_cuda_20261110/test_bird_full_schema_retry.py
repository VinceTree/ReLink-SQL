"""Synthetic integration and contract checks for BIRD full-schema retry."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

from cuda_repro import runner
from cuda_repro.common import read, sha

sys.path.insert(0, str(runner.ROOT / "tests"))
import test_runner as fixtures

spec = importlib.util.spec_from_file_location(
    "bird_full_schema_retry", Path(__file__).with_name("bird_full_schema_retry.py"))
retry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retry)


class RetryTests(unittest.TestCase):
    def test_retry_prompt_uses_full_schema_and_execution_report_method(self):
        calls = []

        def build(_task, state, context, stage, _auxiliary):
            calls.append((state["method"], context["schema_text"], stage))
            return [], {}

        def clone(_fn, **replacement):
            def invoke(_dest, stage, previous, tasks, full, _tables, _bank, _eval, _gate):
                self.assertEqual(stage, 3)
                replacement["build"](tasks["t"], {"method": "CURRENT"}, full["t"])
                return previous
            return invoke

        methods = SimpleNamespace(selectors=SimpleNamespace(clone=clone),
                                  initial_stage=lambda _method: "feedback")
        engine = SimpleNamespace(r=SimpleNamespace(build=build, methods=methods, round_run=object()),
                                 GATE="official", validate_transition=lambda *_args: None)
        retry.full_schema_retry_round(engine, "out", 3, [{"task_id": "t"}], {"t": {}},
                                      {"t": {"schema_text": "FULL SCHEMA"}}, object(), object())
        self.assertEqual(calls, [("FULL_SQL_REPORT", "FULL SCHEMA", "answer")])

    def test_synthetic_five_stage_flow_and_ledger_audit(self):
        f = fixtures.RunnerTests(methodName="runTest")
        f.setUp()
        self.addCleanup(f.doCleanups)
        marker = f.prepare(3)
        f.run_full()
        before = sha(f.output / "CURRENT/initial_paths.jsonl")
        out = f.root / "retry"
        retry.copy_without_repair_stages(f.output, out)
        runner.save(out / "variant.json", {"test": True})
        runner.verify(out)
        config = dict(marker["config"], run_fingerprint=marker["run_fingerprint"])
        runner.configure_models(config)
        bank = retry.VariantBank(fixtures.ScriptedBank(config, out))
        result = retry.execute_stages(runner, f.output, out, marker, bank)
        self.assertTrue(result["audit"]["passed"])
        self.assertEqual(result["summary"]["scorable_n"], 3)
        self.assertEqual(result["summary"]["final_correct"], 3)
        self.assertEqual(result["summary"]["paired_comparison"]["paired_scorable_n"], 3)
        self.assertEqual(before, sha(out / "CURRENT/initial_paths.jsonl"))
        self.assertTrue((out / "complete.json").is_file())
        self.assertTrue(runner.audit(out)["passed"])
        self.assertEqual(read(f.output / "summary.json")["scorable_n"], 3)


if __name__ == "__main__":
    unittest.main()
