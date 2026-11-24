"""Synthetic BIRD contract tests; no GPU or real benchmark data is used."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from cuda_repro import runner
from cuda_repro.common import read, sha

sys.path.insert(0, str(runner.ROOT / 'tests'))
import test_runner as fixtures

spec = importlib.util.spec_from_file_location('bird_cuda', Path(__file__).with_name('bird_cuda.py'))
bird = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bird)


class BirdTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RunnerTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.rows = [dict(db_id='fixture', question=f'Question {i}', evidence='',
                          SQL='SELECT COUNT(*) FROM people') for i in range(6)]
        self.rows[0]['SQL'] = 'SELECT missing_column FROM people'
        f.dev.write_text(json.dumps(self.rows))
        self.db = f.databases / 'fixture/fixture.sqlite'
        self.expected = {'fixture': sha(self.db)}

    def prepare(self, size=6):
        f = self.fixture
        return bird.prepare(f.config, f.dev, f.databases, f.output, size,
                            expected_question=sha(f.dev), expected_databases=self.expected,
                            expected_rows=6)

    def run_flow(self, bank=fixtures.ScriptedBank):
        with patch.object(runner.engine.multiplicity_safe, 'propose', return_value=(None, {'fixture': True})):
            return runner.execute_run(self.fixture.output, bank_factory=bank, check_gpu=False)

    def test_bird_full_flow_references_and_audit(self):
        f = self.fixture
        marker = self.prepare()
        self.assertEqual(len(marker['unscorable']), 1)
        tasks = read(f.output / 'public.jsonl')
        self.assertTrue(all(t['dataset'] == 'bird' and t['split'] == 'full_dev' for t in tasks))
        self.assertNotIn('SELECT COUNT(*) FROM people', (f.output / 'public.jsonl').read_text())
        self.assertNotIn('SELECT COUNT(*) FROM people', (f.output / 'full_contexts.json').read_text())
        evaluator = runner.r.Evaluator(f.output / 'private/references.json')
        task = tasks[1]
        # BIRD's pinned set matcher intentionally ignores duplicate result rows.
        observation = dict(status='ok', rows=[[2], [2]])
        self.assertTrue(evaluator.official(task, observation))
        summary = self.run_flow()
        self.assertEqual((summary['selected_n'], summary['scorable_n']), (6, 5))
        self.assertEqual(summary['final_correct'], 5)
        self.assertTrue(runner.audit(f.output)['passed'])
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.prepare()

    def test_interruption_resume_and_changed_input(self):
        class Interrupt(fixtures.ScriptedBank):
            fail_first_repair = True
        f = self.fixture
        self.prepare()
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.run_flow(Interrupt)
        before = sha(f.output / 'initial/observations.jsonl')
        summary = self.run_flow()
        self.assertEqual(summary['final_correct'], 5)
        self.assertEqual(before, sha(f.output / 'initial/observations.jsonl'))
        self.assertTrue(runner.audit(f.output)['passed'])
        f.dev.write_text(f.dev.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'Changed external input'):
            runner.verify(f.output)

    def test_wrong_database_hash_rejected_before_output(self):
        f = self.fixture
        with self.assertRaisesRegex(ValueError, 'Database hash mismatch'):
            bird.prepare(f.config, f.dev, f.databases, f.output, 6,
                         expected_question=sha(f.dev), expected_databases={'fixture': '0' * 64},
                         expected_rows=6)
        self.assertFalse(f.output.exists())


if __name__ == '__main__':
    unittest.main()
