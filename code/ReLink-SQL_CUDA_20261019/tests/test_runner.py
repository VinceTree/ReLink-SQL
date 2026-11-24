"""Synthetic SQLite integration tests, never benchmark/model inference."""
import json
import hashlib
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cuda_repro.backend import CudaCallBank, QUANTIZATION, REPOS
from cuda_repro.common import read, sha, verify_upstream
from cuda_repro import runner


class ScriptedTokenizer:
    bos_token = None
    bos_token_id = None
    eos_token_id = 99
    pad_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)

    def encode(self, prompt, **kwargs):
        return [1, 2, 3]

    def decode(self, tokens, **kwargs):
        return {10: 'SELECT 0', 11: 'SELECT 101', 12: 'SELECT 102',
                13: 'SELECT 103', 14: 'Count all people.',
                15: 'SELECT COUNT(*) FROM people', 16: 'SELECT COUNT(*) FROM people'}[tokens[0]]


class ScriptedBank(CudaCallBank):
    initial_success = False
    fail_first_repair = False

    def _load_model(self, key):
        return SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[99])), ScriptedTokenizer()

    def _invoke_one(self, task_id, request):
        self.request = request
        return super()._invoke_one(task_id, request)

    def _generate(self, model, tokenizer, prompt_ids, cap, observer, eos_ids):
        request_id = self.request['id']
        if 'INITIAL_R0_' in request_id:
            token = 16 if self.initial_success else 10
        else:
            stage = next(n for n in range(1, 6) if f'_R{n}_' in request_id)
            if stage == 1 and self.fail_first_repair:
                raise RuntimeError('Injected first repair interruption')
            token = 10 + stage
        observer.put([prompt_ids])
        observer.put([token])
        observer.put([99])
        observer.end()
        return prompt_ids + [token, 99]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / 'run'
        config = dict(models={}, device=0, batch_size=1, max_context_tokens=24000,
                      quantization=dict(QUANTIZATION))
        for i, (key, repo) in enumerate(REPOS.items(), 1):
            revision = str(i) * 40
            path = self.root / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
            path.mkdir(parents=True)
            (path / 'config.json').write_text('{"model_type":"qwen2"}')
            for name in ('tokenizer.json', 'tokenizer_config.json', 'generation_config.json'):
                (path / name).write_text('{}')
            (path / 'model.safetensors').write_bytes(b'not real weights; fixture only')
            config['models'][key] = dict(repo=repo, revision=revision, path=str(path))
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps(config))
        self.databases = self.root / 'database'
        db = self.databases / 'fixture/fixture.sqlite'
        db.parent.mkdir(parents=True)
        with sqlite3.connect(db) as con:
            con.execute('CREATE TABLE people(id INTEGER PRIMARY KEY, name TEXT)')
            con.executemany('INSERT INTO people VALUES (?,?)', [(1, 'A'), (2, 'B')])
        self.dev = self.root / 'dev.json'
        self.rows = [dict(db_id='fixture', question=f'How many people are there? Case {i}',
                          query='SELECT COUNT(*) FROM people') for i in range(1034)]
        self.dev.write_text(json.dumps(self.rows))

    def prepare(self, size=10):
        return runner.prepare(self.config, self.dev, self.databases, self.output, size)

    def run_full(self, bank=ScriptedBank):
        # The branch/no-proposal fixture exercises the paid planning + r5 path.
        # The unchanged program-repair algorithm has its own upstream tests.
        with patch.object(runner.engine.multiplicity_safe, 'propose', return_value=(None, {'fixture': True})):
            return runner.execute_run(self.output, bank_factory=bank, check_gpu=False)

    def test_upstream_is_unchanged(self):
        self.assertEqual(verify_upstream(), 84)

    def test_bridge_exposes_only_original_cpu_definitions_without_mlx(self):
        from iteration2 import collect
        self.assertEqual(collect.__all__, ['MODELS', 'seed'])
        expected = int(hashlib.sha256(b'taskactioniteration2').hexdigest()[:8], 16)
        self.assertEqual(collect.seed('task', 'action'), expected)
        self.assertFalse(hasattr(collect, 'main'))
        self.assertNotIn('mlx', sys.modules)
        self.assertNotIn('mlx_lm', sys.modules)

    def test_selection_is_frozen_before_gold_and_is_reproducible(self):
        marker = self.prepare()
        selected = read(self.output / 'selection.json')
        self.assertEqual([x['source_row'] for x in selected['rows']], runner.select_rows(self.rows, 10))
        self.assertEqual(len(read(self.output / 'public.jsonl')), 10)
        self.assertEqual(len(read(self.output / 'private/references.json')), 10)
        self.assertTrue(selected['no_outcome_selection'])
        self.assertEqual(marker, self.prepare())
        public = (self.output / 'public.jsonl').read_text()
        self.assertNotIn('gold_sql', public)
        self.assertNotIn('SELECT COUNT(*) FROM people', public)

    def test_full_five_stage_flow_and_readonly_audit(self):
        self.prepare()
        summary = self.run_full()
        self.assertEqual(summary['final_round'], 5)
        self.assertEqual(summary['initial_correct'], 0)
        self.assertEqual(summary['final_correct'], 10)
        self.assertEqual(summary['path_tokens'], 300)
        for row in read(self.output / 'CURRENT/r4/paths.jsonl'):
            self.assertTrue(row['carried'])
            self.assertEqual(row['total_tokens'], 25)
        before = {str(p): sha(p) for p in self.output.rglob('*') if p.is_file()}
        audit = runner.audit(self.output)
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['completed_calls'], 60)
        self.assertEqual(audit['completed_call_tokens'], 300)
        self.assertEqual(before, {str(p): sha(p) for p in self.output.rglob('*') if p.is_file()})

    def test_initial_success_stops_without_repairs(self):
        class InitialSuccess(ScriptedBank):
            initial_success = True
        self.prepare()
        summary = self.run_full(InitialSuccess)
        self.assertEqual(summary['final_round'], 0)
        self.assertEqual(summary['final_correct'], 10)
        self.assertEqual(runner.audit(self.output)['completed_calls'], 10)
        self.assertFalse((self.output / 'CURRENT/r1').exists())

    def test_interruption_resume_does_not_reexecute_initial_sql(self):
        class Interrupt(ScriptedBank):
            fail_first_repair = True
        self.prepare()
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.run_full(Interrupt)
        initial_before = sha(self.output / 'initial/observations.jsonl')
        execute = runner.r.execute
        def no_initial_reexecution(db, sql, **kwargs):
            if sql == 'SELECT 0':
                raise AssertionError('Saved initial SQL must not be reexecuted')
            return execute(db, sql, **kwargs)
        with patch.object(runner.r, 'execute', side_effect=no_initial_reexecution):
            summary = self.run_full()
        self.assertEqual(summary['final_correct'], 10)
        self.assertEqual(initial_before, sha(self.output / 'initial/observations.jsonl'))
        events = read(self.output / 'ledger/events.jsonl')
        self.assertEqual(sum(x['event'] == 'attempt_failed' for x in events), 1)
        audit = runner.audit(self.output)
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['failed_attempts'], 1)
        self.assertEqual(audit['failed_generation_tokens_observed'], 0)

    def test_input_changes_and_sample_size_changes_reject_resume(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'different input'):
            self.prepare(11)
        self.dev.write_text(self.dev.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'Changed external input'):
            runner.verify(self.output)

    def test_intermediate_stage_mutation_rejected(self):
        self.prepare()
        self.run_full()
        path = self.output / 'CURRENT/r2/local_contexts.json'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'Changed completed stage'):
            runner.audit(self.output)

    def test_unscorable_reference_is_retained_not_replaced(self):
        indices = runner.select_rows(self.rows, 10)
        self.rows[indices[0]]['query'] = 'SELECT missing_column FROM people'
        self.dev.write_text(json.dumps(self.rows))
        marker = self.prepare()
        self.assertEqual(len(marker['unscorable']), 1)
        summary = self.run_full()
        self.assertEqual(summary['selected_n'], 10)
        self.assertEqual(summary['scorable_n'], 9)
        self.assertEqual(summary['all_initial_tokens'], 50)
        self.assertEqual(summary['final_correct'], 9)

    def test_non_dev_and_database_writes_are_rejected(self):
        self.dev.write_text(json.dumps(self.rows[:100]))
        with self.assertRaisesRegex(ValueError, '1034 rows'):
            self.prepare()
        db = self.databases / 'fixture/fixture.sqlite'
        result = runner.r.execute(str(db), 'DELETE FROM people', timeout=10, max_rows=50000)
        self.assertEqual(result['status'], 'error')
        with sqlite3.connect(db) as con:
            self.assertEqual(con.execute('SELECT COUNT(*) FROM people').fetchone()[0], 2)


if __name__ == '__main__':
    unittest.main()
