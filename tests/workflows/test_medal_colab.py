"""Execute real Colab cells on CPU with a synthetic model and mocked setup IO."""

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nbformat
import torch

from dlm_project.models import cache
from dlm_project.tasks.gsm8k import ab_development as dev, local_experiments as runner


class SyntheticModel:
    device = torch.device('cpu')
    config = SimpleNamespace(vocab_size=6, eos_token_id=4)

    def __init__(self):
        self.calls = 0
        self.fail_at = None

    def __call__(self, ids, **kwargs):
        assert not torch.is_grad_enabled()
        self.calls += 1
        if self.calls == self.fail_at:
            raise KeyboardInterrupt('synthetic runtime disconnect')
        # Exactly one confident mask: temporary rollouts have work left to do.
        logits = torch.tensor([0., .4, .35, .25, 0., 0.]).log().expand(*ids.shape, 6).clone()
        for row in range(len(ids)):
            positions = torch.nonzero(ids[row] == 0).flatten()
            if len(positions):
                logits[row, positions[0]] = torch.tensor([0., .85, .1, .05, 0., 0.]).log()
        return SimpleNamespace(logits=logits)


class SyntheticTokenizer:
    eos_token_id = pad_token_id = 4

    def get_vocab(self):
        return {'[MASK]': 0, 'one': 1, 'two': 2, 'three': 3, '<|endoftext|>': 4, '<|eot_id|>': 5}

    def convert_ids_to_tokens(self, token_id):
        return next(k for k, v in self.get_vocab().items() if v == token_id)

    def decode(self, ids, skip_special_tokens=False):
        return ' '.join(self.convert_ids_to_tokens(i) for i in ids if not skip_special_tokens or i not in (0, 4, 5))


class ColabComparisonTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(TemporaryDirectory()))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.book = nbformat.read('notebooks/gsm8k/MEDAL_MCTS_Demo.ipynb', as_version=4)
        nbformat.validate(self.book)
        for cell in self.book.cells:
            if cell.cell_type == 'code':
                compile(cell.source, cell.id, 'exec')
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)
        self.model = SyntheticModel()
        self.wrapper = SimpleNamespace(model=self.model, tokenizer=SyntheticTokenizer(), mask_id=0,
            identity={k: dev.load_shared_protocol()['model'][k] for k in ('family', 'repo_id', 'revision')},
            encode_prompt=lambda text: torch.tensor([[3] * (2 + len(text) % 4)]))
        self.loader = self.stack.enter_context(patch.object(dev, 'load_model', return_value=self.wrapper))
        self.cache = self.stack.enter_context(patch.object(cache, 'cache_model', return_value=self.root/'model'))
        for name, value in [('is_available', True), ('is_bf16_supported', True), ('get_device_name', 'synthetic')]:
            self.stack.enter_context(patch.object(torch.cuda, name, return_value=value))
        self.scope = dict(Path=Path, DRIVE_ROOT=self.root, REPO_DIR=Path.cwd(), IN_COLAB=False)

    def execute(self, mode='comparison', until=None):
        active = False
        for cell in self.book.cells:
            if cell.cell_type != 'code':
                continue
            active |= cell.id == 'ab-settings'
            if active:
                source = cell.source.replace('NOTEBOOK_MODE = "comparison"', f'NOTEBOOK_MODE = "{mode}"')
                exec(source, self.scope)
                if cell.id == 'ab-settings':
                    self.scope['display'] = lambda *args: None
                if cell.id == until:
                    break

    def test_default_run_all_generates_paired_reports_and_resume_skips_completed(self):
        self.execute()
        self.loader.assert_called_once()
        self.cache.assert_called_once()
        self.assertTrue(self.loader.call_args.kwargs['local_files_only'])
        directory = self.scope['COMPARISON_OUTPUT']
        cfg = self.scope['COMPARISON_CONFIG']
        self.assertEqual((cfg['stage'], cfg['num_samples'], cfg['search']['init_tokens'],
                          cfg['search']['max_model_calls']), ('test', 5, 10, 256))
        rows = runner.read_predictions(directory/'predictions.jsonl')
        self.assertEqual(len(rows), 10)
        for baseline, search in zip(rows[:5], rows[5:]):
            self.assertEqual(baseline['condition'], 'confidence_t0')
            self.assertEqual(search['reward_mode'], 'adalook_adaptive')
            for key in ('example_id', 'prompt_ids_hash', 'seed'):
                self.assertEqual(baseline[key], search[key])
            self.assertGreater(search['adalook_extra_nfes'], 0)
            self.assertLessEqual(search['search_nfes'], 256)
        self.assertEqual(sum(r['total_nfes'] for r in rows), self.model.calls - 1)  # Separate warm-up.
        self.assertIsNone(self.scope['SINGLE_OUTPUT'])
        self.assertNotIn('initialization', self.scope)  # No extra demo inference in Run all.
        self.assertIsNone(self.scope['serial_development'])
        self.assertIsNone(self.scope['batch_study'])
        for name in ('config.json', 'selection.json', 'manifest.json', 'summary.json',
                     'report/report.md', 'report/summary.csv', 'report/comparison.png'):
            self.assertTrue((directory/name).is_file(), name)
        self.assertEqual(json.loads((directory/'summary.json').read_text())['status'], 'completed')
        before = (directory/'predictions.jsonl').read_bytes()
        calls = self.model.calls
        self.scope['RESUME_RUN'] = directory
        exec(next(c.source for c in self.book.cells if c.id == 'medal-comparison'), self.scope)
        self.assertEqual(self.model.calls, calls)
        self.assertEqual((directory/'predictions.jsonl').read_bytes(), before)

    def test_demo_skips_comparison_and_audits_real_rollout_nfes(self):
        self.execute(mode='demo')
        self.loader.assert_called_once()
        self.assertIsNone(self.scope['COMPARISON_OUTPUT'])
        record = self.scope['record']
        self.assertGreater(record['adalook_extra_nfes'], 0)
        self.assertEqual(record['total_nfes'], self.model.calls - 1)
        self.assertTrue((self.scope['SINGLE_OUTPUT']/'medal_tree.png').is_file())
        self.assertFalse((self.root/'results/LLaDA').exists())

    def test_interruption_resume_retries_only_unfinished_pair(self):
        self.execute(until='ab-load-model')
        cell = next(c.source for c in self.book.cells if c.id == 'medal-comparison')
        # One baseline completes, then disconnect during the second question.
        self.model.fail_at = self.model.calls + 256 + 3
        with self.assertRaises(KeyboardInterrupt):
            exec(cell, self.scope)
        directory = next((self.root/'results').rglob('config.json')).parent
        rows = runner.read_predictions(directory/'predictions.jsonl')
        self.assertEqual([r['status'] for r in rows], ['completed', 'interrupted'])
        self.assertEqual(json.loads((directory/'summary.json').read_text())['status'], 'interrupted')
        self.scope['RESUME_RUN'] = directory
        exec(cell, self.scope)
        resumed = runner.read_predictions(directory/'predictions.jsonl')
        self.assertEqual(len(resumed), 11)
        self.assertEqual(sum(r['status'] == 'completed' for r in resumed), 10)
        self.assertEqual(resumed[0], rows[0])
        self.assertEqual(len(list((directory/'traces').glob('*.jsonl'))), 11)


if __name__ == '__main__':
    unittest.main()
