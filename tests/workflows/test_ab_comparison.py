"""AB notebook comparisons, durable failure records and common report fields."""

import json
from pathlib import Path

import nbformat
import pytest
import torch

from dlm_project.decoding.ab_medal import ABMEDALSearch
from dlm_project.tasks.gsm8k import ab_comparison as runner, ab_development as dev
from tests.workflows.test_ab_development import wrapper
from tests.decoding.test_ab_batch import BatchModel


def test_real_ab_variants_report_matched_inputs_and_resume_without_a_model(tmp_path, monkeypatch):
    variants = runner.contexts(num_samples=1, reward_modes=('ig', 'adalook_one_step', 'adalook_adaptive'))
    original, classes = dev.initialize, []
    def initialize(*args, **kwargs):
        result = original(*args, **kwargs)
        classes.append(type(result.search))
        return result
    monkeypatch.setattr(dev, 'initialize', initialize)
    w = wrapper()
    directory = runner.run_comparison(variants, output_root=tmp_path, wrapper=w)
    rows = runner.read_predictions(directory / 'predictions.jsonl')
    assert classes == [ABMEDALSearch] * 3
    assert len(rows) == 3 and all(r['status'] == 'completed' for r in rows)
    assert len({r['prompt_ids_hash'] for r in rows}) == len({r['seed'] for r in rows}) == 1
    assert sum(r['total_nfes'] for r in rows) == len(w.model.calls)
    assert all(r['search_nfes'] == 256 for r in rows)
    for name in ('summary.json', 'config.json', 'selection.json', 'report/report.md',
                 'report/summary.csv', 'report/per_question.csv', 'report/paired_results.csv',
                 'report/paired_results.json', 'report/comparison.png'):
        assert (directory / name).is_file()
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['status'] == 'completed'
    assert all(p['completed_pairs'] == 1 for p in summary['paired_comparisons'])
    assert all('mean_total_nfes' in r and 'mean_decoding_seconds' in r for r in summary['conditions'])
    before = (directory / 'predictions.jsonl').read_bytes()
    assert runner.run_comparison(runner.contexts(num_samples=1,
        reward_modes=('ig', 'adalook_one_step', 'adalook_adaptive')), resume=directory) == directory
    assert (directory / 'predictions.jsonl').read_bytes() == before
    changed = runner.contexts(num_samples=1, adalook_options={'adalook_alpha': .2})
    with pytest.raises(ValueError, match='Resume configuration'):
        runner.run_comparison(changed, resume=directory)


def test_interruption_during_temporary_comparison_preserves_attempt_and_retries(tmp_path):
    variants = runner.contexts(num_samples=1, reward_modes=('adalook_adaptive',))
    w = wrapper(BatchModel(failure=KeyboardInterrupt('stop temporary comparison'), fail_at=3))
    with pytest.raises(KeyboardInterrupt):
        runner.run_comparison(variants, output_root=tmp_path, wrapper=w)
    directory = next(tmp_path.glob('dev/A/*'))
    rows = runner.read_predictions(directory / 'predictions.jsonl')
    assert len(rows) == 1 and rows[0]['status'] == 'interrupted'
    assert rows[0]['search_nfes'] == 3 and rows[0]['temporary_candidate_nfes'] == 1
    assert rows[0]['evaluated_children'] == 0 and rows[0]['correct'] is None
    assert json.loads((directory / 'summary.json').read_text())['status'] == 'interrupted'
    runner.run_comparison(runner.contexts(num_samples=1, reward_modes=('adalook_adaptive',)),
                          resume=directory, wrapper=wrapper())
    rows = runner.read_predictions(directory / 'predictions.jsonl')
    assert len(rows) == 2 and sum(r['status'] == 'completed' for r in rows) == 1
    assert len(list((directory / 'traces').glob('*.jsonl'))) == 2
    assert json.loads((directory / 'summary.json').read_text())['conditions'][0]['completed'] == 1


def test_mismatched_variants_are_rejected_before_inference():
    variants = runner.contexts(num_samples=1)
    variants[1] = dev.context(condition='B', reward_options={'reward_mode': 'adalook_adaptive'})
    with pytest.raises(ValueError, match='share frozen inputs'):
        runner.configuration(variants)


def test_pair_report_counts_improvements_only_for_completed_pairs(tmp_path):
    names = ['ab_medal_ig', 'ab_medal_adalook_adaptive']
    cfg = dict(conditions=names, variants=[dict(examples=[dict(example_id=str(i)) for i in range(3)])])
    rows = []
    for name, answers in zip(names, ([False, True, True], [True, False, True])):
        for index, correct in enumerate(answers):
            rows.append(dict(condition=name, example_id=str(index), correct=correct,
                status='completed', seed=index, prompt_ids_hash=str(index), common_protocol_hash='same',
                total_nfes=300, decoding_seconds=2.))
    rows[-1]['status'] = 'interrupted'
    report = runner.write_report(tmp_path, rows, cfg, 'interrupted')
    pair = report['paired_comparisons'][0]
    assert (pair['completed_pairs'], pair['wrong_to_right'], pair['right_to_wrong']) == (2, 1, 1)


def test_notebook_default_comparison_uses_one_model_and_skips_demo(tmp_path, monkeypatch):
    from dlm_project.models import cache
    notebook = nbformat.read('notebooks/gsm8k/AB_MEDAL_MCTS_Colab.ipynb', as_version=4)
    nbformat.validate(notebook)
    w, loads, calls = wrapper(), [], []
    def load(ctx, **kwargs):
        loads.append(ctx)
        return w
    def compare(variants, **kwargs):
        calls.append((variants, kwargs))
        target = tmp_path / 'report-only'
        (target / 'report').mkdir(parents=True)
        (target / 'report/report.md').write_text('Mock comparison')
        return target
    monkeypatch.setattr(dev, 'load_model', load)
    monkeypatch.setattr(cache, 'cache_model', lambda **kw: tmp_path / 'model')
    monkeypatch.setattr(runner, 'run_comparison', compare)
    monkeypatch.setattr(dev, 'initialize', lambda *a, **kw: pytest.fail('comparison ran demo'))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda i: 'stub')
    scope = dict(Path=Path, DRIVE_ROOT=tmp_path, REPO_DIR=Path.cwd(), IN_COLAB=False)
    active = False
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        assert cell.execution_count is None and cell.outputs == []
        compile(cell.source, cell.id, 'exec')
        active |= cell.id == 'ab-settings'
        if active:
            exec(cell.source, scope)
    assert len(loads) == len(calls) == 1
    assert len(w.model.calls) == 1  # setup warm-up only; comparison runner is mocked
    assert [v['search']['reward_mode'] for v in calls[0][0]] == ['ig', 'adalook_adaptive']
    assert calls[0][1]['wrapper'] is w and scope['SINGLE_OUTPUT'] is None
