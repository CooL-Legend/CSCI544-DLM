"""Frozen AB inputs, complete audit exports, real notebook cells, batching gates."""

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import nbformat
import pytest
import torch

from dlm_project.tasks.gsm8k import ab_development as dev, ab_batch_study as study
from dlm_project.tasks.gsm8k.experiment_protocol import build_prompt
from dlm_project.decoding import search_audit
from tests.decoding.test_ab_batch import BatchModel


class Tokenizer:
    eos_token_id = pad_token_id = 4
    def get_vocab(self):
        return {'[MASK]': 0, 'one': 1, 'two': 2, 'three': 3, '<|endoftext|>': 4, '<|eot_id|>': 5}
    def convert_ids_to_tokens(self, i):
        return next(k for k, v in self.get_vocab().items() if v == i)
    def decode(self, ids, skip_special_tokens=False):
        return ' '.join(self.convert_ids_to_tokens(i) for i in ids if not skip_special_tokens or i not in (0, 4, 5))


def wrapper(model=None):
    ctx = dev.context()
    return SimpleNamespace(model=model or BatchModel(), tokenizer=Tokenizer(), mask_id=0,
        identity={k: ctx['common']['model'][k] for k in ('family', 'repo_id', 'revision')},
        encode_prompt=lambda text: torch.tensor([[3] * (2 + len(text) % 4)]))


@pytest.mark.parametrize('condition,depth,budget', [('A', 10, 256), ('B', 10, 512), ('C', 10, 1024)])
def test_shared_conditions_prompts_seeds_and_order(condition, depth, budget):
    ctx = dev.context(condition=condition, indices=(0, 1))
    w = wrapper()
    prepared = dev.prepare(w, ctx)
    assert ctx['search']['init_tokens'] == depth and ctx['search']['max_model_calls'] == budget
    assert ctx['search']['stop_mode'] == 'budget'
    for p in prepared:
        assert p['text'] == build_prompt(p['example']['question'])
        assert p['seed'] == 1 + 5 * p['example']['row_id']
    assert [p['example']['example_id'] for p in prepared] == [e['example_id'] for e in ctx['examples']]
    assert w.model.calls == []


def test_single_serial_and_offline_reconstruction_and_candidate_exports(tmp_path):
    w, ctx = wrapper(), dev.context()
    item = dev.prepare(w, ctx)[0]
    initial = dev.initialize(w, item, ctx, tmp_path)
    completion = dev.complete(w, item, initial, ctx, tmp_path)
    calls = len(w.model.calls)
    rng = copy.deepcopy(initial.search.rng.bit_generator.state)
    record = dev.finish_single(w, item, ctx, initial, completion, tmp_path)
    rebuilt = search_audit.load(tmp_path)
    assert len(w.model.calls) == calls == record['total_nfes']
    assert initial.search.rng.bit_generator.state == rng
    assert record['search_nfes'] == 256 and record['completion_nfes'] == 256 - record['selected_depth']
    assert all(record[k] == v for k, v in rebuilt.items())
    selection = json.loads((tmp_path / 'selection.json').read_text())
    assert len(selection['candidate_statistics']) == record['candidate_count']
    assert len(selection['shortlist_ids']) == min(3, record['candidate_count'])
    assert len((tmp_path / 'nodes.jsonl').read_text().splitlines()) == 256
    assert len((tmp_path / 'forwards.jsonl').read_text().splitlines()) == record['total_nfes']
    # A fresh isolated process can reconstruct with only the standard library.
    source = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('audit', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps(module.load(sys.argv[2])))
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
"""
    offline = subprocess.check_output([sys.executable, '-I', '-c', source, search_audit.__file__, str(tmp_path)], text=True)
    assert json.loads(offline) == rebuilt
    dev.plot_metrics(tmp_path)
    assert (tmp_path / 'compute_allocation.png').read_bytes().startswith(b'\x89PNG')


def test_five_serial_samples_are_same_frozen_pool_with_independent_budgets(tmp_path):
    w, ctx = wrapper(), dev.context(indices=tuple(range(5)))
    prepared = dev.prepare(w, ctx)
    result = dev.run_samples(w, prepared, ctx, tmp_path)
    assert result['summary']['completed'] == 5
    assert result['summary']['search_nfes'] == 5 * 256
    assert result['summary']['total_model_invocations'] == len(w.model.calls)
    assert [r['example_id'] for r in result['records']] == [e['example_id'] for e in ctx['examples']]
    assert all(r['search_nfes'] == 256 for r in result['records'])
    assert result['summary']['decoding_seconds'] == sum(g['decoding_seconds'] for g in result['summary']['groups'])


@pytest.mark.parametrize('error,status', [(RuntimeError('bad forward'), 'error'), (KeyboardInterrupt('stop'), 'interrupted')])
def test_failed_root_is_saved_without_fake_success_or_score(tmp_path, error, status):
    w, ctx = wrapper(BatchModel(failure=error, fail_at=1)), dev.context()
    item = dev.prepare(w, ctx)[0]
    with pytest.raises(type(error)):
        dev.initialize(w, item, ctx, tmp_path)
    record = json.loads((tmp_path / 'metrics.json').read_text())
    assert record['status'] == status and record['correct'] is None
    assert record['search_nfes'] == record['uncompleted_evaluation_nfes'] == 1
    assert record['completion_nfes'] == 0
    assert not (tmp_path / 'completion.json').exists()


def test_batch_run_partial_last_group_and_no_double_counted_runtime(tmp_path):
    w, ctx = wrapper(), dev.context(indices=(0, 1, 2))
    result = dev.run_samples(w, dev.prepare(w, ctx), ctx, tmp_path, batch_size=2)
    assert result['summary']['completed'] == 3
    assert [g['batch_size'] for g in result['summary']['groups']] == [2, 1]
    assert result['summary']['total_model_invocations'] == len(w.model.calls)
    assert result['summary']['total_nfes'] == sum(ids.shape[0] for ids, _ in w.model.calls)


def test_actual_notebook_cells_execute_one_model_and_all_metrics(tmp_path, monkeypatch):
    notebook = nbformat.read('notebooks/gsm8k/AB_MEDAL_MCTS_Colab.ipynb', as_version=4)
    nbformat.validate(notebook)
    w, loads = wrapper(), []
    def load(ctx, **kwargs):
        loads.append((ctx, kwargs))
        return w
    monkeypatch.setattr(dev, 'load_model', load)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda i: 'stub')
    scope = dict(Path=Path, DRIVE_ROOT=tmp_path, Image=lambda **kw: kw, display=lambda value: None)
    # The import cell imports IPython's display; rendering local PNGs is safe.
    active = False
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        assert cell.outputs == [] and cell.execution_count is None
        compile(cell.source, cell.id, 'exec')
        active |= cell.id == 'ab-settings'
        if active:
            exec(cell.source, scope)
    assert len(loads) == 1
    assert scope['RUN_SERIAL_DEV'] is scope['RUN_BATCH_STUDY'] is False
    assert scope['record']['total_nfes'] == len(w.model.calls) - 1  # separate warm-up
    assert scope['record']['search_nfes'] == 256
    assert scope['offline_metrics']['decision_counts']['GEN'] == 255


def test_padding_and_full_run_parity_gate_precedes_benchmark(tmp_path):
    w, ctx = wrapper(), dev.context(indices=(0, 1, 2, 3))
    prepared = dev.prepare(w, ctx)
    baseline = dev.run_samples(w, prepared, ctx, tmp_path / 'serial')
    probe = study.padding_probe(w, prepared, ctx, baseline, 4)
    assert probe['passed'] and len(probe['rows']) == 16
    assert all(r['max_abs_logit_difference'] == 0 for r in probe['rows'])
    batched = dev.run_samples(w, prepared, ctx, tmp_path / 'batch', batch_size=4)
    assert study.compare_runs(baseline, batched)['passed']
    bad = copy.deepcopy(batched)
    bad['records'][0]['selected_state_ids'][0] = -1
    assert not study.compare_runs(baseline, bad)['passed']
    w.model.ignore_attention = True
    probe = study.padding_probe(w, prepared, ctx, baseline, 4)
    assert not probe['passed']


@pytest.mark.parametrize('change', ['common', 'search', 'examples', 'seed', 'ids', 'order'])
def test_protocol_and_prepared_mutations_fail_before_inference(tmp_path, change):
    w, ctx = wrapper(), dev.context(indices=(0, 1))
    prepared = dev.prepare(w, ctx)
    if change == 'common':
        ctx['common']['generation']['gen_length'] = 128
    elif change == 'search':
        ctx['search']['max_model_calls'] = 255
    elif change == 'examples':
        ctx['examples'][0]['question'] = 'Changed question'
    elif change == 'seed':
        prepared[0]['seed'] += 1
    elif change == 'ids':
        prepared[0]['ids'][0, 0] = 2
    else:
        prepared.reverse()
    with pytest.raises(ValueError):
        dev.run_samples(w, prepared, ctx, tmp_path)
    assert w.model.calls == []


def test_failed_completion_retains_search_and_attempt_accounting(tmp_path):
    w, ctx = wrapper(BatchModel(failure=KeyboardInterrupt('stop completion'), fail_at=260)), dev.context()
    prepared = dev.prepare(w, ctx)[0]
    initial = dev.initialize(w, prepared, ctx, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        dev.complete(w, prepared, initial, ctx, tmp_path)
    record = json.loads((tmp_path / 'metrics.json').read_text())
    assert record['status'] == 'interrupted'
    assert record['search_nfes'] == 256 and record['completion_nfes'] == 4
    assert record['total_nfes'] == len(w.model.calls) == 260
    forwards = [json.loads(line) for line in (tmp_path / 'forwards.jsonl').read_text().splitlines()]
    assert len(forwards) == 260 and forwards[-1]['status'] == 'attempted'
    assert record['selected_state_ids'] is not None and record['correct'] is None
    assert search_audit.load(tmp_path)['search_nfes'] == 256


def study_stubs(monkeypatch, *, blocked_size=None, interruption=None):
    """Exercise orchestration gates separately from the real decoder parity test."""
    calls = []
    def run(w, prepared, ctx, directory, *, batch_size):
        phase = Path(directory).name
        calls.append(('run', phase, batch_size))
        stopped = interruption == phase
        group = dict(peak_allocated_bytes=None, peak_reserved_bytes=None,
                     total_model_invocations=400 // batch_size, total_nfes=400)
        summary = dict(status='interrupted' if stopped else 'completed', error='stop' if stopped else None,
            decoding_seconds=4 / batch_size, groups=[group], search_nfes=200, completion_nfes=200,
            total_nfes=400, total_model_invocations=400 // batch_size, accuracy=.5, parsing_rate=1., fallback_rate=0.)
        return dict(records=[], summary=summary)
    def probe(w, prepared, ctx, baseline, size):
        calls.append(('probe', size))
        return dict(passed=size != blocked_size and interruption != 'probe', interrupted=interruption == 'probe')
    def warm(w, prepared, ctx, *, batch_size):
        calls.append(('warm', batch_size))
        if interruption == 'warm':
            raise KeyboardInterrupt('warm-up')
        if interruption == 'oom' and batch_size == 4:
            raise torch.OutOfMemoryError('explicit batch 4 OOM')
        return dict(model_invocations=1, sequence_nfes=batch_size, seconds=.01)
    monkeypatch.setattr(study, 'run_samples', run)
    monkeypatch.setattr(study, 'padding_probe', probe)
    monkeypatch.setattr(study, 'warmup', warm)
    monkeypatch.setattr(study, 'compare_runs', lambda a, b: dict(passed=b['summary']['status'] == 'completed'))
    return calls


def test_batch_study_rotation_and_validation_before_timing(tmp_path, monkeypatch):
    calls = study_stubs(monkeypatch)
    w, ctx = wrapper(), dev.context(indices=(0, 1, 2, 3))
    report = study.run_study(w, dev.prepare(w, ctx), ctx, tmp_path)
    assert report['status'] == 'completed'
    assert [t['batch_size'] for t in report['trials']] == [1, 2, 4, 2, 4, 1, 4, 1, 2]
    assert [r['speedup_over_serial'] for r in report['summary']] == [1, 2, 4]
    assert calls[:5] == [('run', 'validation_serial', 1), ('probe', 2), ('run', 'validation_batch_2', 2),
                         ('probe', 4), ('run', 'validation_batch_4', 4)]
    assert (tmp_path / 'batch_comparison.png').is_file()


def test_failed_padding_parity_never_warms_or_times_that_batch_size(tmp_path, monkeypatch):
    calls = study_stubs(monkeypatch, blocked_size=4)
    w, ctx = wrapper(), dev.context(indices=(0, 1, 2, 3))
    report = study.run_study(w, dev.prepare(w, ctx), ctx, tmp_path)
    assert not report['validation']['4']['passed']
    assert not any(c[-1] == 4 for c in calls if c[0] in ('run', 'warm'))
    assert report['summary'][2]['successful_repetitions'] == 0
    assert report['summary'][2]['speedup_over_serial'] is None


@pytest.mark.parametrize('interruption', ['validation_serial', 'probe', 'validation_batch_2', 'warm', 'repeat_1_batch_2'])
def test_batch_study_interruption_always_exports_partial_report(tmp_path, monkeypatch, interruption):
    study_stubs(monkeypatch, interruption=interruption)
    w, ctx = wrapper(), dev.context(indices=(0, 1, 2, 3))
    report = study.run_study(w, dev.prepare(w, ctx), ctx, tmp_path)
    assert report['status'] == 'interrupted'
    assert json.loads((tmp_path / 'batch_study.json').read_text()) == report
    assert (tmp_path / 'batch_summary.csv').is_file()
    if interruption == 'repeat_1_batch_2':
        assert len(report['trials']) == 2
        assert report['summary'][0]['successful_repetitions'] == 1
        assert report['summary'][1]['successful_repetitions'] == 0


def test_batch_study_oom_retains_earlier_evidence_without_resizing(tmp_path, monkeypatch):
    calls = study_stubs(monkeypatch, interruption='oom')
    w, ctx = wrapper(), dev.context(indices=(0, 1, 2, 3))
    report = study.run_study(w, dev.prepare(w, ctx), ctx, tmp_path, repetitions=1)
    assert report['validation']['4']['passed']
    assert 'OutOfMemoryError' in report['warmup']['4']['error']
    assert [t['batch_size'] for t in report['trials']] == [1, 2]
    assert calls.count(('warm', 4)) == 1
