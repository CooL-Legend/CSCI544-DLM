"""Fixed E0 matrix, real invocation accounting around stub decoders, and reports."""

import copy
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import nbformat
import pytest
import torch

from dlm_project.decoding.decoder_result import DecodeResult
from dlm_project.decoding.mcts_decoding import Node, candidate_value_record, state_record
from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.config import load_config
from dlm_project.models.profiles import get_profile
from dlm_project.paths import PROJECT_ROOT as ROOT
from dlm_project.tasks.gsm8k import experiment_0 as e0
from dlm_project.tasks.gsm8k.e0_report import generate_report, load_records
from dlm_project.tasks.gsm8k.experiment_protocol import get_manifest, load_shared_protocol
from tests.tasks.test_e0_dataset import Tokenizer


CONDITIONS = ['confidence_t0', 'medal_lc7_b256', 'medal_lc7_b512', 'medal_lc10_b512', 'medal_lc10_b1024']
PROMPT = torch.tensor([[5, 4]], dtype=torch.long)


class CounterModel:
    """Cheap invocation-only model: decoder mathematics has separate unit tests."""
    device = 'cpu'
    config = SimpleNamespace(eos_token_id=1, vocab_size=6)

    def __init__(self):
        self.calls = 0

    def __call__(self, ids):
        self.calls += 1
        return None


def wrapper():
    prompts = []
    def encode(text):
        prompts.append(text)
        return PROMPT.clone()
    return SimpleNamespace(model=CounterModel(), tokenizer=Tokenizer(), device='cpu', mask_id=0,
                           identity=get_profile(), encode_prompt=encode), prompts


def stub_decode(counter, prompt_ids, mask_id, *, gen_length, mcts=None, **kwargs):
    """Exercise every forward through CountingModel without doing inference."""
    budget = mcts['max_model_calls'] if mcts else 0
    depth = mcts['init_tokens'] if mcts else 0
    nfes = budget + gen_length - depth
    for _ in range(nfes):
        counter(prompt_ids)
    details = {}
    if mcts:
        candidates = []
        for i, reward in enumerate((.3, .2, .1), start=1):
            state = torch.zeros((1, prompt_ids.shape[1] + gen_length), dtype=torch.long)
            state[:, :prompt_ids.shape[1]] = prompt_ids
            state[:, prompt_ids.shape[1]:prompt_ids.shape[1] + depth] = 3
            state[0, prompt_ids.shape[1]] = i + 2
            node = Node(i, state, depth=depth, visits=1, reward_sum=reward, immediate_reward=reward)
            candidates.append({**candidate_value_record(node), 'model_calls': 0,
                               **state_record(state, prompt_ids.shape[1], mask_id)})
        parents, children = (budget + 3) // 5, budget - 1
        details = dict(search_model_calls=budget, selected_depth=depth, max_depth=depth,
                       average_search_depth=depth / 2, candidate_count=3, target_depth_candidate_count=3,
                       shortlist_size=3, shortlist_underfilled=False, expanded_parents=parents,
                       evaluated_children=children, realized_branching_factor=children / parents,
                       fallback_decode_used=False, selection_source='target_depth', stop_reason='model_call_budget',
                       selected_score=.3, selected_node_id=1, candidate_evaluations=candidates,
                       candidate_shortlist=[{k: c[k] for k in ('node_id', 'depth', 'visits', 'reward_sum',
                                                               'q_value', 'reward', 'state_hash', 'immediate_gain')}
                                            for c in candidates],
                       candidate_evaluation_model_calls=0, search_latency_seconds=0., completion_latency_seconds=0.,
                       selected_initialization={k: candidates[0][k] for k in ('output_ids', 'committed', 'remaining_masks', 'partial_output')})
    output = torch.cat((prompt_ids, torch.full((1, gen_length), 3, dtype=torch.long)), dim=1)
    return DecodeResult(output, nfes, details=details)


@pytest.fixture
def prepared(monkeypatch):
    instance, prompts = wrapper()
    loads = []
    def factory(config):
        loads.append(copy.deepcopy(config))
        return instance
    monkeypatch.setattr(e0, '_load_model', factory)
    monkeypatch.setattr(e0, 'DECODERS', {'confidence': stub_decode, 'medal': stub_decode})
    return instance, prompts, loads


def source(tmp_path, stage='dev', order='condition'):
    result = load_config(ROOT / 'configs/gsm8k/experiment_0.yaml')
    result['experiment_root'] = str(tmp_path)
    result['evaluation'].update(stage=stage, run_order=order)
    return result


def records(directory):
    return [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('stage,order,n', [('dev', 'example', 5), ('test', 'condition', 200)])
def test_five_conditions_one_load_fixed_samples_accounting_and_offline_report(tmp_path, prepared, stage, order, n):
    instance, prompts, loads = prepared
    config = source(tmp_path, stage, order)
    original = copy.deepcopy(config)
    directory, summary = e0.run_experiment(config, run_manifest=tmp_path / 'receipt.json')
    assert config == original and len(loads) == 1
    assert summary['complete'] and summary['common_n'] == n
    assert len(summary['conditions']) == 5 and len(summary['gains']) == 4
    rows = records(directory)
    assert len(rows) == 5 * n
    ids = loads[0]['run']['example_ids']
    expected = ([(example, condition) for condition in CONDITIONS for example in ids] if order == 'condition'
                else [(example, condition) for example in ids for condition in CONDITIONS])
    assert [(r['example_id'], r['condition']) for r in rows] == expected
    assert instance.model.calls == sum(r['total_nfes'] for r in rows) + 1
    assert sum(r['total_nfes'] for r in rows) == n * 3550
    assert len(prompts) == n + 1
    for example in ids:
        paired = [r for r in rows if r['example_id'] == example]
        assert len({r['prompt_ids_hash'] for r in paired}) == len({r['prompt_hash'] for r in paired}) == 1
        assert len({r['seed'] for r in paired}) == 1
    for row in rows:
        assert row['reference_solution'] not in row['prompt']
        assert row['prompt'].endswith('Problem: ' + row['question'])
        assert row['protocol_version'] == 'e0_fixed_budget_v5'
        assert row['completion_nfes'] == 256 - row['selected_depth']
        assert row['search_nfes'] <= row['requested_search_budget']
        assert row['total_nfes'] == row['search_nfes'] + row['completion_nfes']
    receipt = json.loads((tmp_path / 'receipt.json').read_text())
    assert receipt['condition_ids'] == CONDITIONS and receipt['status'] == 'completed'
    assert receipt['common_protocol_hash'] == summary['common_protocol_hash']
    assert generate_report(directory)['conditions'] == summary['conditions']
    text = (directory / 'report/report.md').read_text()
    assert 'Expanded parents' in text and 'Lc=7,10' in text
    assert 'no-search MEDAL control' not in text
    for name in ('summary.csv', 'observed_summary.csv', 'results.csv', 'accuracy_gains.csv'):
        assert (directory / 'report' / name).is_file()
    for key in ('mean_expanded_parents', 'mean_realized_branching_factor', 'mean_target_depth_candidate_count'):
        assert key in summary['conditions'][1]
    if stage == 'test':
        assert len(rows) == 1000


def test_fixed_config_and_migration_errors(tmp_path):
    cfg = source(tmp_path)
    resolved = e0.resolve_settings(cfg)
    assert e0.resolve_settings(resolved) == resolved
    assert [r['id'] for r in resolved['conditions']] == CONDITIONS
    assert [(r['requested_depth'], r['search_budget']) for r in resolved['conditions']] == [(None, 0), (7, 256), (7, 512), (10, 512), (10, 1024)]
    assert e0.resolve_settings(cfg, stage='test')['data']['num_samples'] == 200
    for override in ({'init_tokens': 7}, {'search_budgets': [256, 512, 1024]}, {'search_mode': 'candidates'}):
        with pytest.raises(ValueError, match='fixed A/B/C/D'):
            e0.resolve_settings(cfg, **override)
    for override in ({'num_samples': 2}, {'num_samples': True}, {'dataset_revision': 'a' * 40}, {'run_order': 'random'}):
        with pytest.raises(ValueError):
            e0.resolve_settings(cfg, **override)
    changes = [('prompt', 'template', 'Solve {GSM8K_question}'), ('generation', 'gen_length', 128),
               ('data', 'sample_seed', 2), ('decoding', 'completion_temperature', 1), ('model', 'dtype', 'float32')]
    for section, key, value in changes:
        changed = copy.deepcopy(cfg)
        changed[section][key] = value
        with pytest.raises(ValueError):
            e0.resolve_settings(changed)
    bad = copy.deepcopy(cfg)
    bad['decoding']['mcts']['num_candidates'] = 5
    with pytest.raises(ValueError):
        e0.resolve_settings(bad)


def test_invalid_manifest_fails_before_model_load(tmp_path, prepared):
    _, _, loads = prepared
    changed = get_manifest()
    changed['examples'][0], changed['examples'][1] = changed['examples'][1], changed['examples'][0]
    target = tmp_path / 'tampered.json'
    target.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='hash'):
        e0.run_experiment(source(tmp_path), manifest_path=target)
    assert not loads


def test_real_decoders_use_the_fixed_matrix_and_rebuild_report(tmp_path, monkeypatch):
    from dlm_project.decoding import confidence, mcts_confidence
    from tests.decoding.helpers import FakeModel
    instance, _ = wrapper()
    model = FakeModel(lambda _: [0, .05, .04, .6, .2, .11])
    model.config = CounterModel.config
    instance.model = model
    monkeypatch.setattr(e0, 'DECODERS', {'confidence': confidence.decode, 'medal': mcts_confidence.decode})
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        directory, summary = e0.run_experiment(source(tmp_path), model_factory=lambda _: instance)
    finally:
        torch.set_num_threads(threads)
    rows = records(directory)
    assert summary['complete'] and len(rows) == 25
    assert len(model.seen) == sum(r['total_nfes'] for r in rows) + 1
    calls_before_reporting = len(model.seen)
    assert generate_report(directory)['conditions'] == summary['conditions']
    assert len(model.seen) == calls_before_reporting
    assert (directory / 'report' / 'compute_allocation.csv').is_file()
    for row in rows:
        if row['method'] == 'medal':
            audit = json.loads((directory / row['search_audit_file']).read_text())
            assert len(audit['nodes']) == row['search_nfes']
            assert all(n['state_hash'] and n['evaluation_order'] == i for i, n in enumerate(audit['nodes']))
            allocation = row['search_audit_metrics']
            assert sum(allocation['nfes_per_depth'].values()) == row['search_nfes']
            assert allocation['root_analysis_nfes'] + sum(allocation['nfes_per_root_subtree'].values()) == row['search_nfes']
            assert allocation['decision_counts'] is None
            details = row['details']
            assert row['shortlist_size'] == min(3, row['candidate_count'])
            assert row['selected_depth'] == row['max_depth']
            assert details['candidate_evaluation_model_calls'] == 0
            assert details['simulation_decoding_steps'] == 0
            for position in details['selected_initialization']['committed']:
                assert row['full_output_ids'][position['output_position']] == position['token_id']


@pytest.mark.parametrize('failure', ['exception', 'interrupt'])
def test_partial_run_preserves_finished_rows_and_active_forward_cost(tmp_path, prepared, monkeypatch, failure):
    calls = []
    def fail_second(counter, prompt, mask_id, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            counter(prompt)
            counter(prompt)
            if failure == 'interrupt':
                raise KeyboardInterrupt('controlled interruption')
            raise RuntimeError('controlled failure')
        return stub_decode(counter, prompt, mask_id, **kwargs)
    monkeypatch.setitem(e0.DECODERS, 'confidence', fail_second)
    receipt = tmp_path / 'receipt.json'
    with pytest.raises(KeyboardInterrupt if failure == 'interrupt' else RuntimeError):
        e0.run_experiment(source(tmp_path), run_manifest=receipt)
    saved = json.loads(receipt.read_text())
    directory = Path(saved['run_dir'])
    assert saved['status'] == ('interrupted' if failure == 'interrupt' else 'failed')
    assert len(records(directory)) == 1
    summary = generate_report(directory)
    assert not summary['complete'] and summary['common_n'] == 0
    assert summary['unfinished_attempt_nfes'] == 2
    assert summary['observed_conditions'][0]['n'] == 1


def test_cli_sigterm_flushes_and_restores_handler(tmp_path, prepared, monkeypatch):
    def terminated(counter, prompt, mask, **kwargs):
        counter(prompt)
        signal.raise_signal(signal.SIGTERM)
    monkeypatch.setitem(e0.DECODERS, 'confidence', terminated)
    receipt = tmp_path / 'receipt.json'
    monkeypatch.setattr(sys, 'argv', ['experiment_0_baseline_test.py', '--output-dir', str(tmp_path / 'runs'),
                                    '--run-manifest', str(receipt)])
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        e0.main()
    assert signal.getsignal(signal.SIGTERM) == previous
    directory = Path(json.loads(receipt.read_text())['run_dir'])
    assert generate_report(directory)['unfinished_attempt_nfes'] == 1


def test_historical_v4_report_and_trailing_partial_record(tmp_path):
    fixture = json.loads((ROOT / 'tests/fixtures/e0_v4_run.json').read_text())
    for key in ('config', 'setup', 'manifest'):
        save_json(tmp_path / (key + '.json'), fixture[key])
    prediction = tmp_path / 'predictions.jsonl'
    prediction.write_text(''.join(json.dumps(row) + '\n' for row in fixture['records']))
    summary = generate_report(tmp_path)
    assert summary['complete'] and len(summary['conditions']) == 6
    assert summary['protocol_version'] == 'e0_fixed_budget_v4'
    assert 'no-search MEDAL control' in (tmp_path / 'report/report.md').read_text()
    with prediction.open('a') as handle:
        handle.write('{"interrupted":')
    assert generate_report(tmp_path)['warnings']


def test_offline_report_detects_changed_q_and_selection(tmp_path, prepared):
    directory, _ = e0.run_experiment(source(tmp_path))
    saved = json.loads((directory / 'config.json').read_text())
    rows = records(directory)
    searched = next(r for r in rows if r['method'] == 'medal')
    searched['details']['candidate_evaluations'][0]['q_value'] = 999
    (directory / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError, match='reward'):
        load_records(directory, saved)


@pytest.mark.parametrize('label', ['Dev', 'Evaluation', 'Testing'])
def test_unchanged_notebooks_are_valid_but_explicitly_legacy(label):
    notebook = nbformat.read(ROOT / f'notebooks/gsm8k/Experiment_0_Baseline_Test_{label}.ipynb', as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            compile(cell.source, cell.id, 'exec')
    cells = {cell.id: cell.source for cell in notebook.cells}
    assert 'e0_fixed_budget_v4' in cells['e0-run']
    assert '--init-tokens' in cells['e0-run']
    with pytest.raises(ValueError, match='fixed A/B/C/D'):
        e0.resolve_settings(load_config(ROOT / 'configs/gsm8k/experiment_0.yaml'), init_tokens=5)
