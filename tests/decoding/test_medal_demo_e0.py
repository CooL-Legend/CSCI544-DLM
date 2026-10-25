"""The interactive E0 path reuses cached search, counts and termination policy."""

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dlm_project.decoding import medal_demo as demo, medal_no_search
from dlm_project.decoding.mcts_decoding import MCTSConfig
from dlm_project.experiments.config import load_config
from tests.decoding.helpers import FakeModel, MASK, PROMPT
from tests.tasks.test_e0_dataset import Tokenizer


def prompt_settings():
    return load_config(Path(__file__).resolve().parents[2] / 'configs/gsm8k/experiment_0.yaml')['prompt']


def settings(cap, **overrides):
    return MCTSConfig(**dict(init_tokens=2, num_candidates=3, top_k_tokens=2, top_k_actions=3,
                            stop_mode='budget', max_model_calls=cap, **overrides))


def test_fixed_prompt_renders_offline_without_training_demonstrations(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(demo, 'get_gsm8k_demonstrations', lambda *a, **kw: pytest.fail('Loaded demonstrations'))
    model, prompts = FakeModel(), []
    wrapper = SimpleNamespace(model=model, encode_prompt=lambda text: prompts.append(text) or PROMPT.clone())
    text, ids, meta = demo.prepare_demo_prompt(wrapper, 'How many {items}?', tmp_path / 'absent.json',
                                              prompt_settings=prompt_settings())
    assert text.endswith('Problem: How many {items}?')
    assert text.count('Example 1:') == text.count('Example 2:') == 1
    assert meta['setup_wall_seconds'] == meta['planning_model_calls'] == 0
    assert torch.equal(ids, PROMPT) and model.seen == [] and prompts == [text]
    demo.show_prompt('How many {items}?', prompt=text)
    assert text in capsys.readouterr().out


def test_live_progress_matches_evaluations_and_excludes_printing_time(monkeypatch, capsys):
    clock, events = [0.0], []
    def print_with_cost(*args, **kwargs):
        builtins.print(*args, **kwargs)
        clock[0] += .25
    monkeypatch.setattr(demo, 'print', print_with_cost, raising=False)
    monkeypatch.setattr(demo, 'perf_counter', lambda: clock[0])
    config = MCTSConfig(init_tokens=5, top_k_tokens=2, top_k_actions=3,
                        stop_mode='budget', max_model_calls=70)
    model = FakeModel()
    initial = demo.run_initialization(model, PROMPT, MASK, gen_length=8, mcts=config,
                                      on_event=events.append, show_progress=True)
    output = capsys.readouterr().out
    assert all(f'Search NFEs: {count}/70' in output for count in (1, 32, 64, 70))
    assert '100.0%' in output and 'stop: model_call_budget' in output
    assert [event['search_model_calls'] for event in events if event['stage'] == 'forward'] == list(range(1, 71))
    assert len(model.seen) == initial.result.stats['search_model_calls'] == 70
    evaluations = [event for event in events if event['stage'] == 'evaluation']
    maximum = max(event['depth'] for event in evaluations)
    assert f'evaluated depth: {maximum}/5' in output
    assert f"candidates: {initial.result.stats['candidate_count']} | stop:" in output
    assert initial.wall_seconds == 0 and initial.progress_wall_seconds == clock[0] > 0
    baseline = demo.run_initialization(FakeModel(), PROMPT, MASK, gen_length=8, mcts=config)
    assert torch.equal(initial.state, baseline.state)
    assert initial.result.stats['selected_score'] == baseline.result.stats['selected_score']


@pytest.mark.parametrize('cap,source,shortlist', [(1, 'root', 1), (2, 'deepest_evaluated', 1),
                                               (5, 'deepest_evaluated', 3), (6, 'deepest_evaluated', 3),
                                               (12, 'target_depth', 3), (100, 'target_depth', 3)])
def test_demo_shortlist_fallback_metrics_and_completion_counts(cap, source, shortlist, capsys):
    model, events = FakeModel(), []
    initial = demo.run_initialization(model, PROMPT, MASK, gen_length=4, mcts=settings(cap), on_event=events.append)
    calls = len(model.seen)
    rows = demo.candidate_summaries(initial, tokenizer=Tokenizer())
    demo.show_candidates(initial, tokenizer=Tokenizer())
    assert len(model.seen) == calls
    assert len(rows) == (shortlist or 1) and rows[0]['selected']
    assert rows[0]['is_fallback'] == (source != 'target_depth')
    if not shortlist:
        assert 'fallback at depth' in capsys.readouterr().out
    if shortlist == 3:
        capsys.readouterr()
        demo.show_candidates(initial, tokenizer=Tokenizer(), limit=1)
        assert 'Shortlist underfilled' not in capsys.readouterr().out
    if cap == 12:
        third = next(event for event in events if event['stage'] == 'candidate' and event['candidate_count'] == 3)
        assert third['search_model_calls'] < calls == 12
    if cap == 100:
        assert calls < cap and initial.result.stats['stop_reason'] == 'frontier_exhausted'
    completion = demo.run_completion(initial)
    report = demo.show_metrics(initial, completion, prompt_seconds=2.0, setup_seconds=99.0,
                               score=demo.score_answer('#### 1', '#### 1'), display=False)
    assert report['search_nfes'] == calls
    assert report['completion_nfes'] == 4 - report['selected_depth']
    assert report['total_nfes'] == report['total_nfe'] == len(model.seen)
    assert report['total_nfes'] == calls + completion.model_calls
    assert report['budget_utilization'] == calls / cap
    assert report['shortlist_size'] == shortlist and report['shortlist_underfilled'] == (shortlist < 3)
    assert report['selection_source'] == source
    assert report['fallback_decode_used'] == (source != 'target_depth')
    assert report['fallback_rate'] == float(source != 'target_depth')
    assert report['accuracy_percent'] == 100 and report['parse_rate'] == 1
    assert 0 < report['accuracy_ci95_low'] < report['accuracy_ci95_high'] <= 1
    assert report['total_wall_seconds'] == report['decoding_seconds'] + 2.0
    assert report['decoding_seconds'] == report['search_seconds'] + report['completion_seconds']
    assert sum(stage['nfe'] for stage in report['stages']) == report['total_nfes']
    json.dumps(report, allow_nan=False)


def test_search_excludes_terminators_and_root_completion_matches_no_search_control():
    config = settings(1)
    initial = demo.run_initialization(FakeModel(), PROMPT, MASK, gen_length=64, mcts=config,
                                      search_excluded_token_ids=(1, 2))
    assert all(action.token_id == 3 for action in initial.search.root.analysis.actions)
    completion = demo.run_completion(initial, seed=1, tokenizer=Tokenizer(), termination_token_ids=(1, 2))
    control = medal_no_search.decode(FakeModel(), PROMPT, MASK, gen_length=64, steps=64, block_length=64,
                                     mcts=config, seed=1)
    assert torch.equal(completion.token_ids, control.token_ids)
    assert completion.model_calls == 64
    assert set(completion.token_ids[0, 2:].tolist()) >= {1, 2}
    decoded = completion.decoded_output
    assert len(decoded['full_output_ids']) == 64
    first = next(i for i, token in enumerate(decoded['full_output_ids']) if token in (1, 2))
    assert decoded['scoring_output_ids'] == decoded['full_output_ids'][:first]
    assert '<|' in decoded['raw_output']


@pytest.mark.parametrize('text,reference,accuracy', [('#### 1', '#### 1', 1.0),
                                                  ('Answer: 1', '#### 1', 0.0),
                                                  ('#### 1', None, None)])
def test_no_search_report_adds_no_forwards_and_keeps_control_cost_separate(text, reference, accuracy, capsys):
    model = FakeModel()
    result = medal_no_search.decode(model, PROMPT, MASK, gen_length=4, steps=4, block_length=4,
                                    mcts=settings(1), seed=1)
    calls = len(model.seen)
    report = demo.show_no_search_metrics(result, score=demo.score_answer(text, reference), prompt_seconds=2.0)
    assert report['method'] == 'medal_no_search'
    assert report['selection_source'] == 'no_search' and not report['fallback_decode_used']
    assert report['search_nfes'] == report['selected_depth'] == 0
    assert report['completion_nfes'] == report['total_nfes'] == len(model.seen) == calls == 4
    assert report['requested_search_budget'] is report['budget_utilization'] is None
    assert not report['shortlist_underfilled']
    assert report['search_seconds'] == 0
    assert report['completion_seconds'] == report['decoding_seconds']
    assert report['total_wall_seconds'] == report['decoding_seconds'] + 2.0
    assert report['accuracy'] == accuracy and report['parse_success'] == text.startswith('####')
    assert (report['accuracy_ci95_low'] is None) == (reference is None)
    assert '0 / 4 / 4' in capsys.readouterr().out
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('tokens,prefix', [([3, 1, 4, 2], [3]), ([1, 3, 4, 2], []),
                                         ([3, 4, 5, 3], [3, 4, 5, 3])])
def test_custom_sampler_keeps_full_outputs_and_scores_prefix_with_unknown_reference(tokens, prefix):
    initial = demo.run_initialization(FakeModel(), PROMPT, MASK, gen_length=4, mcts=settings(1),
                                      search_excluded_token_ids=(1, 2))
    def sampler(model, state, prompt_len, mask_id, config, **kwargs):
        model(state.to(model.device))
        state[:, prompt_len:] = torch.tensor(tokens)
        return SimpleNamespace(token_ids=state, model_calls=999)
    completion = demo.run_completion(initial, method=sampler, tokenizer=Tokenizer(), termination_token_ids=(1, 2))
    assert completion.model_calls == 1
    assert completion.decoded_output['full_output_ids'] == tokens
    assert completion.decoded_output['scoring_output_ids'] == prefix
    score = demo.score_answer(completion.decoded_output['scoring_text'], None)
    report = demo.show_metrics(initial, completion, score=score, display=False)
    assert report['accuracy'] is report['accuracy_ci95_low'] is report['exact_match'] is None
    assert report['parse_success'] == bool(prefix)


def test_medal_sampler_cannot_underreport_actual_completion_work(monkeypatch):
    initial = demo.run_initialization(FakeModel(), PROMPT, MASK, gen_length=4, mcts=settings(1))
    def broken_medal(model, state, prompt_len, mask_id, config, **kwargs):
        model(state.to(model.device))
        state[:, prompt_len:] = 3
        return SimpleNamespace(token_ids=state)
    monkeypatch.setattr(demo, '_resolve_sampler', lambda method: broken_medal)
    with pytest.raises(RuntimeError, match='one forward per remaining'):
        demo.run_completion(initial, method='medal')


@pytest.mark.parametrize('failure', [None, RuntimeError, KeyboardInterrupt])
def test_e0_optional_batch_uses_shared_policy_and_keeps_partial_metrics(tmp_path, monkeypatch, failure):
    from dlm_project.tasks.gsm8k import benchmark_dataset
    manifest = {'dataset': {'revision': 'a' * 40}, 'examples': [
        {'example_id': f'test:{i}', 'row_id': i, 'question': f'Question {i}?',
         'reference_solution': 'REFERENCE_ONLY\n#### 1'} for i in range(2)]}
    monkeypatch.setattr(benchmark_dataset, 'get_manifest', lambda *a, **kw: manifest)
    monkeypatch.setattr(demo, 'get_gsm8k_demonstrations', lambda *a, **kw: pytest.fail('Loaded training data'))
    model, prompts, attempts = FakeModel(lambda _: [0, .4, .3, .1, .1, .1]), [], []
    wrapper = SimpleNamespace(model=model, tokenizer=Tokenizer(), mask_id=MASK,
                              encode_prompt=lambda text: prompts.append(text) or PROMPT.clone())
    def sampler(model, state, prompt_len, mask_id, config, **kwargs):
        model(state.to(model.device))
        attempts.append(True)
        if failure and len(attempts) == 2:
            raise failure('Controlled stop')
        state[:, prompt_len:] = torch.tensor([3, 2, 4, 1])
        return SimpleNamespace(token_ids=state)
    arguments = dict(num_samples=2, gen_length=4, mcts=settings(1), method=sampler,
                     prompt_settings=prompt_settings(), show_progress=True)
    if failure:
        with pytest.raises(failure):
            demo.run_demo_batch(wrapper, tmp_path / 'absent.json', tmp_path / 'test.json', tmp_path / 'runs', **arguments)
        directory = next((tmp_path / 'runs').rglob('summary.json')).parent
        summary = json.loads((directory / 'summary.json').read_text())
    else:
        directory, summary = demo.run_demo_batch(wrapper, tmp_path / 'absent.json', tmp_path / 'test.json',
                                                 tmp_path / 'runs', **arguments)
    rows = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()]
    count = 1 if failure else 2
    assert len(rows) == summary['processed'] == count
    assert summary['status'] == ('interrupted' if failure is KeyboardInterrupt else 'failed' if failure else 'completed')
    assert summary['total_nfes'] == summary['total_nfe'] == count * 2
    assert summary['search_nfes'] == summary['completion_nfes'] == count
    assert summary['parse_rate'] == summary['accuracy'] == summary['fallback_rate'] == 1
    assert summary['mean_budget_utilization'] == summary['shortlist_underfilled_rate'] == 1
    assert summary['mean_selected_depth'] == summary['max_evaluated_depth'] == 0
    assert summary['accuracy_ci95_low'] < summary['accuracy_ci95_high']
    assert all(row['full_output_ids'] == [3, 2, 4, 1] and row['scoring_output_ids'] == [3] for row in rows)
    assert all(row['prompt'].endswith('Problem: ' + row['question']) for row in rows)
    assert all('REFERENCE_ONLY' not in prompt for prompt in prompts)
    saved = json.loads((directory / 'config.json').read_text())
    assert saved['termination_token_ids'] == [1, 2] and saved['prompt'] == prompt_settings()
