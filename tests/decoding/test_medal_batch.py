"""The cooperative MEDAL adapter preserves the ordinary UCB search exactly."""

import pytest
import torch

from dlm_project.decoding import mcts_decoding as medal, search_audit
from dlm_project.decoding.medal_batch import CooperativeMEDALSearch
from dlm_project.decoding.ab_batch import execute_trees
from tests.decoding.test_ab_batch import BatchModel


def trees(model, cls, budgets=(1, 2, 4, 9, 20, 64)):
    return [cls(model, torch.tensor([[3] * (i + 1)]), 0, 5,
                medal.MCTSConfig(init_tokens=2, max_model_calls=budget, stop_mode='budget'),
                seed=i + 7, search_excluded_token_ids=(4, 5)) for i, budget in enumerate(budgets)]


def assert_same_tree(left, right):
    assert left.stop_reason == right.stop_reason
    assert left.model_calls == right.model_calls
    assert left.result().selected.node_id == right.result().selected.node_id
    assert len(left.nodes) == len(right.nodes)
    for a, b in zip(left.nodes, right.nodes):
        assert torch.equal(a.state, b.state)
        assert (a.node_id, a.depth, a.visits, a.reward_sum, a.immediate_reward) == (
            b.node_id, b.depth, b.visits, b.reward_sum, b.immediate_reward)
    for stage in ('selection', 'node_evaluated', 'backpropagation', 'candidate', 'search_selection'):
        assert [e for e in left.trace if e['stage'] == stage] == [e for e in right.trace if e['stage'] == stage]
    assert not hasattr(right, 'posteriors')
    assert not any(e['stage'] == 'adaptive_decision' for e in right.trace)


@pytest.mark.parametrize('batch_size', [1, 2, 4])
def test_ucb_rounds_and_completion_match_unmodified_medal(batch_size):
    ordinary = trees(BatchModel(), medal.MCTSSearch)
    outputs = []
    for tree in ordinary:
        result = tree.run()
        outputs.append(medal.complete_llada(tree.model, result.selected.state,
            tree.prompt_len, tree.mask_id, temperature=0.).token_ids)
    model = BatchModel()
    cooperative = trees(model, CooperativeMEDALSearch)
    runs = [execute_trees(cooperative[i:i+batch_size], pad_token_id=4)
            for i in range(0, len(cooperative), batch_size)]
    assert all(r.status == 'completed' for r in runs)
    for left, right, output, completion in zip(ordinary, cooperative, outputs,
                                              [c for r in runs for c in r.completions]):
        assert_same_tree(left, right)
        assert torch.equal(output, completion.token_ids)
        audit = search_audit.from_events(right.trace,
            prompt_ids=right.root.state[0, :right.prompt_len].tolist(), mask_id=0, gen_length=5,
            requested_depth=2, requested_budget=right.config.max_model_calls, search_nfes=right.model_calls)
        metrics = search_audit.metrics(audit)
        assert metrics['decision_counts'] is None
        assert metrics['search_nfes'] == sum(metrics['nfes_per_depth'].values())
    assert sum(r.metrics['total_model_invocations'] for r in runs) == len(model.calls)
    assert sum(r.metrics['total_nfes'] for r in runs) == sum(ids.shape[0] for ids, _ in model.calls)


def test_cooperative_step_run_matches_medal_without_bayesian_draws(monkeypatch):
    from dlm_project.decoding.ab_medal import GaussianPosterior
    def fail(*args):
        pytest.fail('MEDAL must not create or update Bayesian search statistics')
    monkeypatch.setattr(GaussianPosterior, 'sample', fail)
    monkeypatch.setattr(GaussianPosterior, 'update', fail)
    for left, right in zip(trees(BatchModel(), medal.MCTSSearch), trees(BatchModel(), CooperativeMEDALSearch)):
        left.run()
        right.run()
        assert_same_tree(left, right)


@pytest.mark.parametrize('failure,status', [(KeyboardInterrupt('stop'), 'interrupted'),
                                           (torch.OutOfMemoryError('OOM'), 'oom')])
def test_medal_batch_partial_failure_preserves_forward_attempts(failure, status):
    model = BatchModel(failure=failure, fail_at=2)
    run = execute_trees(trees(model, CooperativeMEDALSearch, (9, 20)), pad_token_id=4)
    assert run.status == status
    assert run.metrics['search_nfes'] == 4 and run.metrics['total_model_invocations'] == 2
    assert all(t.root.visits == 0 for t in run.searches)
    assert all(t.nodes[1].analysis is None for t in run.searches)


def test_root_frontier_exhaustion_matches_standard_medal(monkeypatch):
    # A hand-constructed empty retained frontier; excluding every legal token
    # is instead an explicit model-analysis error, tested separately below.
    monkeypatch.setattr(medal, '_token_topk', lambda probabilities, *args: [[] for _ in probabilities])
    config = medal.MCTSConfig(init_tokens=2, max_model_calls=64, stop_mode='budget')
    prompt = torch.tensor([[3]])
    standard = medal.MCTSSearch(BatchModel(), prompt, 0, 5, config)
    cooperative = CooperativeMEDALSearch(BatchModel(), prompt, 0, 5, config)
    standard.run()
    result = execute_trees([cooperative], pad_token_id=4)
    assert result.status == 'completed'
    assert_same_tree(standard, cooperative)
    assert standard.model_calls == 1 and result.completions[0].model_calls == 5


def test_analysis_error_is_not_root_fallback():
    tree = CooperativeMEDALSearch(BatchModel(), torch.tensor([[3]]), 0, 5,
        medal.MCTSConfig(init_tokens=2, max_model_calls=64, stop_mode='budget'),
        search_excluded_token_ids=(1, 2, 3, 4, 5))
    result = execute_trees([tree], pad_token_id=4)
    assert result.status == 'error' and 'no legal token' in result.error
    assert tree.root.analysis is None and not tree.done
    assert result.metrics['search_nfes'] == 1 and result.completions == [None]
