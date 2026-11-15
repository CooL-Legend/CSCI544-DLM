"""Batched questions preserve reward B's branch-local serial search semantics."""

import pytest
import torch

from dlm_project.decoding import mcts_decoding as medal
from dlm_project.decoding.ab_batch import execute_trees
from dlm_project.decoding.ab_medal import ABMEDALConfig, ABMEDALSearch
from dlm_project.decoding.ig_kl_search import (
    CooperativeIGKLMCTSSearch, IGKLABMEDALSearch, IGKLMCTSSearch,
)
from dlm_project.decoding.medal_batch import CooperativeMEDALSearch
from tests.decoding.test_ab_batch import BatchModel


PAIRS = [(IGKLMCTSSearch, CooperativeIGKLMCTSSearch, medal.MCTSConfig),
         (IGKLABMEDALSearch, IGKLABMEDALSearch, ABMEDALConfig)]


def trees(model, cls, config_class, *, budgets=(1, 4, 9, 20), **options):
    return [cls(model, torch.tensor([[3] * (i + 1)]), 0, 5,
                config_class(init_tokens=3, num_candidates=2, top_k_tokens=2,
                             top_k_actions=2, max_model_calls=budget, stop_mode='budget'),
                seed=i + 7, search_excluded_token_ids=(4, 5),
                checkpoint_budgets=tuple(b for b in (1, 4, 9, 20) if b <= budget),
                **options) for i, budget in enumerate(budgets)]


def assert_same_search(left, right):
    assert left.stop_reason == right.stop_reason
    assert left.model_calls == right.model_calls
    assert left.result().selected.node_id == right.result().selected.node_id
    assert len(left.nodes) == len(right.nodes)
    for a, b in zip(left.nodes, right.nodes):
        assert torch.equal(a.state, b.state)
        assert (a.node_id, a.depth, a.visits, a.reward_sum, a.immediate_reward) == (
            b.node_id, b.depth, b.visits, b.reward_sum, b.immediate_reward)
    for stage in ('selection', 'evaluation', 'backpropagation', 'candidate',
                  'search_selection', 'adaptive_decision'):
        # The existing cooperative adapter omits the optional rendered child
        # state; compare its actual tensors above and all reward/policy fields.
        def events(tree):
            return [{k: v for k, v in e.items() if k != 'child_state'}
                    for e in tree.trace if e['stage'] == stage]
        assert events(left) == events(right)
    if isinstance(left, ABMEDALSearch):
        assert left.posteriors == right.posteriors
        assert left.rng.bit_generator.state == right.rng.bit_generator.state


@pytest.mark.parametrize('serial_class,cooperative_class,config_class', PAIRS)
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('top_k', [0, 2, 32])
def test_batched_ig_kl_matches_serial_tree_checkpoints_and_completion(
        serial_class, cooperative_class, config_class, batch_size, top_k):
    serial = trees(BatchModel(), serial_class, config_class, kl_top_k=top_k)
    expected_outputs = []
    for tree in serial:
        result = tree.run()
        expected_outputs.append(medal.complete_llada(
            tree.model, result.selected.state, tree.prompt_len, 0, temperature=0.).token_ids)

    model = BatchModel()
    batched = trees(model, cooperative_class, config_class, kl_top_k=top_k)
    runs = [execute_trees(batched[i:i + batch_size], pad_token_id=4)
            for i in range(0, len(batched), batch_size)]
    assert all(run.status == 'completed' for run in runs)
    outputs = [completion.token_ids for run in runs for completion in run.completions]
    for left, right, expected, actual in zip(serial, batched, expected_outputs, outputs):
        assert_same_search(left, right)
        assert torch.equal(expected, actual)
        assert left._node_rewards == right._node_rewards
        assert not right._histories and not right._historical_kl
        assert left.checkpoints.keys() == right.checkpoints.keys()
        for budget in left.checkpoints:
            a, b = left.checkpoints[budget], right.checkpoints[budget]
            for key in ('node_rewards', 'selected_state_hash', 'shortlist_node_ids',
                        'selected_q', 'reward_metadata'):
                assert a[key] == b[key]
        evaluations = [e for e in right.trace if e['stage'] == 'evaluation']
        assert right.model_calls == len(evaluations) + 1
        assert all(e['kl_extra_model_calls'] == 0 and e['kl_weight'] == .05
                   for e in evaluations)
        assert right.result().stats['kl_extra_model_calls'] == 0
    assert any(e['action_history_kl'] > 0 for tree in batched
               for e in tree.trace if e['stage'] == 'evaluation')
    assert sum(run.metrics['search_nfes'] for run in runs) == sum(t.model_calls for t in serial)
    assert sum(run.metrics['total_model_invocations'] for run in runs) == len(model.calls)
    assert sum(run.metrics['total_nfes'] for run in runs) == sum(ids.shape[0] for ids, _ in model.calls)
    if batch_size > 1:
        assert any(ids.shape[0] > 1 and mask is not None and (mask == 0).any()
                   for ids, mask in model.calls)
        assert sum(run.metrics['total_model_invocations'] for run in runs) < sum(
            run.metrics['total_nfes'] for run in runs)


@pytest.mark.parametrize('original,variant,config_class', [
    (CooperativeMEDALSearch, CooperativeIGKLMCTSSearch, medal.MCTSConfig),
    (ABMEDALSearch, IGKLABMEDALSearch, ABMEDALConfig),
])
def test_zero_lambda_keeps_original_batched_search_and_allocates_no_history(
        original, variant, config_class):
    before = trees(BatchModel(), original, config_class, budgets=(9, 20))
    after = trees(BatchModel(), variant, config_class, budgets=(9, 20), kl_weight=0)
    a = execute_trees(before, pad_token_id=4)
    b = execute_trees(after, pad_token_id=4)
    assert a.status == b.status == 'completed'
    for left, right, old_completion, new_completion in zip(before, after, a.completions, b.completions):
        assert left.model_calls == right.model_calls
        assert torch.equal(old_completion.token_ids, new_completion.token_ids)
        assert [(n.node_id, n.visits, n.reward_sum, n.immediate_reward) for n in left.nodes] == [
            (n.node_id, n.visits, n.reward_sum, n.immediate_reward) for n in right.nodes]
        assert right.peak_history_cache_bytes == 0
        assert not right._histories and not right._historical_kl


def test_cooperative_run_method_preserves_serial_ig_kl_and_cleans_failed_histories():
    serial = trees(BatchModel(), IGKLMCTSSearch, medal.MCTSConfig, budgets=(20,))[0]
    cooperative = trees(BatchModel(), CooperativeIGKLMCTSSearch,
                        medal.MCTSConfig, budgets=(20,))[0]
    serial.run()
    cooperative.run()
    assert_same_search(serial, cooperative)
    failed = trees(BatchModel(failure=RuntimeError('forward failed'), fail_at=2),
                   CooperativeIGKLMCTSSearch, medal.MCTSConfig, budgets=(20,))[0]
    with pytest.raises(RuntimeError, match='forward failed'):
        failed.run()
    assert not failed._histories and not failed._historical_kl
