"""Fixed-budget AB, shared MEDAL semantics, and offline allocation oracles."""

from dataclasses import asdict
import copy
import json

import pytest
import torch

from dlm_project.decoding.ab_medal import ABMEDALConfig, ABMEDALSearch
from dlm_project.decoding import mcts_decoding as md, search_audit as audit
from dlm_project.decoding.ab_medal_demo import run_completion
from dlm_project.decoding.medal_demo import InitializationResult
from tests.decoding.helpers import FakeModel, MASK, PROMPT


def make(budget=16, **options):
    config = ABMEDALConfig(**dict(init_tokens=2, max_model_calls=budget, **options))
    model = FakeModel()
    tree = ABMEDALSearch(model, PROMPT, MASK, 4, config, seed=7)
    return tree


def saved(tree):
    return audit.from_events(tree.trace, prompt_ids=PROMPT[0].tolist(), mask_id=MASK,
        gen_length=4, requested_depth=tree.config.init_tokens,
        requested_budget=tree.config.max_model_calls, search_nfes=tree.model_calls, adaptive=True)


@pytest.mark.parametrize('budget', [1, 2, 3, 4, 16, 64])
def test_budget_fallback_completes_once_and_offline_counts_reconcile(budget):
    tree = make(budget)
    result = tree.run()
    assert tree.model_calls <= budget
    assert result.stats['stop_reason'] in ('model_call_budget', 'frontier_exhausted')
    assert result.stats['fallback_decode_used'] == (result.selected.depth < 2)
    before_rng = json.dumps(tree.rng.bit_generator.state)
    record = saved(tree)
    states = audit.reconstruct(record, PROMPT[0].tolist())
    metrics = audit.metrics(record)
    assert states[result.selected.node_id] == result.selected.state[0].tolist()
    assert metrics['search_nfes'] == sum(metrics['nfes_per_depth'].values()) == tree.model_calls
    assert metrics['root_analysis_nfes'] + sum(metrics['nfes_per_root_subtree'].values()) == tree.model_calls
    assert len(record['nodes']) == tree.model_calls
    assert json.dumps(tree.rng.bit_generator.state) == before_rng
    assert len(tree.model.seen) == tree.model_calls
    decoded = run_completion(InitializationResult(tree, result, tree.initialization_seconds), seed=7)
    assert decoded.method == 'llada'
    assert decoded.model_calls == 4 - result.selected.depth
    fixed = result.selected.state != MASK
    fixed[:, :PROMPT.shape[1]] = True
    assert torch.equal(decoded.token_ids[fixed], result.selected.state[fixed])
    assert len(tree.model.seen) == tree.model_calls + decoded.model_calls
    if budget == 1:
        assert result.stats['shortlist_size'] == 1 and result.selected.node_id == 0
        assert metrics['ig_mean'] is metrics['negative_ig_percentage'] is None
        assert metrics['branching_factor_min'] == metrics['branching_factor_max'] == 0


def test_quota_does_not_stop_budget_search_and_first_occurrences_are_retained():
    tree = make(64)
    result = tree.run()
    assert len(result.candidates) > 3
    assert result.stats['shortlist_size'] == 3
    assert result.stats['stop_reason'] == 'frontier_exhausted'
    record = saved(tree)
    metrics = audit.metrics(record)
    assert metrics['duplicate_state_count'] > 0
    for candidate in result.candidates:
        same = [n for n in tree.nodes if torch.equal(n.state, candidate.state)]
        assert candidate.node_id == min(n.node_id for n in same)
    assert sum(metrics['decision_counts'].values()) >= tree.simulations_completed
    assert metrics['decision_counts']['GEN'] == tree.simulations_completed


@pytest.mark.parametrize('budget', [1, 2, 4, 9, 20])
def test_fixed_budget_equivalent_policy_matches_medal(budget):
    adaptive = make(budget, search_policy='medal_equivalent')
    settings = {name: getattr(adaptive.config, name) for name in md.MCTSConfig.__dataclass_fields__}
    ordinary = md.MCTSSearch(FakeModel(), PROMPT, MASK, 4, md.MCTSConfig(**settings), seed=7)
    left, right = ordinary.run(), adaptive.run()
    assert left.stats['stop_reason'] == right.stats['stop_reason']
    assert ordinary.model_calls == adaptive.model_calls
    assert left.selected.node_id == right.selected.node_id
    for a, b in zip(ordinary.nodes, adaptive.nodes):
        assert torch.equal(a.state, b.state)
        assert (a.visits, a.reward_sum, a.immediate_reward) == (b.visits, b.reward_sum, b.immediate_reward)


def test_search_exclusions_are_shared_and_completion_still_allows_tokens():
    tree = ABMEDALSearch(FakeModel(), PROMPT, MASK, 4,
        ABMEDALConfig(init_tokens=2, max_model_calls=8), search_excluded_token_ids=(1, 2))
    result = tree.run()
    assert all(n.action.token_id == 3 for n in tree.nodes[1:])
    output = run_completion(InitializationResult(tree, result, 0)).token_ids
    assert 1 in output[0, PROMPT.shape[1]:].tolist()


def test_log_hash_order_and_statistics_tampering_are_rejected():
    tree = make()
    tree.run()
    record = saved(tree)
    for field, value in [('state_hash', 'altered'), ('evaluation_order', 9), ('visits', 99)]:
        bad = copy.deepcopy(record)
        bad['nodes'][1][field] = value
        with pytest.raises(ValueError):
            audit.reconstruct(bad, PROMPT[0].tolist())


def test_hand_checkable_rewards_subtrees_and_duplicate_counts(monkeypatch):
    # Three root children, one evaluation each; entropy increases for token 2.
    def analyze(model, state, prompt_len, mask_id, config, **kwargs):
        model(state)
        output = state[0, prompt_len:]
        token = int(output[0])
        entropy = {0: 4., 1: 3., 2: 6., 3: 2.}[token]
        actions = tuple(md.Action(0, t, 1 / t) for t in (1, 2, 3)) if not token else ()
        return md.StateAnalysis(torch.tensor([entropy, 0., 0., 0.]), actions, tuple(state[0].tolist()))
    monkeypatch.setattr(md, 'analyze_state', analyze)
    tree = ABMEDALSearch(FakeModel(), PROMPT, MASK, 4,
        ABMEDALConfig(init_tokens=1, max_model_calls=20))
    tree.run()
    m = audit.metrics(saved(tree))
    assert m['nfes_per_depth'] == {'0': 1, '1': 3}
    assert m['nfes_per_root_subtree'] == {'1': 1, '2': 1, '3': 1}
    assert m['negative_ig_count'] == 1 and m['negative_ig_percentage'] == pytest.approx(100/3)
    assert m['ig_min'] == -.5 and m['ig_max'] == .5 and m['ig_mean'] == pytest.approx(1/12)
    assert (m['branching_factor_min'], m['branching_factor_mean'], m['branching_factor_max']) == (3, 3, 3)
    assert m['root_subtree_coverage'] == 1 and m['selected_q'] == .5


def test_legacy_recovery_does_not_invent_hashes():
    tree = make()
    tree.run()
    old = [e for e in tree.trace if e['stage'] not in ('node_evaluated', 'search_selection', 'adaptive_decision')]
    record = audit.from_events(old, mask_id=MASK, gen_length=4, requested_depth=2,
        requested_budget=16, search_nfes=tree.model_calls, allow_legacy=True)
    m = audit.metrics(record)
    assert not m['state_hashes_available'] and m['decision_counts'] is None
    assert m['search_nfes'] == sum(m['nfes_per_depth'].values())
