"""Budget stop mode must not alter rewards or legacy quota-based experiments."""

import pytest
import torch

from dlm_project.decoding.mcts_decoding import MCTSConfig, MCTSSearch, complete_confidence
from dlm_project.experiments.paths import experiment_name
from tests.decoding.helpers import FakeModel, MASK, PROMPT


def tree(**settings):
    return MCTSSearch(FakeModel(), PROMPT, MASK, 4, MCTSConfig(**settings))


def test_budget_continues_after_quota_and_deduplicates_until_frontier_exhaustion():
    legacy = tree(num_candidates=1).run()
    budget = tree(num_candidates=1, stop_mode="budget").run()
    assert legacy.stats["stop_reason"] == "candidate_quota"
    assert budget.stats["stop_reason"] == "frontier_exhausted"
    assert budget.stats["search_model_calls"] > legacy.stats["search_model_calls"]
    assert budget.stats["candidate_count"] == 3
    assert sum(e.get("duplicate", False) for e in budget.trace) == 1
    assert not budget.stats["fallback_used"]
    assert all(node.closed and not node.children for node in budget.candidates)
    # The shared prefix, including all reward updates, is unchanged.
    assert legacy.trace[:-1] == budget.trace[:len(legacy.trace) - 1]


@pytest.mark.parametrize("budget,depth,candidates", [(1, 0, 0), (2, 1, 0), (3, 1, 0), (4, 2, 1), (5, 2, 2)])
def test_hard_cap_mid_siblings_and_fallback_only_without_target_candidate(budget, depth, candidates):
    search = tree(max_model_calls=budget, num_candidates=99, stop_mode="budget")
    result = search.run()
    stats = result.stats
    assert stats["search_model_calls"] == len(search.model.seen) == budget
    assert stats["selected_depth"] == depth
    assert stats["candidate_count"] == candidates
    assert stats["fallback_used"] == (candidates == 0)
    assert stats["evaluated_children"] == budget - 1
    evaluations = [event for event in result.trace if event["stage"] == "evaluation"]
    parents = {event["parent_id"] for event in evaluations}
    assert stats["expanded_parents"] == len(parents)
    assert stats["realized_branching_factor"] == ((budget - 1) / len(parents) if parents else 0)
    assert stats["candidate_target"] is None
    before = len(search.model.seen)
    completion = complete_confidence(search.model, result.selected.state, 2, MASK,
                                     initial_analysis=result.selected.analysis)
    assert completion.model_calls == len(search.model.seen) - before == max(0, 4 - depth - 1)
    assert completion.trace[0]["used_cached_analysis"]


def test_single_path_exhausts_before_cap_without_budget_mode_fallback():
    result = tree(stop_mode="budget", top_k_actions=1).run()
    assert result.stats["search_model_calls"] == 3
    assert result.stats["candidate_count"] == 1
    assert result.stats["stop_reason"] == "frontier_exhausted"
    assert not result.stats["fallback_used"]


def test_repeated_steps_and_run_are_equivalent_in_budget_mode():
    stepped = tree(stop_mode="budget")
    events = []
    while not stepped.done:
        events.extend(stepped.step())
    direct = tree(stop_mode="budget").run()
    assert stepped.result().stats == direct.stats
    assert events == direct.trace
    assert torch.equal(stepped.result().selected.state, direct.selected.state)


def test_names_preserve_legacy_and_distinguish_budget_mode():
    generation = dict(gen_length=4, steps=4, block_length=4, temperature=0, remasking="low_confidence")
    legacy = experiment_name(generation, {"method": "mcts_entropy"})
    explicit = experiment_name(generation, {"method": "mcts_entropy", "mcts": {"stop_mode": "candidates"}})
    budget = experiment_name(generation, {"method": "mcts_entropy", "mcts": {"stop_mode": "budget"}})
    assert legacy == explicit
    assert budget == legacy + "-stop-budget"
    with pytest.raises(ValueError, match="stop_mode"):
        MCTSConfig(stop_mode="typo")
