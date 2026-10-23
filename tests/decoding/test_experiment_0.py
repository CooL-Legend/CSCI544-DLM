"""Behavioral tests for E0 ordering, search exclusion, budget selection and control."""

import pytest
import torch

from dlm_project.decoding import confidence, entropy, medal_no_search, mcts_confidence
from dlm_project.decoding.mcts_decoding import MCTSConfig, MCTSSearch, SearchIncompleteError, analyze_state
from tests.decoding.helpers import FakeModel, MASK, PROMPT


def settings(**changes):
    return MCTSConfig(**{"init_tokens": 2, "num_candidates": 3, "top_k_tokens": 2,
                         "top_k_actions": 3, "stop_mode": "budget", "max_model_calls": 12, **changes})


def test_confidence_and_entropy_choose_different_positions_without_extra_forwards():
    rows = [[0, .6, .15, .15, .1], [0, .55, .45, 0, 0],
            [0, .25, .25, .25, .25], [0, .25, .25, .25, .25]]
    first_positions = []
    for decoder in (confidence, entropy):
        model = FakeModel(lambda _: rows)
        result = decoder.decode(model, PROMPT, MASK, gen_length=4, steps=4, block_length=4, temperature=0)
        first_positions.append(result.trace[0]["actions"][0]["output_position"])
        assert result.total_nfes == len(model.seen) == 4
        assert all(len(event["actions"]) == 1 for event in result.trace)
        assert torch.equal(result.token_ids[:, :2], PROMPT)
        assert (result.token_ids[:, 2:] != MASK).all()
    assert first_positions == [0, 1]
    tied = entropy.decode(FakeModel(lambda _: [0, .5, .5]), PROMPT, MASK,
                          gen_length=4, steps=4, block_length=4)
    assert [e["actions"][0]["output_position"] for e in tied.trace] == [0, 1, 2, 3]
    assert tied.token_ids[0, 2:].tolist() == [1] * 4


def test_exclusions_refill_topk_without_changing_distribution_statistics():
    model = FakeModel(lambda _: [0, .45, .35, .1, .06, .04])
    state = torch.cat((PROMPT, torch.zeros((1, 4), dtype=torch.long)), dim=1)
    config = settings(top_k_tokens=3, top_k_actions=12)
    original = analyze_state(model, state, 2, MASK, config)
    filtered = analyze_state(model, state, 2, MASK, config, search_excluded_token_ids=(1, 2))
    assert torch.equal(original.entropies, filtered.entropies)
    assert original.mean_margin == filtered.mean_margin
    assert len(filtered.actions) == 12
    assert {a.token_id for a in filtered.actions} == {3, 4, 5}
    for action in filtered.actions:
        expected = {3: .1, 4: .06, 5: .04}[action.token_id]
        assert action.probability == pytest.approx(expected)
    search = MCTSSearch(model, PROMPT, MASK, 4, config,
                        search_excluded_token_ids=(1, 2))
    result = search.run()
    assert all(n.action.token_id not in (1, 2) for n in search.nodes[1:])
    assert all(a.token_id not in (1, 2) for n in search.nodes for a in n.analysis.actions)
    assert result.stats["search_excluded_token_ids"] == [1, 2]
    with pytest.raises(ValueError, match="no legal token"):
        analyze_state(FakeModel(), state, 2, MASK, config, search_excluded_token_ids=(1, 2, 3))


def test_budget_continues_beyond_three_candidates_and_has_zero_candidate_scoring_nfes():
    budget = MCTSSearch(FakeModel(), PROMPT, MASK, 4, settings()).run()
    quota = MCTSSearch(FakeModel(), PROMPT, MASK, 4, settings(stop_mode="candidates")).run()
    assert budget.stats["search_model_calls"] == 12 > quota.stats["search_model_calls"]
    assert quota.stats["stop_reason"] == "candidate_quota"
    assert budget.stats["stop_reason"] == "model_call_budget"
    assert budget.stats["candidate_evaluation_model_calls"] == 0
    expected = sorted(budget.candidates, key=lambda n: (-n.mean_reward, n.node_id))[:3]
    assert budget.selected is expected[0]
    assert [row["node_id"] for row in budget.stats["candidate_shortlist"]] == [n.node_id for n in expected]
    assert not any(e["stage"] == "rollout_step" for e in budget.trace)
    frontier = MCTSSearch(FakeModel(), PROMPT, MASK, 4, settings(max_model_calls=100)).run()
    assert frontier.stats["search_model_calls"] < 100 and frontier.stats["stop_reason"] == "frontier_exhausted"


@pytest.mark.parametrize("cap,source,count", [(1, "root", 1), (2, "deepest_evaluated", 1),
                                            (5, "deepest_evaluated", 3), (6, "deepest_evaluated", 3),
                                            (12, "target_depth", 3)])
def test_shortlists_and_fallback_are_distinct_from_quota_shortfalls(cap, source, count):
    search = MCTSSearch(FakeModel(), PROMPT, MASK, 4, settings(max_model_calls=cap))
    result = search.run()
    assert result.stats["search_model_calls"] == len(search.model.seen) == cap
    assert result.stats["selection_source"] == source
    assert result.stats["shortlist_size"] == count
    assert result.stats["shortlist_underfilled"] == (count < 3)
    assert result.stats["fallback_decode_used"] == (source != "target_depth")
    assert result.selected is search.root or result.selected.immediate_reward is not None
    if source == "deepest_evaluated":
        eligible = [n for n in search.nodes if n.immediate_reward is not None and n.depth == result.selected.depth]
        assert result.selected is min(eligible, key=lambda n: (-n.mean_reward, n.node_id))
    if cap <= 6:
        with pytest.raises(SearchIncompleteError):
            MCTSSearch(FakeModel(), PROMPT, MASK, 4, settings(max_model_calls=cap, stop_mode="candidates")).run()


def test_root_fallback_is_identical_to_matched_control_and_allows_termination():
    options = dict(gen_length=64, steps=64, block_length=64, temperature=1,
                   mcts=settings(max_model_calls=1), seed=1)
    control_model, searched_model = FakeModel(), FakeModel()
    control = medal_no_search.decode(control_model, PROMPT, MASK, **options)
    searched = mcts_confidence.decode(searched_model, PROMPT, MASK, **options, search_excluded_token_ids=(1, 2))
    assert torch.equal(control.token_ids, searched.token_ids)
    assert set(searched.token_ids[0, 2:].tolist()) >= {1, 2}
    assert (searched.token_ids[0, 2:] != MASK).all()
    assert searched.total_nfes == len(searched_model.seen) == control.total_nfes + 1 == 65
    assert searched.details["completion_seed"] == control.details["completion_seed"]
    assert MCTSConfig().init_tokens == 20 and MCTSConfig().stop_mode == "candidates"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")
def test_e0_search_exclusion_and_accounting_on_cuda():
    from tests.decoding.test_mcts_cuda import ControlledModel
    model = ControlledModel("cuda", [0, .7, .2, .1], [0, .7, .2, .1])
    result = mcts_confidence.decode(model, PROMPT.cuda(), MASK, gen_length=8, steps=8, block_length=8,
                                   mcts=settings(max_model_calls=8), search_excluded_token_ids=(1, 2))
    assert result.total_nfes == len(model.forward_devices)
    assert all(e["action"]["token_id"] == 3 for e in result.trace if e["stage"] == "expansion")
