"""MCTS-to-sampled-confidence handoff and actual forward accounting."""

from dataclasses import asdict

import pytest
import torch

from dlm_project.decoding import mcts_confidence
from dlm_project.decoding.mcts_decoding import MCTSConfig
from tests.decoding.helpers import FakeModel, FakeTokenizer, MASK, PROMPT


def options(**overrides):
    return {"init_tokens": 2, "num_candidates": 1, "top_k_tokens": 2,
            "top_k_actions": 2, "max_model_calls": 32, **overrides}


@pytest.mark.parametrize("search,depth,reason,fallback", [
    (options(), 2, "candidate_quota", False),
    (options(max_model_calls=1), 0, "model_call_budget", True),
    (options(max_model_calls=2), 1, "model_call_budget", True),
    (options(top_k_actions=1, num_candidates=3), 2, "frontier_exhausted", True),
])
def test_selected_state_is_completed_and_all_search_branches_are_charged(search, depth, reason, fallback):
    model = FakeModel()
    result = mcts_confidence.decode(model, PROMPT, MASK, gen_length=6, steps=6,
                                   block_length=3, mcts=search, tokenizer=FakeTokenizer())
    details = result.details
    assert details["selected_depth"] == depth and details["requested_depth"] == 2
    assert details["stop_reason"] == reason and details["fallback_used"] is fallback
    assert details["search_model_calls"] <= search["max_model_calls"]
    assert details["completion_model_calls"] == 6 - depth
    assert result.total_nfes == len(model.seen) == (
        details["search_model_calls"] + details["completion_model_calls"])
    assert result.scoring_nfes == 0 and details["remaining_masks"] == 0
    assert details["search_latency_seconds"] >= 0 and details["completion_latency_seconds"] >= 0
    assert details["search_state_device"] == "cpu"
    assert all(torch.equal(state[:, :2], PROMPT) for state in model.seen)
    initialization = details["selected_initialization"]
    assert len(initialization["committed"]) == depth
    for state in model.seen[details["search_model_calls"]:]:
        for item in initialization["committed"]:
            assert state[0, 2 + item["output_position"]] == item["token_id"]
    completed_positions = [action["output_position"] for event in result.trace
                           if event["phase"] == "completion" for action in event["actions"]]
    initial_positions = [item["output_position"] for item in initialization["committed"]]
    assert sorted(initial_positions + completed_positions) == list(range(6))
    evaluations = [event for event in result.trace
                   if event["phase"] == "search" and event["stage"] == "evaluation"]
    assert details["search_model_calls"] == 1 + len(evaluations)
    completion_events = [event for event in result.trace if event["phase"] == "completion"]
    assert completion_events[0]["generation_nfes"] == details["search_model_calls"] + 1
    assert completion_events[-1]["generation_nfes"] == result.total_nfes
    if fallback:
        assert details["fallback_reason"] is not None


def test_search_initialization_in_future_block_is_preserved_during_completion():
    rows = [[0., .5, .25, .25]] * 4 + [[0., .99, .005, .005], [0., .98, .01, .01]]
    model = FakeModel(lambda _: rows)
    result = mcts_confidence.decode(model, PROMPT, MASK, gen_length=6, steps=6,
                                   block_length=3, mcts=options(top_k_actions=1))
    initialized = result.details["selected_initialization"]["committed"]
    assert [item["output_position"] for item in initialized] == [4, 5]
    for state in model.seen[result.details["search_model_calls"]:]:
        assert state[0, 6:].tolist() == [1, 1]
    assert result.details["completion_model_calls"] == 4
    assert result.token_ids[0, 6:].tolist() == [1, 1]


def test_full_initialization_needs_no_completion_forward():
    model = FakeModel()
    result = mcts_confidence.decode(model, PROMPT, MASK, gen_length=4, steps=4,
                                   block_length=2, mcts=options(init_tokens=4, top_k_actions=1))
    assert result.details["selected_depth"] == 4
    assert result.details["completion_model_calls"] == 0
    assert result.total_nfes == len(model.seen) == 5  # Root + four child evaluations.
    assert not any(event["phase"] == "completion" for event in result.trace)
    assert not (result.token_ids[:, 2:] == MASK).any()


def test_completion_uses_token_probability_instead_of_legacy_adjusted_confidence():
    # Position 0 has higher token probability; position 1 has a larger margin.
    rows = [[0., .60, .39, .01], [0., .55, .225, .225]]
    result = mcts_confidence.decode(FakeModel(lambda _: rows), PROMPT, MASK,
                                   gen_length=2, steps=2, block_length=2, temperature=0,
                                   mcts=options(max_model_calls=1))
    event = next(event for event in result.trace if event["phase"] == "completion")
    assert event["actions"][0]["output_position"] == 0
    assert event["actions"][0]["selection_score"] == pytest.approx(.60)


def test_temperature_one_completion_is_seeded_and_reproducible():
    def model():
        return FakeModel(lambda _: [0., 1 / 3, 1 / 3, 1 / 3])

    kwargs = dict(gen_length=32, steps=32, block_length=16, temperature=1.0,
                  mcts=options(max_model_calls=1))
    first = mcts_confidence.decode(model(), PROMPT, MASK, seed=1, **kwargs)
    again = mcts_confidence.decode(model(), PROMPT, MASK, seed=1, **kwargs)
    other = mcts_confidence.decode(model(), PROMPT, MASK, seed=2, **kwargs)
    assert torch.equal(first.token_ids, again.token_ids)
    assert first.trace == again.trace
    assert not torch.equal(first.token_ids, other.token_ids)


def test_method_defaults_are_separate_from_legacy_mcts_defaults():
    resolved = mcts_confidence.resolve_mcts_config(gen_length=256)
    assert asdict(resolved) == {
        "init_tokens": 20, "num_candidates": 3, "top_k_tokens": 3, "top_k_actions": 5,
        "max_model_calls": 512, "margin_gamma": 10.0, "exploration_c": .01,
        "epsilon": 1e-8, "stop_mode": "candidates",
    }
    assert MCTSConfig().init_tokens == MCTSConfig().top_k_tokens == 2
    assert MCTSConfig().max_model_calls == 32
    supplied = {"max_model_calls": 1}
    assert mcts_confidence.resolve_mcts_config(supplied, gen_length=256).init_tokens == 20
    assert supplied == {"max_model_calls": 1}


@pytest.mark.parametrize("settings", [
    "invalid", {"unknown": 2}, {"init_tokens": 5}, {"top_k_tokens": True},
    {"max_model_calls": 0}, {"stop_mode": "invalid"}, {"epsilon": 0},
])
def test_invalid_search_settings_fail_before_model_forward(settings):
    model = FakeModel()
    if isinstance(settings, dict):
        settings = {**options(), **settings}
    with pytest.raises(ValueError):
        mcts_confidence.decode(model, PROMPT, MASK, gen_length=4, steps=4,
                               block_length=2, mcts=settings)
    assert model.seen == []
