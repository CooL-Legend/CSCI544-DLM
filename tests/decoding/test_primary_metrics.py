"""Simulation counts differ from forwards, cycles, and final decoding steps."""

import pytest
import torch
import math

from dlm_project.decoding.confidence import decode
from dlm_project.decoding.mcts_decoding import MCTSConfig, MCTSSearch
from dlm_project.decoding.signals import masked_state_signals
from tests.decoding.helpers import FakeModel, MASK, PROMPT


@pytest.mark.parametrize("budget", [1, 2, 3, 4])
def test_simulation_cap_stops_between_siblings_and_records_evaluated_children(budget):
    model = FakeModel()
    search = MCTSSearch(model, PROMPT, MASK, 4, MCTSConfig(init_tokens=2,
        num_candidates=10, top_k_actions=3, top_k_tokens=2,
        max_simulations=budget, stop_mode="budget"))
    result = search.run()
    assert result.stats["stop_reason"] == "simulation_budget"
    assert result.stats["mcts_simulations"] == result.stats["simulations_started"] == budget
    assert result.stats["simulations_per_invocation"] == [budget]
    assert result.stats["simulations_interrupted"] == 0
    assert search.root.visits == budget
    assert sum(event["stage"] == "backpropagation" for event in result.trace) == budget
    assert result.stats["search_model_calls"] == len(model.seen) == 1 + budget
    assert result.stats["simulation_decoding_steps"] == 0
    assert result.stats["average_search_depth"] == pytest.approx(sum(node.depth for node in search.nodes if node.immediate_reward is not None) / budget)


def test_last_forward_at_cap_completes_one_immediate_child_simulation():
    result = MCTSSearch(FakeModel(), PROMPT, MASK, 4, MCTSConfig(init_tokens=2,
        max_model_calls=2, max_simulations=4, stop_mode="budget")).run()
    assert result.stats["search_model_calls"] == 2
    assert result.stats["simulations_started"] == result.stats["mcts_simulations"] == 1
    assert result.stats["simulations_interrupted"] == 0


def test_signals_include_future_blocks_without_additional_forwards():
    model = FakeModel(lambda _: [[.1, .6, .2, .1], [.1, .6, .2, .1],
                                 [.25, .25, .25, .25], [.25, .25, .25, .25]])
    result = decode(model, PROMPT, MASK, gen_length=4, steps=2, block_length=2, temperature=0)
    first = result.trace[0]
    expected_entropy = (-sum(p * math.log(p + 1e-8) for p in (.1, .6, .2, .1))
                        - sum(p * math.log(p + 1e-8) for p in (.25, .25, .25, .25))) / 2
    assert first["masked_count"] == 4
    assert first["mean_token_entropy"] == pytest.approx(expected_entropy)
    assert first["mean_confidence_margin"] == pytest.approx(.2)
    assert result.generation_nfes == result.details["decoding_steps"] == 2
    assert len(model.seen) == 2
    assert all(event["signal_extra_nfes"] == 0 for event in result.trace)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_simulation_cap(value):
    with pytest.raises(ValueError, match="max_simulations"):
        MCTSConfig(max_simulations=value)
