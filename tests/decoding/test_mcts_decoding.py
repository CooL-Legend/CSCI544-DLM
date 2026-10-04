"""Analytic CPU checks: no weights, Hub calls, GPU or Sudoku answers required."""

import json
import math

import pytest
import torch

from dlm_project.decoding.mcts_decoding import (
    Action, MCTSConfig, MCTSSearch, Node, analyze_state, backpropagate,
    complete_confidence, evaluate_action, expand_node, information_gain, select_node,
)

from tests.decoding.helpers import MASK, PROMPT, FakeModel, FakeTokenizer


def search(model=None, **settings):
    return MCTSSearch(model or FakeModel(), PROMPT, MASK, 4, MCTSConfig(**settings),
                      tokenizer=FakeTokenizer())


def entropy(probabilities):
    return -sum(p * math.log(p) for p in probabilities if p)


def test_confidence_filter_matches_independent_calculation_and_excludes_prompt_and_mask():
    rows = [[0.5, 0.2, 0.2, 0.1], [0.0, 0.7, 0.2, 0.1], [0.25] * 4]
    model = FakeModel(lambda output: rows)
    config = MCTSConfig(top_k_tokens=2, top_k_actions=5)
    state = torch.cat([PROMPT, torch.zeros(1, 3, dtype=torch.long)], dim=1)
    analysis = analyze_state(model, state, 2, MASK, config)
    expected = []
    for position, row in enumerate(rows):
        h = entropy(row)
        top = sorted(row, reverse=True)
        factor = math.exp(-h) / (1 + math.exp(-10 * (top[0] - top[1])))
        for token in sorted(range(1, 4), key=lambda t: (-row[t], t))[:2]:
            expected.append((row[token] * factor, position, token))
    expected.sort(key=lambda item: (-item[0], item[1], item[2]))
    assert [(a.output_position, a.token_id) for a in analysis.actions] == [(p, t) for _, p, t in expected[:5]]
    assert [a.confidence_score for a in analysis.actions] == pytest.approx([c for c, _, _ in expected[:5]])
    assert analysis.entropies.tolist() == pytest.approx([entropy(row) for row in rows])
    assert len(model.seen) == 1
    assert analysis.entropies.device.type == "cpu"
    assert set(vars(analysis)) == {"entropies", "actions", "state_key"}


def test_boundary_ties_choose_smallest_token_then_position_ids():
    model = FakeModel(lambda output: [0.25] * 4)
    config = MCTSConfig(top_k_tokens=1, top_k_actions=3)
    state = torch.tensor([[3, 0, 0, 0, 0]])
    analysis = analyze_state(model, state, 2, MASK, config)
    assert [(a.output_position, a.token_id) for a in analysis.actions] == [(0, 1), (1, 1), (2, 1)]


@pytest.mark.parametrize("root_row,child_row", [
    ([0, .7, .2, .1], [0, .7, .2, .1]),
    ([0, .99, .005, .005], [0, 1/3, 1/3, 1/3]),  # Remaining masks become less certain.
    ([0, 1, 0, 0], [0, 1/3, 1/3, 1/3]),  # Zero denominator, finite reward.
])
def test_one_token_evaluation_entropy_reward_and_context_gain(root_row, child_row):
    model = FakeModel(lambda output: root_row if (output == MASK).all() else child_row)
    tree = search(model)
    events = tree.step()
    evaluations = [e for e in events if e["stage"] == "evaluation"]
    before, after, removed = 4 * entropy(root_row), 3 * entropy(child_row), entropy(root_row)
    expected = (before - after) / before if before > 1e-8 else 0.0
    assert len(evaluations) == 2
    assert len(model.seen) == tree.model_calls == 3
    for event in evaluations:
        assert event["entropy_before"] == pytest.approx(before)
        assert event["entropy_after"] == pytest.approx(after)
        assert event["committed_position_entropy"] == pytest.approx(removed)
        assert event["reward"] == pytest.approx(expected, abs=1e-6, rel=2e-6)
        assert event["context_gain"] == pytest.approx(before - removed - after, abs=1e-6)
        assert len(event["child_state"]["committed"]) == 1
    if root_row[1] == .99:
        assert evaluations[0]["reward"] < 0
        assert tree.root.reward_sum < 0


def test_effectively_zero_entropy_guard_and_negative_gain():
    assert information_gain(5e-9, 4, 1e-8) == 0.0
    assert information_gain(2, 3, 1e-8) == -0.5


def test_ucb_prefers_exploration_then_unvisited_with_stable_ties():
    root = Node(0, torch.zeros(1, 6, dtype=torch.long), visits=20, expanded=True)
    a = Node(1, root.state.clone(), root, Action(0, 1, .8), visits=10, reward_sum=6)
    b = Node(2, root.state.clone(), root, Action(1, 1, .7), visits=1, reward_sum=.4)
    closed = Node(3, root.state.clone(), root, Action(2, 1, .9), visits=1, reward_sum=10, closed=True)
    root.children = [a, b, closed]
    selected, decisions = select_node(root, .2)
    assert selected is b
    score = decisions[0]["siblings"][1]
    assert score["exploration_bonus"] == pytest.approx(.2 * math.sqrt(math.log(20)))
    assert score["ucb"] == pytest.approx(.4 + score["exploration_bonus"])
    assert decisions[0]["siblings"][2]["eligible"] is False
    a.visits = b.visits = 0
    selected, decisions = select_node(root, .2)
    assert selected is a  # Both unvisited: highest action confidence first.
    assert decisions[0]["siblings"][0]["ucb"] is None
    json.dumps(decisions, allow_nan=False)
    a.action = Action(2, 1, .7)
    assert select_node(root, .2)[0] is b  # Tied confidence: smaller position.


def test_backpropagation_exact_child_to_root_updates():
    root = Node(0, PROMPT, visits=2, reward_sum=.5)
    parent = Node(1, PROMPT, root, visits=1, reward_sum=.2)
    child = Node(2, PROMPT, parent)
    updates = backpropagate(child, -.3)
    assert [u["node_id"] for u in updates] == [2, 1, 0]
    assert updates[0]["before"] == {"visits": 0, "reward_sum": 0., "mean_reward": 0.}
    for update, visits, total in zip(updates, [1, 2, 3], [-.3, -.1, .2]):
        assert update["after"]["visits"] == visits
        assert update["after"]["reward_sum"] == pytest.approx(total)
        assert update["after"]["mean_reward"] == pytest.approx(total / visits)


def test_independent_siblings_fixed_prompt_and_cached_completion():
    model = FakeModel()
    tree = search(model)
    tree.step()
    left, right = tree.root.children
    assert left.state.data_ptr() != right.state.data_ptr() != tree.root.state.data_ptr()
    root_copy, right_copy = tree.root.state.clone(), right.state.clone()
    with torch.inference_mode():
        left.state[0, -1] = 3
    assert torch.equal(tree.root.state, root_copy)
    assert torch.equal(right.state, right_copy)
    # Reject stale statistics rather than silently completing a different state.
    with pytest.raises(ValueError, match="different partial state"):
        complete_confidence(model, left.state, 2, MASK, initial_analysis=left.analysis)
    calls_before = len(model.seen)
    completion = complete_confidence(model, right.state, 2, MASK, initial_analysis=right.analysis)
    assert completion.model_calls == len(model.seen) - calls_before == 2
    assert completion.trace[0]["used_cached_analysis"] is True
    assert torch.equal(completion.token_ids[:, :2], PROMPT)
    committed = right_copy != MASK
    assert torch.equal(completion.token_ids[committed], right_copy[committed])
    assert torch.equal(right.state, right_copy)
    for state in model.seen[calls_before:]:
        assert torch.equal(state[:, :2], PROMPT)
        assert torch.equal(state[committed], right_copy[committed])
    assert (completion.token_ids[:, 2:] != MASK).all()


def test_candidates_deduplicate_across_orders_and_target_nodes_close():
    tree = search(num_candidates=99)
    result = tree.run()
    assert result.stats["stop_reason"] == "frontier_exhausted"
    assert result.stats["candidate_count"] == 3
    assert sum(e.get("duplicate", False) for e in result.trace) == 1
    assert len({tuple(n.state[0].tolist()) for n in result.candidates}) == 3
    assert all(n.closed and not n.children and not n.expanded for n in result.candidates)
    assert result.stats["fallback_used"]
    assert result.stats["selected_depth"] == 2
    assert tree.step() == []


def test_single_action_branch_exhausts_without_reexpanding_target():
    tree = search(top_k_actions=1)
    result = tree.run()
    assert result.stats["stop_reason"] == "frontier_exhausted"
    assert result.stats["candidate_count"] == 1
    assert result.stats["search_model_calls"] == 3


@pytest.mark.parametrize("budget,depth,candidates", [(1, 0, 0), (2, 1, 0), (3, 1, 0), (4, 2, 1), (5, 2, 2)])
def test_budget_is_hard_cap_including_root_and_partial_sibling_sets(budget, depth, candidates):
    tree = search(max_model_calls=budget)
    result = tree.run()
    assert tree.model_calls == len(tree.model.seen) == budget
    assert result.stats["stop_reason"] == "model_call_budget"
    assert result.stats["selected_depth"] == depth
    assert result.stats["candidate_count"] == candidates
    assert result.stats["fallback_used"]
    assert result.stats["fallback_reason"]
    assert result.trace[-1]["stage"] == "termination"
    assert len(result.trace[-1]["selected_state"]["committed"]) == depth


def test_final_ranking_uses_root_reduction_not_last_edge_reward():
    # Branch with token at position 0 has a large first gain, tiny second gain.
    # The other branch has a worse final state despite a better last-edge gain.
    def policy(output):
        if (output == MASK).all():
            row = [0, .7, .2, .1]
        elif output[0] != MASK:
            row = [0, .99, .005, .005] if (output != MASK).sum() == 1 else [0, .97, .02, .01]
        else:
            row = [0, .5, .25, .25]
        return row
    tree = search(FakeModel(policy), num_candidates=99)
    result = tree.run()
    gains = {e["child_id"]: e["reward"] for e in result.trace if e["stage"] == "evaluation"}
    assert result.selected.state[0, 2] != MASK
    assert gains[result.selected.node_id] < 0  # Still the best root-relative candidate.
    assert result.stats["selected_score"] > 0
    scores = [information_gain(tree.root.analysis.total_entropy, n.analysis.total_entropy, 1e-8)
              for n in result.candidates]
    assert result.stats["selected_score"] == max(scores)


def test_cycle_order_trace_backups_call_accounting_and_step_run_equivalence():
    streamed = []
    tree = search()
    tree.on_event = streamed.append
    first = tree.step()
    assert [e["stage"] for e in first] == ["initialization", "selection", "expansion", "evaluation",
                                          "backpropagation", "expansion", "evaluation", "backpropagation"]
    assert [e["search_model_calls"] for e in first if e["stage"] == "evaluation"] == [2, 3]
    assert first[2]["action"]["decoded_token"] == " one"
    result = tree.run()
    pure_run = search().run()
    stepped = search()
    while not stepped.done:
        stepped.step()
    assert result.trace == streamed == pure_run.trace == stepped.result().trace
    assert result.stats == pure_run.stats
    assert torch.equal(result.selected.state, pure_run.selected.state)
    assert tree.root.visits == tree.model_calls - 1
    assert sum(torch.equal(state, tree.root.state) for state in tree.model.seen) == 1
    for event in result.trace:
        json.dumps(event, allow_nan=False)
        if event["stage"] == "backpropagation":
            for update in event["updates"]:
                assert update["after"]["visits"] == update["before"]["visits"] + 1
                assert update["after"]["reward_sum"] == pytest.approx(update["before"]["reward_sum"] + event["reward"])
    assert tree.model_calls == 1 + sum(e["stage"] == "evaluation" for e in result.trace)


def test_completed_state_needs_no_model_call_and_illegal_commitments_fail():
    model = FakeModel()
    state = torch.tensor([[3, MASK, 1, 2, 3, 1]])
    analysis = analyze_state(model, state, 2, MASK, MCTSConfig())
    assert analysis.total_entropy == 0 and analysis.actions == ()
    assert complete_confidence(model, state, 2, MASK).model_calls == 0
    assert model.seen == []
    parent = Node(0, state)
    for action in [Action(-1, 1, .5), Action(4, 1, .5), Action(0, MASK, .5), Action(0, 2, .5)]:
        with pytest.raises(ValueError):
            expand_node(parent, action, 1, 2, MASK, 2)


def test_search_can_reach_full_length_with_one_forward_per_evaluated_child():
    tree = search(init_tokens=4, top_k_actions=1, num_candidates=1)
    result = tree.run()
    assert result.stats["selected_depth"] == 4
    assert result.stats["stop_reason"] == "candidate_quota"
    assert tree.model_calls == len(tree.model.seen) == 5
    assert result.selected.analysis.total_entropy == 0
    completion = complete_confidence(tree.model, result.selected.state, 2, MASK,
                                     initial_analysis=result.selected.analysis)
    assert completion.model_calls == 0 and completion.trace == []


@pytest.mark.parametrize("settings", [{"init_tokens": 0}, {"top_k_tokens": True},
                                      {"max_model_calls": 0}, {"exploration_c": -1},
                                      {"epsilon": 0}, {"margin_gamma": float("nan")}])
def test_invalid_search_config(settings):
    with pytest.raises(ValueError):
        MCTSConfig(**settings)


def test_model_nonfinite_probabilities_rejected():
    with pytest.raises(ValueError, match="Non-finite"):
        search(FakeModel(lambda output: [float("nan")] * 4)).step()
