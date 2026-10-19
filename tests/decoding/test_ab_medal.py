"""Adaptive traversal, exact posterior recipients and full-depth finite trees."""

from dataclasses import asdict
import json
import math

import numpy as np
import pytest
import torch

from dlm_project.decoding import mcts_decoding as md
from dlm_project.decoding.ab_medal import (
    ABMEDALConfig, ABMEDALSearch, GaussianPosterior, tree_record, tree_statistics,
)
from tests.decoding.helpers import FakeModel, FakeTokenizer, MASK, PROMPT


def search(**settings):
    config = ABMEDALConfig(**{"init_tokens": 3, "num_candidates": 2,
                              "top_k_tokens": 2, "top_k_actions": 2, **settings})
    return ABMEDALSearch(FakeModel(), PROMPT, MASK, 4, config, seed=1)


def test_adaptive_search_inherits_terminal_token_filtering():
    class TerminalTokenizer(FakeTokenizer):
        def get_vocab(self):
            return {"[MASK]": 0, "<|endoftext|>": 1, "<|eot_id|>": 2, "3": 3}

    config = ABMEDALConfig(init_tokens=1, num_candidates=1,
                           top_k_tokens=3, top_k_actions=1)
    result = ABMEDALSearch(
        FakeModel(), PROMPT, MASK, 4, config,
        tokenizer=TerminalTokenizer(), seed=1,
    ).run()
    committed = result.selected.state[0, PROMPT.shape[1]:].tolist()
    assert [token for token in committed if token != MASK] == [3]
    assert result.stats["search_excluded_token_ids"] == [1, 2]


def finite_tree(monkeypatch, kind, **settings):
    """Only action/entropy fixtures are synthetic; traversal and backups are real.

    Every analysis actually forwards through the counted fake model. One extra
    output mask ensures that depth-20 children also require a forward.
    """
    def analyze(model, state, prompt_len, mask_id, config, **kwargs):
        model(state)
        output = state[0, prompt_len:]
        depth = int((output != mask_id).sum())
        actions = []
        if depth < 20:
            if kind == "shared":
                tokens = [1] if depth < 19 else [1, 2, 3]
            else:
                tokens = [1, 2, 3] if depth == 0 else [int(output[0])]
            actions = [md.Action(depth, token, 1 / token) for token in tokens]
        return md.StateAnalysis((output == mask_id).float(), tuple(actions), tuple(state[0].tolist()))

    monkeypatch.setattr(md, "analyze_state", analyze)
    config = ABMEDALConfig(**{"init_tokens": 20, "num_candidates": 3, **settings})
    return ABMEDALSearch(FakeModel(), PROMPT, MASK, 21, config, seed=1)


def test_old_parameter_update_with_negative_reward():
    posterior = GaussianPosterior(.5, 2, 3, .2)
    record = posterior.update(-.4)
    assert record["before"] == {"m": .5, "kappa": 2, "nu": 3, "tau2": .2}
    assert record["after"] == asdict(posterior)
    assert posterior.m == pytest.approx((2 * .5 - .4) / 3)
    assert posterior.kappa == 3 and posterior.nu == 4
    assert posterior.tau2 == pytest.approx((3 * .2 + 2 / 3 * (-.4 - .5) ** 2) / 4)
    assert record["reward"] == -.4


def test_predictive_location_scale_df_and_no_normal_switch():
    class Draws:
        def __init__(self):
            self.df = []

        def standard_t(self, nu):
            self.df.append(nu)
            return 2.25

    rng = Draws()
    posterior = GaussianPosterior(.5, 2, 3, .2)
    assert posterior.sample(rng) == pytest.approx(.5 + math.sqrt(.2 * 1.5) * 2.25)
    for _ in range(40):
        posterior.update(-.4)
    posterior.sample(rng)
    assert rng.df == [3, 43]


@pytest.mark.parametrize("settings", [dict(kappa=0), dict(nu=-1), dict(tau2=0),
                                         dict(m=float("nan")), dict(tau2=float("inf")), dict(nu=True)])
def test_invalid_priors(settings):
    with pytest.raises(ValueError):
        GaussianPosterior(**settings)


def test_nonfinite_updates_and_samples_leave_parameters_intact():
    posterior = GaussianPosterior()
    before = asdict(posterior)
    for reward in (float("inf"), float("nan"), 1e308):
        with pytest.raises(ValueError):
            posterior.update(reward)
        assert asdict(posterior) == before
    with pytest.raises(ValueError):
        posterior.sample(type("BadDraw", (), {"standard_t": lambda self, nu: float("inf")})())


def test_posteriors_are_independent_and_seed_is_isolated():
    first, second = search(), search()
    first.step()
    second.step()
    objects = [p for row in first.posteriors.values() for p in row.values()]
    assert len({id(p) for p in objects}) == len(objects)
    assert asdict(first.posteriors[0]["node"]) == asdict(GaussianPosterior())
    np.random.seed(103)
    first.step()
    np.random.seed(999)
    second.step()
    assert first.trace == second.trace
    assert first.history == second.history


@pytest.mark.parametrize("choice", ["GEN", "CONT"])
def test_explicit_gen_cont_outcomes_and_exact_update_recipients(monkeypatch, choice):
    tree = search()
    tree.step()  # Forced GEN at the root.
    assert tree.history[0]["decisions"] == [{"node_id": 0, "choice": "GEN", "forced": True,
                                             "gen_available": True, "cont_available": False, "scores": {}}]
    before = {node_id: {name: asdict(p) for name, p in row.items()}
              for node_id, row in tree.posteriors.items()}
    root_gen = tree.posteriors[0]["GEN"]
    root_cont = tree.posteriors[0]["CONT"]
    def sample(posterior, rng):
        if posterior is root_gen:
            return 2 if choice == "GEN" else -2
        if posterior is root_cont:
            return 1
        return 0
    monkeypatch.setattr(GaussianPosterior, "sample", sample)
    tree.step()
    record = tree.history[-1]
    assert record["decisions"][0]["choice"] == choice
    recipients = [(r["node_id"], r["model"]) for r in record["posterior_updates"]]
    expected = [(2, "node"), (0, "GEN")] if choice == "GEN" else [
        (2, "node"), (1, "GEN"), (1, "node"), (0, "CONT")]
    assert recipients == expected
    assert len(recipients) == len(set(recipients))
    for node_id, models in before.items():
        for name, params in models.items():
            if (node_id, name) not in expected:
                assert asdict(tree.posteriors[node_id][name]) == params
    for update in record["posterior_updates"]:
        assert update["after"]["kappa"] == update["before"]["kappa"] + 1
        assert update["reward"] == record["reward"]
    assert tree.root.visits == 2  # Ordinary backups happen only once per child.
    assert tree.nodes[1].visits == (2 if choice == "CONT" else 1)
    assert tree.nodes[2].visits == 1
    assert len(tree.model.seen) == 3


def test_ties_choose_cont_and_smallest_child_id(monkeypatch):
    tree = search(top_k_actions=3)
    tree.step()
    root_gen = tree.posteriors[0]["GEN"]
    monkeypatch.setattr(GaussianPosterior, "sample", lambda p, rng: 2 if p is root_gen else 0)
    tree.step()  # A second eligible root child.
    monkeypatch.setattr(GaussianPosterior, "sample", lambda p, rng: 0)
    tree.step()
    decision = tree.history[-1]["decisions"][0]
    assert decision["choice"] == "CONT" and decision["chosen_id"] == 1


@pytest.mark.parametrize("kind,nfe,children,prefix,distance,parents,mean_depth", [
    ("shared", 23, 22, 19, 2 / 21, 20, 250 / 22),
    ("separated", 61, 60, 0, 1, 58, 10.5),
])
def test_full_depth_acceptance_fixtures(monkeypatch, kind, nfe, children, prefix, distance, parents, mean_depth):
    tree = finite_tree(monkeypatch, kind)
    result = tree.run()  # Genuine seeded Student-t draws, no patched sampling.
    stats = result.stats
    assert len(result.candidates) == 3
    assert all(n.depth == 20 and n.closed for n in result.candidates)
    assert len({md._state_key(n.state) for n in result.candidates}) == 3
    assert stats["stop_reason"] == "candidate_quota" and stats["target_reached"]
    assert stats["search_model_calls"] == stats["search_sequence_nfes"] == len(tree.model.seen) == nfe
    assert stats["tree_analysis_model_calls"] == stats["tree_analysis_sequence_nfes"] == 1
    assert stats["simulation_model_calls"] == stats["simulation_sequence_nfes"] == children
    assert stats["candidate_evaluation_model_calls"] == 0
    assert stats["evaluated_children"] == stats["iterations"] == tree.root.visits == children
    assert stats["expanded_parents"] == parents
    assert stats["unique_states"] == children + 1
    assert stats["mean_evaluated_depth"] == pytest.approx(mean_depth)
    assert stats["max_evaluated_depth"] == 20
    assert stats["mean_branching_factor"] == pytest.approx(children / parents)
    assert stats["branching_factor_distribution"] == {1: parents - 1, 3: 1}
    assert sum(stats["depth_histogram"].values()) == children
    assert sum(stats["branching_factors"].values()) == children
    assert [r["depth"] for r in stats["branch_depths"]] == [20, 20, 20]
    diversity = stats["candidate_diversity"]
    assert diversity["available"] and diversity["shared_prefix_length"] == prefix
    assert len(diversity["pairs"]) == 3
    assert all(p["shared_prefix_length"] == prefix for p in diversity["pairs"])
    assert all(p["jaccard_distance"] == pytest.approx(distance) for p in diversity["pairs"])
    assert diversity["mean_jaccard_distance"] == pytest.approx(distance)
    assert len(tree.history) == children
    for record in tree.history:
        assert record["sequence_nfes"] == record["model_calls"] == record["iteration"] + 1
        assert record["audit"]["evaluated_children"] == record["iteration"]
        expected = {(record["child_id"], "node"), (record["parent_id"], "GEN")}
        expected.update((node_id, "node") for node_id in record["path"][1:])
        expected.update((node_id, "CONT") for node_id in record["path"][:-1])
        recipients = [(u["node_id"], u["model"]) for u in record["posterior_updates"]]
        assert set(recipients) == expected and len(recipients) == len(expected)
    assert all(value >= 0 for value in stats["timing_seconds"].values())
    assert sum(stats["timing_seconds"].values()) == pytest.approx(stats["initialization_wall_seconds"])
    assert stats["timing_seconds"]["full_rollout"] == 0


def test_forced_cont_exhaustion_and_incomplete_diversity(monkeypatch):
    tree = finite_tree(monkeypatch, "shared", num_candidates=4)
    with pytest.raises(md.SearchIncompleteError):
        tree.run()
    assert tree.stop_reason == "frontier_exhausted" and tree.root.closed
    assert len(tree.candidates) == 3 and len(tree.model.seen) == 23
    decisions = [d for record in tree.history for d in record["decisions"]]
    assert any(d["choice"] == "CONT" and d["forced"] for d in decisions)
    assert all(not tree.remaining_actions[n.node_id] and all(c.closed for c in n.children)
               for n in tree.nodes if n.closed and n.depth < 20)
    one = search(num_candidates=1)
    one.run()
    assert one.result().stats["candidate_diversity"] == {
        "available": False, "pairs": [], "shared_prefix_length": None, "mean_jaccard_distance": None}


@pytest.mark.parametrize("cap,complete", [(22, False), (23, True)])
def test_quota_at_cap_and_no_over_budget_forward(monkeypatch, cap, complete):
    tree = finite_tree(monkeypatch, "shared", max_model_calls=cap)
    if complete:
        tree.run()
    else:
        with pytest.raises(md.SearchIncompleteError):
            tree.run()
    stats = tree.result().stats
    assert stats["target_reached"] is complete
    assert stats["stop_reason"] == ("candidate_quota" if complete else "model_call_budget")
    assert tree.model_calls == len(tree.model.seen) == cap
    assert tree.root.visits == cap - 1
    assert tree.step() == []


def test_duplicate_candidate_states_are_evaluated_but_not_counted_twice():
    # Different action orders reach the same two-commitment state.
    tree = search(init_tokens=2, num_candidates=100, top_k_actions=2)
    with pytest.raises(md.SearchIncompleteError):
        tree.run()
    candidates = [e for e in tree.trace if e["stage"] == "candidate"]
    assert any(e["duplicate"] for e in candidates)
    assert len(tree.candidates) < len(candidates)
    assert tree.root.visits == tree.simulations_completed


def test_negative_information_gain_is_backed_up_without_clipping(monkeypatch):
    def analyze(model, state, prompt_len, mask_id, config, **kwargs):
        model(state)
        depth = int((state[0, prompt_len:] != mask_id).sum())
        actions = (md.Action(depth, 1, 1),)
        return md.StateAnalysis(torch.tensor([float(depth + 1)]), actions, tuple(state[0].tolist()))
    monkeypatch.setattr(md, "analyze_state", analyze)
    tree = search(init_tokens=1, num_candidates=1)
    result = tree.run()
    assert result.selected.immediate_reward == result.selected.mean_reward == -1
    assert tree.posteriors[0]["GEN"].m == tree.posteriors[1]["node"].m == -.5


def test_nondefault_priors_are_shared_by_all_model_types_but_independent():
    tree = search(prior_m=-.2, prior_kappa=2, prior_nu=3, prior_tau2=.4)
    tree.step()
    prior = asdict(GaussianPosterior(-.2, 2, 3, .4))
    for node_id, name in ((0, "node"), (0, "CONT"), (1, "GEN"), (1, "CONT")):
        assert asdict(tree.posteriors[node_id][name]) == prior
    assert tree.posteriors[1]["node"].kappa == 3
    assert tree.posteriors[0]["GEN"].kappa == 3


def test_reports_are_cached_and_trace_contains_no_timing(monkeypatch):
    tree = finite_tree(monkeypatch, "shared")
    tree.run()
    before = len(tree.model.seen)
    rng_state = json.dumps(tree.rng.bit_generator.state)
    for _ in range(3):
        json.dumps(tree_record(tree), allow_nan=False)
        json.dumps(tree_statistics(tree), allow_nan=False)
        json.dumps(tree.result().stats, allow_nan=False)
    assert len(tree.model.seen) == before
    assert json.dumps(tree.rng.bit_generator.state) == rng_state
    assert "seconds" not in json.dumps(tree.trace)


@pytest.mark.parametrize("cap", [None, 1, 2, 4, 7])
def test_medal_equivalent_mode_preserves_buffered_sibling_rounds(cap):
    settings = dict(init_tokens=2, num_candidates=3, top_k_tokens=2, top_k_actions=2,
                    max_model_calls=cap)
    original = md.MCTSSearch(FakeModel(), PROMPT, MASK, 4, md.MCTSConfig(**settings))
    adaptive = ABMEDALSearch(FakeModel(), PROMPT, MASK, 4,
                             ABMEDALConfig(**settings, search_policy="medal_equivalent"))
    for tree in (original, adaptive):
        try:
            tree.run()
        except md.SearchIncompleteError:
            pass
    left, right = original.result(), adaptive.result()
    assert left.stats["stop_reason"] == right.stats["stop_reason"]
    assert original.model_calls == adaptive.model_calls
    assert left.stats["cycles"] == right.stats["cycles"]
    assert [n.node_id for n in left.candidates] == [n.node_id for n in right.candidates]
    assert left.selected.node_id == right.selected.node_id
    assert len(original.nodes) == len(adaptive.nodes)
    for a, b in zip(original.nodes, adaptive.nodes):
        assert torch.equal(a.state, b.state)
        assert (a.depth, a.visits, a.reward_sum, a.mean_reward, a.immediate_reward, a.expanded, a.closed) == (
            b.depth, b.visits, b.reward_sum, b.mean_reward, b.immediate_reward, b.expanded, b.closed)
    if cap is None or cap == 7:
        assert original.model_calls == adaptive.model_calls == 7
        assert [n.node_id for n in left.candidates] == [3, 4, 6]


@pytest.mark.parametrize("kind", ["shared", "separated"])
def test_full_depth_medal_equivalent_tree_parity(monkeypatch, kind):
    adaptive = finite_tree(monkeypatch, kind, search_policy="medal_equivalent")
    config = md.MCTSConfig(**{name: getattr(adaptive.config, name) for name in md.MCTSConfig.__dataclass_fields__})
    original = md.MCTSSearch(FakeModel(), PROMPT, MASK, 21, config)
    left, right = original.run(), adaptive.run()
    assert original.model_calls == adaptive.model_calls == (23 if kind == "shared" else 61)
    assert [n.node_id for n in left.candidates] == [n.node_id for n in right.candidates]
    assert left.selected.node_id == right.selected.node_id
    plain_nodes = tree_record(original)["nodes"]
    adaptive_nodes = tree_record(adaptive)["nodes"]
    assert plain_nodes == [{k: v for k, v in row.items() if k not in ("posteriors", "remaining_actions", "decisions")}
                           for row in adaptive_nodes]
