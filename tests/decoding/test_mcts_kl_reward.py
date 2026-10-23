"""B: pre-commit branch-local historical KL, never a child-effect penalty."""

from dataclasses import replace
import json

import pytest
import torch

from dlm_project.decoding import mcts_decoding as md, mcts_confidence
from tests.decoding.helpers import FakeModel, MASK, PROMPT


def config(**overrides):
    return replace(md.MCTSConfig(init_tokens=2, num_candidates=3, top_k_tokens=1,
                               top_k_actions=2, stop_mode="budget", max_model_calls=7,
                               reward_mode="ig_kl", kl_weight=3.0, kl_top_k=0), **overrides)


def changing_policy(output):
    rows = [[0, .5, .25, .25] for _ in range(4)]
    if output[0].item() == 1:
        # Same entropy/confidence; only token identity at position 1 changes.
        rows[1] = [0, .25, .5, .25]
    return rows


def reference_kl(p, q, epsilon=1e-8):
    p, q = p.double().clamp_min(epsilon), q.double().clamp_min(epsilon)
    p, q = p / p.sum(-1, keepdim=True), q / q.sum(-1, keepdim=True)
    return (p * (p.log() - q.log())).sum(-1)


@pytest.mark.parametrize("bad", [dict(reward_mode="unknown"), dict(kl_weight=-1),
    dict(kl_weight=True), dict(kl_weight=float("nan")), dict(kl_weight=float("inf")),
    dict(kl_top_k=-1), dict(kl_top_k=True), dict(kl_top_k=1.5)])
def test_configuration_rejects_invalid_reward_settings(bad):
    with pytest.raises(ValueError):
        replace(config(), **bad)


@pytest.mark.parametrize("top_k", [0, 2, 4])
def test_kl_uses_old_token_ids_tail_and_aligned_positions(top_k):
    previous = torch.tensor([[.6, .2, .1, .1], [.1, .2, .6, .1], [.1, .1, .2, .6]])
    current = torch.tensor([[.1, .1, .3, .5], [.5, .3, .1, .1]])
    history = md.cache_distribution(torch.tensor([0, 2, 3]), previous, top_k)
    actual = md.historical_kl(history, torch.tensor([0, 3]), current, 1e-8)
    p = previous[[0, 2]]
    if top_k == 2:
        p, ids = p.topk(2, dim=-1)
        q = current.gather(-1, ids)  # Not the child's independently chosen top-k!
        p = torch.cat((p, 1 - p.sum(-1, keepdim=True)), -1)
        q = torch.cat((q, 1 - q.sum(-1, keepdim=True)), -1)
    else:
        q = current
    assert torch.allclose(actual.double(), reference_kl(p, q), atol=1e-6)
    assert not torch.allclose(actual.double(), reference_kl(q, p), atol=1e-4)
    assert history.probabilities.device.type == "cpu"
    assert torch.equal(history.positions, torch.tensor([0, 2, 3]))
    with pytest.raises(ValueError, match="masked"):
        md.historical_kl(history, torch.tensor([1, 3]), current, 1e-8)
    with pytest.raises(ValueError, match="masked"):
        md.historical_kl(history, torch.tensor([4, 5]), current, 1e-8)


@pytest.mark.parametrize("top_k", [0, 1])
def test_kl_stays_finite_at_zero_mass_and_zero_for_identical_distributions(top_k):
    p, q = torch.tensor([[1., 0., 0.]]), torch.tensor([[0., 0., 1.]])
    positions = torch.tensor([0])
    history = md.cache_distribution(positions, p, top_k)
    assert md.historical_kl(history, positions, p, 1e-8).item() == 0
    changed = md.historical_kl(history, positions, q, 1e-8).item()
    assert 0 < changed < 100


@pytest.mark.parametrize("top_k", [0, 2])
def test_reward_uses_grandparent_to_parent_at_outgoing_position_not_sibling_or_child(top_k):
    settings = replace(config(), kl_top_k=top_k)
    model = FakeModel(changing_policy)
    root = md.Node(0, torch.cat((PROMPT, torch.zeros((1, 4), dtype=torch.long)), 1))
    root.analysis = md.analyze_state(model, root.state, 2, MASK, settings)
    left = md.expand_node(root, md.Action(0, 1, .5), 1, 2, MASK, 2)
    right = md.expand_node(root, md.Action(0, 2, .25), 2, 2, MASK, 2)
    for node in (left, right):
        edge = md.evaluate_action(model, root, node, 2, MASK, settings)
        assert edge["kl_penalty"] == edge["action_history_kl"] == 0
        assert not edge["kl_history_available"]
        assert edge["reward"] == edge["immediate_gain"]
    # Right was visited last but must not replace left's own history.
    assert left.analysis.historical_kl[1] > 0
    assert right.analysis.historical_kl[1] == 0
    unstable = md.expand_node(left, md.Action(1, 2, .5), 3, 2, MASK, 2)
    stable = md.expand_node(left, md.Action(2, 1, .5), 4, 2, MASK, 2)
    a = md.evaluate_action(model, left, unstable, 2, MASK, settings)
    b = md.evaluate_action(model, left, stable, 2, MASK, settings)
    assert a["immediate_gain"] == pytest.approx(b["immediate_gain"])
    assert a["reward"] == pytest.approx(a["immediate_gain"] - settings.kl_weight * a["action_history_kl"])
    assert a["reward"] < b["reward"]
    assert b["reward"] == b["immediate_gain"] and b["kl_penalty"] == 0
    assert a["kl_reference_node_id"] == root.node_id
    assert a["kl_current_node_id"] == left.node_id and a["kl_output_position"] == 1
    assert unstable.analysis.distribution_history is None  # Terminal candidates need no history.
    assert len(model.seen) == 5  # No history reconstruction forwards.
    md.backpropagate(unstable, a["reward"])
    assert root.mean_reward == left.mean_reward == unstable.mean_reward == a["reward"]


@pytest.mark.parametrize("top_k", [0, 2])
def test_candidate_selection_changes_without_extra_forwards_and_releases_cache(top_k):
    settings = replace(config(), kl_top_k=top_k)
    original = md.MCTSSearch(FakeModel(changing_policy), PROMPT, MASK, 4,
                            replace(settings, reward_mode="ig"))
    modified = md.MCTSSearch(FakeModel(changing_policy), PROMPT, MASK, 4, settings)
    before, after = original.run(), modified.run()
    assert original.model_calls == modified.model_calls == 7
    assert after.stats["search_model_calls"] == 1 + after.stats["evaluated_children"]
    assert before.selected.state[0, 3].item() == 2  # Unstable position was frozen.
    assert after.selected.state[0, 3].item() == MASK  # Defer it with KL reward.
    assert after.selected.state[0, 4].item() == 1
    assert after.stats["selected_action_history_kl"] == 0
    assert after.stats["peak_history_cache_bytes"] > 0
    assert before.stats["peak_history_cache_bytes"] == 0
    assert after.stats["kl_extra_model_calls"] == 0
    assert all(n.analysis.distribution_history is None for n in modified.nodes)
    assert modified.result().selected is after.selected  # Read-only ranking needs no cache.
    json.dumps(after.stats, allow_nan=False)
    json.dumps(after.trace, allow_nan=False)


@pytest.mark.parametrize("mode", ["budget", "candidates"])
def test_zero_lambda_exactly_recovers_original_search_and_llada_completion(mode):
    settings = replace(config(), stop_mode=mode, max_model_calls=30)
    outputs, models = [], []
    for options in (replace(settings, reward_mode="ig"), replace(settings, kl_weight=0)):
        model = FakeModel(changing_policy)
        outputs.append(mcts_confidence.decode(model, PROMPT, MASK, gen_length=4, steps=4,
            block_length=4, temperature=0, seed=1, mcts=options, completion_method="llada"))
        models.append(model)
    a, b = outputs
    assert torch.equal(a.token_ids, b.token_ids)
    assert a.total_nfes == b.total_nfes == len(models[0].seen) == len(models[1].seen)
    assert all(torch.equal(x, y) for x, y in zip(models[0].seen, models[1].seen))
    assert a.details["selected_node_id"] == b.details["selected_node_id"]
    assert b.details["peak_history_cache_bytes"] == 0
    assert b.details["kl_estimator"] == "disabled"


@pytest.mark.parametrize("sampler", [md.complete_llada, md.complete_medal, md.complete_confidence])
def test_kl_never_caches_probabilities_in_completion(monkeypatch, sampler):
    settings = config()
    state = torch.cat((PROMPT, torch.tensor([[1, 0, 0, 0]])), 1)
    monkeypatch.setattr(md, "cache_distribution", lambda *a: pytest.fail("Completion cached KL history"))
    model = FakeModel(changing_policy)
    result = sampler(model, state, 2, MASK, settings)
    assert result.model_calls == len(model.seen) == 3


def test_history_is_released_when_search_raises():
    calls = 0
    def fail(output):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("controlled failure")
        return changing_policy(output)
    tree = md.MCTSSearch(FakeModel(fail), PROMPT, MASK, 4, config())
    with pytest.raises(RuntimeError, match="controlled failure"):
        tree.run()
    assert tree.root.analysis.distribution_history is None


@pytest.mark.parametrize("probability", [.99, 1.0])
def test_penalty_never_improves_negative_or_zero_ig(probability):
    def policy(output):
        tail = (1 - probability) / 2
        rows = [[0, probability, tail, tail] for _ in range(4)]
        if output[0].item() == 1:
            rows[1] = [0, tail, probability, tail]
        if (output != MASK).sum().item() > 1:
            rows = [[0, 1/3, 1/3, 1/3] for _ in range(4)]
        return rows
    settings, model = config(), FakeModel(policy)
    root = md.Node(0, torch.cat((PROMPT, torch.zeros((1, 4), dtype=torch.long)), 1))
    root.analysis = md.analyze_state(model, root.state, 2, MASK, settings)
    parent = md.expand_node(root, md.Action(0, 1, .5), 1, 2, MASK, 2)
    md.evaluate_action(model, root, parent, 2, MASK, settings)
    child = md.expand_node(parent, md.Action(1, 2, .5), 2, 2, MASK, 2)
    edge = md.evaluate_action(model, parent, child, 2, MASK, settings)
    assert edge["reward"] < edge["immediate_gain"] <= 0
    assert edge["kl_penalty"] > 0
    json.dumps(edge, allow_nan=False)


@pytest.mark.parametrize("budget,quota,reason", [(1, 1, "model_call_budget"), (7, 1, "candidate_quota")])
def test_quota_and_budget_termination_release_history(budget, quota, reason):
    settings = config(stop_mode="candidates", num_candidates=quota, max_model_calls=budget)
    tree = md.MCTSSearch(FakeModel(changing_policy), PROMPT, MASK, 4, settings)
    if budget == 1:
        with pytest.raises(md.SearchIncompleteError):
            tree.run()
    else:
        result = tree.run()
        assert len(result.candidates) == quota
    assert tree.stop_reason == reason
    assert tree.model_calls <= budget
    assert all(n.analysis is None or n.analysis.distribution_history is None for n in tree.nodes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")
@pytest.mark.parametrize("top_k", [0, 2])
def test_history_cache_transfers_on_cuda(top_k):
    positions = torch.tensor([0, 2])
    previous = torch.tensor([[.7, .2, .1], [.2, .1, .7]])
    current = torch.tensor([[.1, .3, .6], [.5, .3, .2]])
    cpu_history = md.cache_distribution(positions, previous, top_k)
    gpu_history = md.cache_distribution(positions, previous.cuda(), top_k)
    assert gpu_history.probabilities.device.type == "cpu"
    cpu = md.historical_kl(cpu_history, positions, current, 1e-8)
    gpu = md.historical_kl(gpu_history, positions, current.cuda(), 1e-8)
    assert torch.allclose(cpu, gpu, atol=1e-6)
