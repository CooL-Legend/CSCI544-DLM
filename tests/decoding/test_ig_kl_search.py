"""Reward B has the same meaning for UCB and Student-t search."""

import math

import pytest
import torch

from dlm_project.decoding import mcts_decoding as md, search_audit
from dlm_project.decoding.ab_medal import ABMEDALConfig, ABMEDALSearch
from dlm_project.decoding.ig_kl_search import (
    IGKLMCTSSearch, IGKLABMEDALSearch, cache_distribution, historical_kl, reward_settings,
)
from tests.decoding.helpers import FakeModel, MASK, PROMPT


PAIRS = [(md.MCTSSearch, IGKLMCTSSearch, md.MCTSConfig),
         (ABMEDALSearch, IGKLABMEDALSearch, ABMEDALConfig)]


def changing_policy(output):
    rows = [[0.0, 0.60, 0.25, 0.15] for _ in output]
    if (output != MASK).any():
        # Different siblings produce different histories for the SAME position.
        weights = [0.0, 0.15, 0.75, 0.10] if output[0] == 1 else [0.0, 0.40, 0.10, 0.50]
        rows = [weights for _ in output]
    return rows


def config(cls, budget=12):
    return cls(init_tokens=3, num_candidates=2, top_k_tokens=2, top_k_actions=2,
               stop_mode="budget", max_model_calls=budget)


def smoothed_kl(p, q, epsilon=1e-8):
    p, q = p.clamp_min(epsilon), q.clamp_min(epsilon)
    p, q = p / p.sum(-1, keepdim=True), q / q.sum(-1, keepdim=True)
    return (p * (p.log() - q.log())).sum(-1).clamp_min(0)


@pytest.mark.parametrize("top_k", [0, 1, 2, 3, 99])
def test_kl_direction_alignment_and_old_topk_ids(top_k):
    old = torch.tensor([[.6, .3, .1], [.2, .7, .1], [.1, .2, .7]])
    current = torch.tensor([[.8, .05, .15], [.8, .1, .1]])
    cached = cache_distribution(torch.tensor([0, 2, 5]), old, top_k)
    observed = historical_kl(cached, torch.tensor([2, 5]), current, 1e-8)
    previous = old[1:]
    if 0 < top_k < 3:
        previous, ids = previous.topk(top_k, dim=-1)
        current = current.gather(-1, ids)
        previous = torch.cat([previous, 1 - previous.sum(-1, keepdim=True)], -1)
        current = torch.cat([current, 1 - current.sum(-1, keepdim=True)], -1)
    assert observed.tolist() == pytest.approx(smoothed_kl(previous, current).tolist())
    assert not torch.allclose(observed, smoothed_kl(current, previous))


def test_kl_finite_at_zeros_and_rejects_invalid_position():
    cached = cache_distribution(torch.tensor([1]), torch.tensor([[1., 0.]]), 0)
    assert math.isfinite(historical_kl(cached, torch.tensor([1]), torch.tensor([[0., 1.]]), 1e-8).item())
    with pytest.raises(ValueError, match="still be masked"):
        historical_kl(cached, torch.tensor([2]), torch.tensor([[0., 1.]]), 1e-8)


@pytest.mark.parametrize("original,variant,cfg_cls", PAIRS)
def test_branch_local_precommit_reward_backups_and_no_extra_forwards(original, variant, cfg_cls):
    model = FakeModel(changing_policy)
    search = variant(model, PROMPT, MASK, 5, config(cfg_cls, 20), kl_weight=0.5, kl_top_k=0)
    result = search.run()
    events = [e for e in search.trace if e["stage"] == "evaluation"]
    assert len(model.seen) == search.model_calls == 1 + len(events)
    assert any(e["action_history_kl"] > 0 for e in events)
    for e in events:
        parent = search.nodes[e["parent_id"]]
        child = search.nodes[e["child_id"]]
        expected = 0.0
        if parent.parent:
            i = child.action.output_position
            old = torch.tensor(changing_policy(parent.parent.state[0, 2:]))[i]
            current = torch.tensor(changing_policy(parent.state[0, 2:]))[i]
            expected = smoothed_kl(old, current).item()
            assert e["kl_reference_node_id"] == parent.parent.node_id
        assert e["action_history_kl"] == pytest.approx(expected, abs=1e-6)
        gain = md.information_gain(parent.analysis.total_entropy, child.analysis.total_entropy, 1e-8)
        assert e["immediate_gain"] == gain
        assert child.immediate_reward == pytest.approx(gain - 0.5 * expected, abs=1e-6)
        assert e["reward"] <= e["immediate_gain"]
    # Reconstruct and validate every backed-up visit/Q, not only the selected leaf.
    audit = search_audit.from_events(search.trace, mask_id=MASK, gen_length=5,
                                    requested_depth=3, requested_budget=20, search_nfes=search.model_calls,
                                    adaptive=(original is ABMEDALSearch), shortlist_limit=2)
    metrics = search_audit.metrics(audit)
    assert metrics["ig_mean"] == pytest.approx(sum(e["immediate_gain"] for e in events) / len(events))
    assert metrics["negative_ig_count"] == sum(e["immediate_gain"] < 0 for e in events)
    assert result.stats["kl_extra_model_calls"] == 0
    assert search.peak_history_cache_bytes > 0
    assert not search._histories and not search._historical_kl
    if original is ABMEDALSearch:
        updates = [e for e in search.trace if e["stage"] == "backpropagation"]
        for e, backup in zip(events, updates):
            assert backup["reward"] == e["reward"]
            assert all(u["reward"] == e["reward"] for u in backup["posterior_updates"])


@pytest.mark.parametrize("original,variant,cfg_cls", PAIRS)
def test_zero_lambda_reproduces_original_search_and_allocates_no_history(original, variant, cfg_cls):
    old = original(FakeModel(changing_policy), PROMPT, MASK, 5, config(cfg_cls), seed=3)
    new = variant(FakeModel(changing_policy), PROMPT, MASK, 5, config(cfg_cls), seed=3, kl_weight=0)
    a, b = old.run(), new.run()
    assert old.model_calls == new.model_calls
    assert torch.equal(a.selected.state, b.selected.state)
    assert [(n.analysis.state_key, n.visits, n.reward_sum) for n in old.nodes] == [
        (n.analysis.state_key, n.visits, n.reward_sum) for n in new.nodes]
    assert new.peak_history_cache_bytes == 0
    if original is ABMEDALSearch:
        assert old.posteriors == new.posteriors


@pytest.mark.parametrize("original,variant,cfg_cls", PAIRS)
@pytest.mark.parametrize("checkpoint", [1, 4, 7])
def test_checkpoint_prefix_equivalence_with_penalized_rewards(original, variant, cfg_cls, checkpoint):
    standalone = variant(FakeModel(changing_policy), PROMPT, MASK, 5, config(cfg_cls, checkpoint),
                         seed=42, kl_top_k=2, checkpoint_budgets=(checkpoint,))
    continuous = variant(FakeModel(changing_policy), PROMPT, MASK, 5, config(cfg_cls, 12),
                         seed=42, kl_top_k=2, checkpoint_budgets=(checkpoint, 12))
    standalone.run()
    continuous.run()
    left, right = standalone.checkpoints[checkpoint], continuous.checkpoints[checkpoint]
    for key in ("node_rewards", "selected_state_hash", "shortlist_node_ids", "selected_q", "reward_metadata"):
        assert left[key] == right[key]
    assert [(n.node_id, n.visits, n.reward_sum) for n in left["nodes"]] == [
        (n.node_id, n.visits, n.reward_sum) for n in right["nodes"]]
    assert all(not hasattr(n.analysis, "distribution_history") for n in right["nodes"])
    if original is ABMEDALSearch:
        assert left["posteriors"] == right["posteriors"]


@pytest.mark.parametrize("original,variant,cfg_cls", PAIRS)
def test_history_cleanup_on_forward_failure(original, variant, cfg_cls):
    def fail_after_root(output):
        if (output != MASK).any():
            raise RuntimeError("synthetic failure")
        return changing_policy(output)
    search = variant(FakeModel(fail_after_root), PROMPT, MASK, 5, config(cfg_cls))
    with pytest.raises(RuntimeError, match="synthetic failure"):
        search.run()
    assert not search._histories and not search._historical_kl


@pytest.mark.parametrize("options", [{"kl_weight": -1}, {"kl_weight": float("nan")},
                                    {"kl_weight": True}, {"kl_top_k": -1}, {"kl_top_k": 1.5}])
def test_reward_settings_validation(options):
    with pytest.raises(ValueError):
        reward_settings("ig_kl", **options)
