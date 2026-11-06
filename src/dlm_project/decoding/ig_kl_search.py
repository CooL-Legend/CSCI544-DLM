"""Opt-in reward B for MEDAL and AB-MEDAL, without changing their policies.

For g -> s -> s_next, the outgoing action (i, v) receives
IG(s, s_next) - lambda * KL(p_g[i] || p_s[i]). History is branch-local,
pre-commit, and kept outside nodes so checkpoint copies never duplicate it.
"""

from dataclasses import dataclass
import math

import torch

from dlm_project.decoding import mcts_decoding as medal
from dlm_project.decoding.ab_medal import ABMEDALSearch


REWARD_VERSION = "ig_action_history_kl_v1"


def reward_settings(mode="ig", kl_weight=0.05, kl_top_k=32):
    if mode not in ("ig", "ig_kl"):
        raise ValueError("reward_mode must be ig or ig_kl.")
    if (isinstance(kl_weight, bool) or not isinstance(kl_weight, (int, float))
            or not math.isfinite(kl_weight) or kl_weight < 0):
        raise ValueError("kl_weight must be finite and nonnegative.")
    if type(kl_top_k) is not int or kl_top_k < 0:
        raise ValueError("kl_top_k must be a nonnegative integer (0 = full vocabulary).")
    return {"reward_mode": mode, "reward_version": REWARD_VERSION if mode == "ig_kl" else "ig_v1",
            "kl_weight": float(kl_weight) if mode == "ig_kl" else 0.0,
            "kl_top_k": kl_top_k if mode == "ig_kl" else 0}


@dataclass
class DistributionHistory:
    positions: torch.Tensor
    probabilities: torch.Tensor
    token_ids: torch.Tensor | None
    vocab_size: int

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (self.positions, self.probabilities, self.token_ids) if t is not None)


def cache_distribution(positions, probabilities, top_k):
    """The previous top-k IDs plus one tail bucket, or full-vocabulary CPU rows."""
    vocabulary = probabilities.shape[-1]
    if 0 < top_k < vocabulary:
        values, ids = probabilities.topk(top_k, dim=-1)
        values = torch.cat((values, (1 - values.sum(-1, keepdim=True)).clamp_min(0)), -1)
        return DistributionHistory(positions.clone(), values.detach().cpu(), ids.cpu(), vocabulary)
    return DistributionHistory(positions.clone(), probabilities.detach().to("cpu", copy=True), None, vocabulary)


def historical_kl(history, positions, probabilities, epsilon):
    """KL(previous || current), aligned by position and previous token IDs.

    Top-k + tail is a coarse-grained approximation. Smoothing and renormalizing
    both distributions keeps zero-probability cases finite, as in the sean version.
    """
    if history.vocab_size != probabilities.shape[-1]:
        raise ValueError("Historical KL requires the same vocabulary.")
    rows = torch.searchsorted(history.positions, positions)
    if ((rows >= len(history.positions)).any()
            or not torch.equal(history.positions[rows], positions)):
        raise ValueError("Historical KL positions must still be masked in both states.")
    result = torch.empty(len(positions))
    chunk_size = max(1, len(positions)) if history.token_ids is not None else 16
    for start in range(0, len(positions), chunk_size):
        stop = min(start + chunk_size, len(positions))
        indices = rows[start:stop]
        previous = history.probabilities[indices].to(probabilities.device)
        current = probabilities[start:stop]
        if history.token_ids is not None:
            ids = history.token_ids[indices].to(probabilities.device)
            current = current.gather(-1, ids)
            current = torch.cat((current, (1 - current.sum(-1, keepdim=True)).clamp_min(0)), -1)
        previous, current = previous.clamp_min(epsilon), current.clamp_min(epsilon)
        previous = previous / previous.sum(-1, keepdim=True)
        current = current / current.sum(-1, keepdim=True)
        kl = (previous * (previous.log() - current.log())).sum(-1)
        if not torch.isfinite(kl).all():
            raise ValueError("Non-finite historical KL.")
        result[start:stop] = kl.clamp_min(0).cpu()
    return result


class _HistoryKLReward:
    """Shared reward-only mixin: no policy changes, global patching, or re-forwards."""

    def __init__(self, *args, kl_weight=0.05, kl_top_k=32, **kwargs):
        self.reward_settings = reward_settings("ig_kl", kl_weight, kl_top_k)
        self._histories, self._historical_kl, self._node_rewards = {}, {}, {}
        self.peak_history_cache_bytes = 0
        self.kl_estimator = "disabled"
        super().__init__(*args, **kwargs)

    @property
    def _uses_history(self):
        return self.reward_settings["kl_weight"] > 0

    def _analyze_node(self, node, *, logits=None):
        if not self._uses_history:
            return super()._analyze_node(node, logits=logits)
        if logits is None:
            analysis, positions, probabilities, _ = medal._distribution(
                self._counted, node.state, self.prompt_len, self.mask_id, self.config,
                search_excluded_token_ids=self.search_excluded_token_ids)
        else:
            analysis, positions, probabilities, _ = medal.distribution_from_logits(
                logits, node.state, self.prompt_len, self.mask_id, self.config,
                search_excluded_token_ids=self.search_excluded_token_ids)
        # Leaves have no outgoing action, so they need neither a history nor KL.
        if node.depth < self.config.init_tokens and probabilities is not None:
            if node.parent is not None:
                history = self._histories.get(node.parent.node_id)
                if history is None:
                    raise ValueError("Missing branch-local history; do not re-forward to reconstruct it.")
                values = historical_kl(history, positions, probabilities, self.config.epsilon)
                self._historical_kl[node.node_id] = dict(zip(positions.tolist(), values.tolist()))
            cached = cache_distribution(positions, probabilities, self.reward_settings["kl_top_k"])
            self._histories[node.node_id] = cached
            self.kl_estimator = "full" if cached.token_ids is None else "top_k_tail"
            self.peak_history_cache_bytes = max(
                self.peak_history_cache_bytes, sum(h.nbytes for h in self._histories.values()))
        return analysis

    def _apply_child_analysis(self, parent, child, analysis):
        action_kl = 0.0
        if self._uses_history and parent.parent is not None:
            action_kl = self._historical_kl[parent.node_id][child.action.output_position]
        evaluation = super()._apply_child_analysis(parent, child, analysis)
        gain = evaluation["immediate_gain"]
        penalty = self.reward_settings["kl_weight"] * action_kl
        reward = gain - penalty
        if not math.isfinite(reward):
            raise ValueError("Non-finite IG-KL reward.")
        child.immediate_reward = reward
        components = {"immediate_gain": gain, "action_history_kl": action_kl,
                      "kl_penalty": penalty, "reward": reward}
        self._node_rewards[child.node_id] = components
        evaluation.update(**components, **self.reward_settings,
                          kl_history_available=self._uses_history and parent.parent is not None,
                          kl_reference_node_id=parent.parent.node_id if parent.parent else None,
                          kl_current_node_id=parent.node_id,
                          kl_output_position=child.action.output_position,
                          kl_extra_model_calls=0)
        # Both policies evaluate each retained action at most once. AB can return
        # to this parent later: keep its history until the LAST sibling is done.
        if sum(c.immediate_reward is not None for c in parent.children) == len(parent.analysis.actions):
            self._histories.pop(parent.node_id, None)
            self._historical_kl.pop(parent.node_id, None)
        return evaluation

    def _assumptions(self):
        assumptions = super()._assumptions()
        assumptions["reward"] = (
            "IG(s,s_next) minus lambda * KL(p_g[i] || p_s[i]); g is s.parent, "
            "i is the outgoing pre-commit position; root penalty zero; no extra forwards")
        return assumptions

    def _emit(self, stage, **details):
        if stage == "candidate_evaluation":
            components = self._node_rewards.get(details["node_id"], {})
            for key in ("immediate_gain", "action_history_kl", "kl_penalty"):
                if key in components:
                    details[key] = components[key]
        super()._emit(stage, **details)

    def _reward_metadata(self, selected):
        components = self._node_rewards.get(selected.node_id, {})
        return {**self.reward_settings, "kl_estimator": self.kl_estimator,
                "kl_direction": "previous_to_current", "kl_extra_model_calls": 0,
                "peak_history_cache_bytes": self.peak_history_cache_bytes,
                "selected_ig_reward": components.get("immediate_gain"),
                "selected_action_history_kl": components.get("action_history_kl", 0.0),
                "selected_kl_penalty": components.get("kl_penalty", 0.0)}

    def _snapshot_checkpoint(self, budget, stop_reason="model_call_budget"):
        snapshot = super()._snapshot_checkpoint(budget, stop_reason=stop_reason)
        snapshot["reward_metadata"] = self._reward_metadata(snapshot["selected"])
        snapshot["node_rewards"] = {key: dict(value) for key, value in self._node_rewards.items()}
        return snapshot

    def result(self):
        result = super().result()
        result.stats.update(self._reward_metadata(result.selected))
        result.stats["evaluation_variant"] += " + pre-commit historical KL penalty"
        for key in ("candidate_evaluations", "candidate_shortlist"):
            for row in result.stats[key]:
                components = self._node_rewards.get(row["node_id"], {})
                for field in ("immediate_gain", "action_history_kl", "kl_penalty"):
                    if field in components:
                        row[field] = components[field]
        return result

    def _clear_histories(self):
        self._histories.clear()
        self._historical_kl.clear()

    def _finish(self, reason):
        try:
            return super()._finish(reason)
        finally:
            self._clear_histories()

    def step(self):
        try:
            return super().step()
        except Exception:
            self._clear_histories()
            raise


class IGKLMCTSSearch(_HistoryKLReward, medal.MCTSSearch):
    """Original MEDAL UCB search with reward B."""


class IGKLABMEDALSearch(_HistoryKLReward, ABMEDALSearch):
    """Original AB-MEDAL Student-t GEN/CONT search with reward B."""
