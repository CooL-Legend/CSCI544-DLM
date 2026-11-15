"""Two-step lookahead MCTS search for Experiment 2 (E2).

Implements E2MEDALSearch and E2ABMEDALSearch supporting four reward modes:
1. "fig" (Fractional Information Gain):
    R_FIG(s, s_i) = (H(s) - H(g*_i)) / max(H(s), epsilon)
2. "srig" (Sum of Relative Information Gains / FutureR_ig):
    FutureR_ig(s, a_i) = R_ig(s, s_i) + lambda * R_ig(s_i, g*_i)
                       = (H(s) - H(s_i)) / H(s) + lambda * (H(s_i) - H(g*_i)) / H(s_i)
3. "cmer" (Common-Mask Entropy Reduction):
    R_CMER(s, g*_i) = (1 / |M(g*_i)|) * sum_{j in M(g*_i)} [h_j(s) - h_j(g*_i)]
4. "swd" (Stability-Weighted Decoding two-step lookahead):
    Q_SWD(s, a_i) = sum_{j in M(g*_i)} q^j_{t+2} * exp(-beta * (KL^j(s, s_i) + KL^j(s_i, g*_i)))
    where q^j_{t+2} = max_v p_{t+2}^j(v) and KL^j(A, B) = D_KL(p_A^j || p_B^j).

Key properties:
- Virtual grandchildren g*_i are evaluated and cached in `cached_analyses` (and
  `cached_swd_states` when reward_mode == "swd") to save redundant forwards, and are
  never attached to the MCTS tree or assigned GEN/CONT posteriors.
- Terminal children (0 remaining masks) have H(s_i) = 0 and second-step gain 0 (and
  R_CMER = 0.0 / Q_SWD = 0.0 since |M(g*_i)| = 0).
- Non-terminal children with no legal actions fall back to a reward-consistent 1-step
  fallback (1-step R_ig for fig/srig; 1-step R_CMER over M(s_i) for cmer; 1-step
  R_stability over M(s_i) for swd) with lookahead_available=False.
- Intermediate checkpoints are enforced before dispatching uncached grandchild forwards
  via isolated hypothetical backups that never mutate the continuing live tree, while
  also enforcing the 4B simulation-cap safeguard after complete simulations.
"""

from collections import Counter
import copy
from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from dlm_project.decoding import mcts_decoding as medal
from dlm_project.decoding.ab_medal import ABMEDALConfig, ABMEDALSearch, EvaluationRequest
from dlm_project.decoding.medal_batch import CooperativeMEDALSearch

VALID_REWARD_MODES = ("fig", "srig", "cmer", "swd")
DEFAULT_SIM_CAP_MULTIPLIER = 4
DEFAULT_SWD_BETA = 1.0
DEFAULT_SWD_TOPK = 512


def compute_fig_reward(parent_entropy: float, grandchild_entropy: float, epsilon: float = 1e-8) -> float:
    """Compute 2-step lookahead Fractional Information Gain (FIG) reward."""
    denom = max(parent_entropy, epsilon)
    gain = (parent_entropy - grandchild_entropy) / denom
    if not math.isfinite(gain):
        raise ValueError(f"Non-finite FIG reward: parent={parent_entropy}, grandchild={grandchild_entropy}")
    return gain


def compute_srig_reward(
    parent_entropy: float,
    child_entropy: float,
    grandchild_entropy: float,
    reward_lambda: float = 1.0,
    epsilon: float = 1e-8,
) -> float:
    """Compute 2-step Sum of Relative Information Gains (SRIG / FutureR_ig) reward:
        FutureR_ig(s, a_i) = R_ig(s, s_i) + lambda * R_ig(s_i, g*_i)
    """
    if (
        isinstance(reward_lambda, bool)
        or not isinstance(reward_lambda, (int, float))
        or not math.isfinite(reward_lambda)
    ):
        raise ValueError(f"Invalid reward_lambda: {reward_lambda}")
    r_step1 = medal.information_gain(parent_entropy, child_entropy, epsilon)
    r_step2 = medal.information_gain(child_entropy, grandchild_entropy, epsilon)
    gain = r_step1 + float(reward_lambda) * r_step2
    if not math.isfinite(gain):
        raise ValueError(
            f"Non-finite SRIG reward: parent={parent_entropy}, child={child_entropy}, "
            f"grandchild={grandchild_entropy}, lambda={reward_lambda}"
        )
    return gain


def compute_cmer_reward(
    parent_entropies: torch.Tensor,
    target_entropies: torch.Tensor,
    common_mask: torch.Tensor,
) -> Tuple[float, int]:
    """Compute Common-Mask Entropy Reduction over positions remaining masked in `target`.

    Returns ``(reward, common_mask_count)``. When ``|M(target)| == 0``, returns ``(0.0, 0)``.
    """
    mask = common_mask.to(dtype=torch.bool, device=parent_entropies.device)
    n_common = int(mask.sum().item())
    if n_common == 0:
        return 0.0, 0
    diff = parent_entropies[mask].double() - target_entropies[mask].double()
    reward = float(diff.mean().item())
    if not math.isfinite(reward):
        raise ValueError(f"Non-finite CMER reward: common_mask_count={n_common}")
    return reward, n_common


@dataclass(frozen=True)
class SWDStateData:
    """Compact CPU summary of per-position token distributions for SWD KL evaluation."""

    positions: torch.Tensor
    q_values: torch.Tensor
    vocab_size: int
    top_ids: Optional[torch.Tensor]
    top_probs: Optional[torch.Tensor]
    tail_prob: Optional[torch.Tensor]
    dense_probs: Optional[torch.Tensor]
    pos_lookup: Dict[int, int]


def extract_swd_state_data(
    probabilities: Optional[torch.Tensor],
    positions: torch.Tensor,
    *,
    top_k: Optional[int] = DEFAULT_SWD_TOPK,
) -> SWDStateData:
    """Extract exact max-prob q_values and compact distributions from masked-position softmax probabilities."""
    if probabilities is None or positions.numel() == 0:
        return SWDStateData(
            positions=torch.empty(0, dtype=torch.long),
            q_values=torch.empty(0, dtype=torch.float32),
            vocab_size=0 if probabilities is None else int(probabilities.shape[-1]),
            top_ids=None,
            top_probs=None,
            tail_prob=None,
            dense_probs=None,
            pos_lookup={},
        )

    pos_cpu = positions.detach().to(dtype=torch.long, device="cpu")
    pos_list = pos_cpu.tolist()
    pos_lookup = {pos: idx for idx, pos in enumerate(pos_list)}
    vocab_size = int(probabilities.shape[-1])
    q_values = probabilities.max(dim=-1).values.float().cpu()

    if top_k is None or vocab_size <= top_k:
        dense_cpu = probabilities.detach().float().cpu()
        return SWDStateData(
            positions=pos_cpu,
            q_values=q_values,
            vocab_size=vocab_size,
            top_ids=None,
            top_probs=None,
            tail_prob=None,
            dense_probs=dense_cpu,
            pos_lookup=pos_lookup,
        )

    top_vals, top_idx = torch.topk(probabilities, k=top_k, dim=-1)
    sorted_idx, sort_order = torch.sort(top_idx, dim=-1)
    sorted_vals = torch.gather(top_vals, -1, sort_order)

    probs_cpu = sorted_vals.detach().float().cpu()
    ids_cpu = sorted_idx.detach().to(dtype=torch.int32, device="cpu")
    row_sums = probs_cpu.double().sum(dim=-1)
    over = row_sums > 1.0
    if over.any():
        probs_cpu[over] = (probs_cpu[over].double() / row_sums[over].unsqueeze(-1)).float()
        row_sums = probs_cpu.double().sum(dim=-1)
    residual = (1.0 - row_sums).clamp(min=0.0)
    tail_prob = residual / float(vocab_size - top_k)

    return SWDStateData(
        positions=pos_cpu,
        q_values=q_values,
        vocab_size=vocab_size,
        top_ids=ids_cpu,
        top_probs=probs_cpu,
        tail_prob=tail_prob,
        dense_probs=None,
        pos_lookup=pos_lookup,
    )


def _expand_swd_rows_to_dense(dist: SWDStateData, row_indices: List[int]) -> torch.Tensor:
    if dist.dense_probs is not None:
        return dist.dense_probs[row_indices].double()
    dense = dist.tail_prob[row_indices].unsqueeze(-1).expand(-1, dist.vocab_size).clone()
    ids = dist.top_ids[row_indices].to(dtype=torch.long)
    dense.scatter_(-1, ids, dist.top_probs[row_indices].double())
    return dense


def compute_pairwise_kl(
    dist_a: SWDStateData,
    dist_b: SWDStateData,
    target_positions: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute forward KL D_KL(p_A^j || p_B^j) aligned at `target_positions` in float64."""
    pos_list = target_positions.detach().cpu().tolist()
    if not pos_list:
        return torch.empty(0, dtype=torch.float64)
    if dist_a.vocab_size != dist_b.vocab_size:
        raise ValueError(f"Vocabulary mismatch in KL calculation: {dist_a.vocab_size} vs {dist_b.vocab_size}")
    try:
        idx_a = [dist_a.pos_lookup[p] for p in pos_list]
        idx_b = [dist_b.pos_lookup[p] for p in pos_list]
    except KeyError as error:
        raise ValueError(f"Target position {error} missing from cached SWD distributions.") from error

    if dist_a.dense_probs is not None and dist_b.dense_probs is not None:
        pa = dist_a.dense_probs[idx_a].double()
        pb = dist_b.dense_probs[idx_b].double()
        kl = (pa * (torch.log(pa + epsilon) - torch.log(pb + epsilon))).sum(dim=-1)
        return kl.clamp(min=0.0)

    if dist_a.dense_probs is None and dist_b.dense_probs is None:
        ids_a = dist_a.top_ids[idx_a].to(dtype=torch.long)
        ids_b = dist_b.top_ids[idx_b].to(dtype=torch.long)
        probs_a = dist_a.top_probs[idx_a].double()
        probs_b = dist_b.top_probs[idx_b].double()
        tau_a = dist_a.tail_prob[idx_a]
        tau_b = dist_b.tail_prob[idx_b]

        k_a = ids_a.shape[-1]
        k_b = ids_b.shape[-1]

        pos_in_b = torch.searchsorted(ids_b, ids_a).clamp(max=k_b - 1)
        matched_b = torch.gather(ids_b, -1, pos_in_b) == ids_a
        pb_at_a = torch.where(matched_b, torch.gather(probs_b, -1, pos_in_b), tau_b.unsqueeze(-1))
        term_a = (probs_a * (torch.log(probs_a + epsilon) - torch.log(pb_at_a + epsilon))).sum(dim=-1)

        pos_in_a = torch.searchsorted(ids_a, ids_b).clamp(max=k_a - 1)
        in_a = torch.gather(ids_a, -1, pos_in_a) == ids_b
        term_b_only = torch.where(
            ~in_a,
            tau_a.unsqueeze(-1) * (torch.log(tau_a.unsqueeze(-1) + epsilon) - torch.log(probs_b + epsilon)),
            0.0,
        ).sum(dim=-1)

        n_rest = (dist_a.vocab_size - k_a - (~in_a).sum(dim=-1)).double()
        term_rest = n_rest * tau_a * (torch.log(tau_a + epsilon) - torch.log(tau_b + epsilon))
        return (term_a + term_b_only + term_rest).clamp(min=0.0)

    pa = _expand_swd_rows_to_dense(dist_a, idx_a)
    pb = _expand_swd_rows_to_dense(dist_b, idx_b)
    kl = (pa * (torch.log(pa + epsilon) - torch.log(pb + epsilon))).sum(dim=-1)
    return kl.clamp(min=0.0)


def _validate_reward_beta(reward_beta: float) -> float:
    if (
        isinstance(reward_beta, bool)
        or not isinstance(reward_beta, (int, float))
        or not math.isfinite(reward_beta)
        or reward_beta < 0.0
    ):
        raise ValueError(f"Invalid reward_beta: {reward_beta}")
    return float(reward_beta)


def compute_swd_reward(
    parent_swd: Optional[SWDStateData],
    child_swd: Optional[SWDStateData],
    grandchild_swd: Optional[SWDStateData],
    common_mask: torch.Tensor,
    *,
    beta: float = DEFAULT_SWD_BETA,
    epsilon: float = 1e-8,
) -> Tuple[float, int]:
    """Compute 2-step Stability-Weighted Decoding (SWD) reward over remaining masks in grandchild:
        Q_swd(Node, Action) = sum_{j in M(g*)} q^j_{t+2} * exp(-beta * (KL^j(Node, Child) + KL^j(Child, GrandChild)))
    """
    beta_val = _validate_reward_beta(beta)
    mask = common_mask.to(dtype=torch.bool, device="cpu")
    target_positions = torch.nonzero(mask, as_tuple=False).flatten()
    n_common = int(target_positions.numel())
    if n_common == 0:
        return 0.0, 0
    if parent_swd is None or child_swd is None or grandchild_swd is None:
        raise ValueError("2-step SWD reward requires cached SWD distributions for parent, child, and grandchild.")

    kl_pc = compute_pairwise_kl(parent_swd, child_swd, target_positions, epsilon=epsilon)
    kl_cg = compute_pairwise_kl(child_swd, grandchild_swd, target_positions, epsilon=epsilon)
    try:
        idx_g = [grandchild_swd.pos_lookup[p] for p in target_positions.tolist()]
    except KeyError as error:
        raise ValueError(f"Target position {error} missing from grandchild SWD q_values.") from error
    q_t2 = grandchild_swd.q_values[idx_g].double()
    weights = torch.exp(-beta_val * (kl_pc + kl_cg))
    reward = float((q_t2 * weights).sum().item())
    if not math.isfinite(reward):
        raise ValueError(f"Non-finite SWD reward: common_mask_count={n_common}, beta={beta_val}")
    return reward, n_common


def compute_swd_one_step_fallback(
    parent_swd: Optional[SWDStateData],
    child_swd: Optional[SWDStateData],
    child_mask: torch.Tensor,
    *,
    beta: float = DEFAULT_SWD_BETA,
    epsilon: float = 1e-8,
) -> Tuple[float, int]:
    """Compute 1-step stability reward R_stability(Node, Child) when grandchild lookahead is unavailable."""
    beta_val = _validate_reward_beta(beta)
    mask = child_mask.to(dtype=torch.bool, device="cpu")
    target_positions = torch.nonzero(mask, as_tuple=False).flatten()
    n_common = int(target_positions.numel())
    if n_common == 0:
        return 0.0, 0
    if parent_swd is None or child_swd is None:
        raise ValueError("1-step SWD fallback requires cached SWD distributions for parent and child.")

    kl_pc = compute_pairwise_kl(parent_swd, child_swd, target_positions, epsilon=epsilon)
    try:
        idx_c = [child_swd.pos_lookup[p] for p in target_positions.tolist()]
    except KeyError as error:
        raise ValueError(f"Target position {error} missing from child SWD q_values.") from error
    q_t1 = child_swd.q_values[idx_c].double()
    weights = torch.exp(-beta_val * kl_pc)
    reward = float((q_t1 * weights).sum().item())
    if not math.isfinite(reward):
        raise ValueError(f"Non-finite 1-step SWD fallback reward: common_mask_count={n_common}, beta={beta_val}")
    return reward, n_common


def _compute_e2_two_step_reward(
    reward_mode: str,
    reward_lambda: float,
    epsilon: float,
    parent_analysis: medal.StateAnalysis,
    child_analysis: medal.StateAnalysis,
    grandchild_analysis: Optional[medal.StateAnalysis],
    *,
    grandchild_state: Optional[torch.Tensor] = None,
    prompt_len: int = 0,
    mask_id: int = 0,
    reward_beta: float = DEFAULT_SWD_BETA,
    parent_swd: Optional[SWDStateData] = None,
    child_swd: Optional[SWDStateData] = None,
    grandchild_swd: Optional[SWDStateData] = None,
) -> Tuple[float, Optional[int]]:
    """Compute the 2-step lookahead reward and optional common-mask count."""
    if reward_mode == "swd":
        if grandchild_analysis is None or grandchild_state is None:
            return 0.0, 0
        common_mask = grandchild_state[0, prompt_len:] == mask_id
        return compute_swd_reward(
            parent_swd,
            child_swd,
            grandchild_swd,
            common_mask,
            beta=reward_beta,
            epsilon=epsilon,
        )
    if reward_mode == "cmer":
        if grandchild_analysis is None or grandchild_state is None:
            return 0.0, 0
        common_mask = grandchild_state[0, prompt_len:] == mask_id
        return compute_cmer_reward(parent_analysis.entropies, grandchild_analysis.entropies, common_mask)
    grandchild_entropy = 0.0 if grandchild_analysis is None else grandchild_analysis.total_entropy
    if reward_mode == "srig":
        return (
            compute_srig_reward(
                parent_analysis.total_entropy,
                child_analysis.total_entropy,
                grandchild_entropy,
                reward_lambda,
                epsilon,
            ),
            None,
        )
    return compute_fig_reward(parent_analysis.total_entropy, grandchild_entropy, epsilon), None


def _compute_e2_one_step_fallback(
    reward_mode: str,
    epsilon: float,
    parent_analysis: medal.StateAnalysis,
    child_analysis: medal.StateAnalysis,
    *,
    child_state: torch.Tensor,
    prompt_len: int,
    mask_id: int,
    reward_beta: float = DEFAULT_SWD_BETA,
    parent_swd: Optional[SWDStateData] = None,
    child_swd: Optional[SWDStateData] = None,
) -> Tuple[float, Optional[int]]:
    """Compute a reward-consistent 1-step fallback when 2-step lookahead is unavailable."""
    if reward_mode == "swd":
        child_mask = child_state[0, prompt_len:] == mask_id
        return compute_swd_one_step_fallback(
            parent_swd,
            child_swd,
            child_mask,
            beta=reward_beta,
            epsilon=epsilon,
        )
    if reward_mode == "cmer":
        child_mask = child_state[0, prompt_len:] == mask_id
        return compute_cmer_reward(parent_analysis.entropies, child_analysis.entropies, child_mask)
    return (
        medal.information_gain(parent_analysis.total_entropy, child_analysis.total_entropy, epsilon),
        None,
    )


def _validate_e2_params(
    reward_mode: str,
    reward_lambda: float,
    sim_cap_multiplier: int,
    reward_beta: float = DEFAULT_SWD_BETA,
    swd_topk: Optional[int] = DEFAULT_SWD_TOPK,
):
    if reward_mode not in VALID_REWARD_MODES:
        raise ValueError(f"reward_mode must be one of {VALID_REWARD_MODES}, got {reward_mode!r}.")
    if (
        isinstance(reward_lambda, bool)
        or not isinstance(reward_lambda, (int, float))
        or not math.isfinite(reward_lambda)
    ):
        raise ValueError(f"reward_lambda must be a finite number, got {reward_lambda!r}.")
    _validate_reward_beta(reward_beta)
    if swd_topk is not None and (type(swd_topk) is not int or swd_topk < 1):
        raise ValueError(f"swd_topk must be None or a positive integer, got {swd_topk!r}.")
    if type(sim_cap_multiplier) is not int or sim_cap_multiplier < 1:
        raise ValueError(f"sim_cap_multiplier must be a positive integer, got {sim_cap_multiplier!r}.")


def _build_snapshot_from_evaluated(
    evaluated_nodes: List[medal.Node],
    trace_copy: List[Dict[str, Any]],
    *,
    budget: int,
    actual_model_calls: int,
    simulations_completed: int,
    sim_cap_multiplier: int,
    init_tokens: int,
    num_candidates: int,
    root_action_count: int,
    stop_reason: str,
) -> Dict[str, Any]:
    """Construct a self-contained checkpoint dictionary from isolated copied state."""
    candidates, shortlist = medal.select_deepest_candidates(evaluated_nodes, limit=num_candidates)
    selected = shortlist[0]
    distinct_counts: Dict[int, set] = {}
    for n in evaluated_nodes:
        distinct_counts.setdefault(n.depth, set()).add(medal._state_key(n.state))
    counts = {d: len(states) for d, states in distinct_counts.items()}

    selection_record = {
        "stage": "search_selection",
        "trace_schema_version": "search_trace_v1",
        "requested_depth": init_tokens,
        "requested_budget": budget,
        "candidate_ids": [n.node_id for n in candidates],
        "shortlist_ids": [n.node_id for n in shortlist],
        "selected_node_id": selected.node_id,
        "selected_depth": selected.depth,
        "stop_reason": stop_reason or "model_call_budget",
        "fallback": selected.depth < init_tokens,
        "candidate_selection_version": medal.CANDIDATE_SELECTION_VERSION,
        "root_action_count": root_action_count,
        "node_statistics": [
            {
                "node_id": n.node_id,
                "visits": n.visits,
                "reward_sum": n.reward_sum,
                "q_value": n.mean_reward,
            }
            for n in evaluated_nodes
        ],
    }

    return {
        "budget": budget,
        "model_calls": actual_model_calls,
        "simulations_completed": simulations_completed,
        "simulation_cap": sim_cap_multiplier * budget,
        "nfe_utilization": (actual_model_calls / budget) if budget > 0 else 1.0,
        "nodes": evaluated_nodes,
        "candidates": candidates,
        "shortlist": shortlist,
        "selected": selected,
        "selection": selection_record,
        "trace": trace_copy,
        "stop_reason": stop_reason or "model_call_budget",
        "distinct_candidates_by_depth": counts,
        "max_depth_reached": max(distinct_counts.keys(), default=0),
        "selected_depth": selected.depth,
        "selection_depth_gap": max(distinct_counts.keys(), default=0) - selected.depth,
        "selected_depth_candidate_count": counts.get(selected.depth, 0),
        "shortlist_node_ids": [n.node_id for n in shortlist],
        "selected_node_id": selected.node_id,
        "selected_q": selected.mean_reward,
        "selected_state_hash": hashlib.sha256(repr(medal._state_key(selected.state)).encode()).hexdigest(),
        "reached_target_depth": init_tokens in distinct_counts,
        "selection_below_target": selected.depth < init_tokens,
        "fallback_used": selected.depth < init_tokens,
    }


class _E2RewardMixin:
    """Shared E2 two-step reward and 1-step fallback computation across tree search policies."""

    def _cache_swd_from_distribution(
        self,
        state_key: Tuple[int, ...],
        positions: torch.Tensor,
        probabilities: Optional[torch.Tensor],
    ):
        if self.reward_mode == "swd" and state_key not in self.cached_swd_states:
            self.cached_swd_states[state_key] = extract_swd_state_data(
                probabilities,
                positions,
                top_k=self.swd_topk,
            )

    def _resolve_evaluation_payload(
        self,
        request: EvaluationRequest,
        *,
        logits: Optional[torch.Tensor] = None,
        analysis: Optional[medal.StateAnalysis] = None,
    ) -> medal.StateAnalysis:
        if request is not self._pending or not request.attempted:
            raise ValueError("Expected the current dispatched evaluation request.")
        if (logits is None) == (analysis is None):
            raise ValueError("Supply exactly one of logits and analysis.")

        if analysis is None:
            analysis, positions, probabilities, _ = medal.distribution_from_logits(
                logits,
                request.node.state,
                self.prompt_len,
                self.mask_id,
                self.config,
                search_excluded_token_ids=self.search_excluded_token_ids,
            )
            if self.reward_mode == "swd":
                self._cache_swd_from_distribution(analysis.state_key, positions, probabilities)
            del probabilities
        elif self.reward_mode == "swd" and analysis.state_key not in self.cached_swd_states:
            if (request.node.state[0, self.prompt_len:] == self.mask_id).any():
                raise ValueError(
                    "SWD reward mode requires logits or cached SWD distribution data for non-terminal states."
                )
            self._cache_swd_from_distribution(
                analysis.state_key,
                torch.empty(0, dtype=torch.long),
                None,
            )
        return analysis

    def _compute_two_step_reward(
        self,
        parent_analysis: medal.StateAnalysis,
        child_analysis: medal.StateAnalysis,
        grandchild_analysis: Optional[medal.StateAnalysis],
        grandchild_state: Optional[torch.Tensor] = None,
    ) -> Tuple[float, Optional[int]]:
        parent_swd = child_swd = grandchild_swd = None
        if self.reward_mode == "swd":
            parent_swd = self.cached_swd_states.get(parent_analysis.state_key)
            child_swd = self.cached_swd_states.get(child_analysis.state_key)
            if grandchild_analysis is not None:
                grandchild_swd = self.cached_swd_states.get(grandchild_analysis.state_key)
        return _compute_e2_two_step_reward(
            self.reward_mode,
            self.reward_lambda,
            self.config.epsilon,
            parent_analysis,
            child_analysis,
            grandchild_analysis,
            grandchild_state=grandchild_state,
            prompt_len=self.prompt_len,
            mask_id=self.mask_id,
            reward_beta=self.reward_beta,
            parent_swd=parent_swd,
            child_swd=child_swd,
            grandchild_swd=grandchild_swd,
        )

    def _compute_one_step_fallback(
        self,
        parent_analysis: medal.StateAnalysis,
        child_analysis: medal.StateAnalysis,
        child_state: torch.Tensor,
    ) -> Tuple[float, Optional[int]]:
        parent_swd = child_swd = None
        if self.reward_mode == "swd":
            parent_swd = self.cached_swd_states.get(parent_analysis.state_key)
            child_swd = self.cached_swd_states.get(child_analysis.state_key)
        return _compute_e2_one_step_fallback(
            self.reward_mode,
            self.config.epsilon,
            parent_analysis,
            child_analysis,
            child_state=child_state,
            prompt_len=self.prompt_len,
            mask_id=self.mask_id,
            reward_beta=self.reward_beta,
            parent_swd=parent_swd,
            child_swd=child_swd,
        )


class E2MEDALSearch(_E2RewardMixin, CooperativeMEDALSearch):
    """Two-step lookahead MEDAL search with cooperative batching and state caching."""

    def __init__(
        self,
        *args,
        reward_mode: str = "fig",
        reward_lambda: float = 1.0,
        sim_cap_multiplier: int = DEFAULT_SIM_CAP_MULTIPLIER,
        reward_beta: float = DEFAULT_SWD_BETA,
        swd_topk: Optional[int] = DEFAULT_SWD_TOPK,
        **kwargs,
    ):
        _validate_e2_params(
            reward_mode,
            reward_lambda,
            sim_cap_multiplier,
            reward_beta=reward_beta,
            swd_topk=swd_topk,
        )
        super().__init__(*args, **kwargs)
        self.reward_mode = reward_mode
        self.reward_lambda = float(reward_lambda)
        self.reward_beta = float(reward_beta)
        self.swd_topk = swd_topk
        self.sim_cap_multiplier = sim_cap_multiplier
        self.cached_analyses: Dict[Tuple[int, ...], medal.StateAnalysis] = {}
        self.cached_swd_states: Dict[Tuple[int, ...], SWDStateData] = {}
        self._pending_grandchild: Optional[Dict[str, Any]] = None

    def _snapshot_checkpoint(self, budget: int, stop_reason: str = "model_call_budget") -> Dict[str, Any]:
        evaluated_source = [n for n in self.nodes if n.evaluation_order is not None]
        evaluated = copy.deepcopy(evaluated_source)
        root_actions = len(self.root.analysis.actions) if self.root.analysis else 0
        return _build_snapshot_from_evaluated(
            evaluated,
            copy.deepcopy(self.trace),
            budget=budget,
            actual_model_calls=min(self.model_calls, budget),
            simulations_completed=self.simulations_completed,
            sim_cap_multiplier=self.sim_cap_multiplier,
            init_tokens=self.config.init_tokens,
            num_candidates=self.config.num_candidates,
            root_action_count=root_actions,
            stop_reason=stop_reason or "model_call_budget",
        )

    def _snapshot_boundary_checkpoint(
        self,
        budget: int,
        child: medal.Node,
        parent: medal.Node,
        child_analysis: medal.StateAnalysis,
    ) -> Dict[str, Any]:
        """Snapshot checkpoint `budget` with a hypothetical 1-step fallback backup on an isolated copy.

        Never mutates the continuing live tree, guaranteeing that the checkpoint matches an
        independently terminated `budget`-call run while the live tree proceeds to `budget + 1`.
        """
        nodes_copy = copy.deepcopy(self.nodes)
        by_id = {n.node_id: n for n in nodes_copy}
        child_copy = by_id[child.node_id]
        parent_copy = by_id[parent.node_id]
        trace_copy = copy.deepcopy(self.trace)

        reward, common_mask_count = self._compute_one_step_fallback(
            parent_copy.analysis, child_analysis, child_copy.state
        )
        rig_step1 = medal.information_gain(
            parent_copy.analysis.total_entropy, child_analysis.total_entropy, self.config.epsilon
        )
        sim_order = self.simulations_completed + 1
        child_copy.analysis = child_analysis
        child_copy.immediate_reward = reward
        child_copy.evaluation_order = sim_order

        trace_copy.append({
            "cycle": self.cycle,
            "stage": "node_evaluated",
            "search_model_calls": budget,
            "trace_schema_version": "search_trace_v1",
            "node_id": child_copy.node_id,
            "parent_id": parent_copy.node_id,
            "depth": child_copy.depth,
            "evaluation_order": sim_order,
            "state_hash": medal.candidate_value_record(child_copy)["state_hash"],
            "immediate_reward": reward,
            "action": None if child_copy.action is None else asdict(child_copy.action),
        })

        masked_count = int((child_copy.state[0, self.prompt_len:] == self.mask_id).sum().item())
        trace_copy.append({
            "cycle": self.cycle,
            "stage": "evaluation",
            "search_model_calls": budget,
            "parent_id": parent_copy.node_id,
            "child_id": child_copy.node_id,
            "depth": child_copy.depth,
            "action": medal.action_record(child_copy.action, self.tokenizer),
            "entropy_before": parent_copy.analysis.total_entropy,
            "entropy_child": child_analysis.total_entropy,
            "entropy_grandchild": None,
            "rig_step1": rig_step1,
            "rig_step2": 0.0,
            "reward_lambda": self.reward_lambda,
            "reward_beta": self.reward_beta,
            "reward_mode": self.reward_mode,
            "common_mask_count": common_mask_count,
            "lookahead_available": False,
            "lookahead_status": "budget_boundary_fallback",
            "reward": reward,
            "immediate_gain": reward,
            "masked_count": masked_count,
            "mean_token_entropy": child_analysis.total_entropy / max(1, masked_count),
            "mean_confidence_margin": child_analysis.mean_margin,
            "signal_nfe": budget,
        })

        updates = medal.backpropagate(child_copy, reward)
        trace_copy.append({
            "cycle": self.cycle,
            "stage": "backpropagation",
            "search_model_calls": budget,
            "child_id": child_copy.node_id,
            "reward": reward,
            "updates": updates,
        })

        evaluated_copy = [n for n in nodes_copy if n.evaluation_order is not None]
        root_actions = len(self.root.analysis.actions) if self.root.analysis else 0
        return _build_snapshot_from_evaluated(
            evaluated_copy,
            trace_copy,
            budget=budget,
            actual_model_calls=budget,
            simulations_completed=sim_order,
            sim_cap_multiplier=self.sim_cap_multiplier,
            init_tokens=self.config.init_tokens,
            num_candidates=self.config.num_candidates,
            root_action_count=root_actions,
            stop_reason="model_call_budget",
        )

    def _check_checkpoint(self, stop_reason: Optional[str] = None):
        if not self.checkpoint_budgets:
            return
        for b in self.checkpoint_budgets:
            if b in self.checkpoints:
                continue
            if self.model_calls >= b:
                reason = stop_reason or "model_call_budget"
            elif self.simulations_completed >= self.sim_cap_multiplier * b:
                reason = stop_reason or "simulation_budget"
            else:
                continue
            self.checkpoints[b] = self._snapshot_checkpoint(b, stop_reason=reason)
            self._emit(
                "search_checkpoint",
                budget=b,
                model_calls=self.checkpoints[b]["model_calls"],
                simulations_completed=self.simulations_completed,
                stop_reason=reason,
                selected_node_id=self.checkpoints[b]["selected_node_id"],
                selected_depth=self.checkpoints[b]["selected_depth"],
            )

    def next_evaluation(self) -> Optional[EvaluationRequest]:
        while not self.done:
            if self._pending is not None:
                return self._pending

            # 1. Root initialization
            if self.root.analysis is None:
                self.cycle += 1
                self._root_cycle = True
                self._pending = EvaluationRequest(self.root, None, [], [], "tree_analysis")
                return self._pending

            if self._at_cap() or self._at_simulation_cap():
                self._finish("model_call_budget" if self._at_cap() else "simulation_budget")
                return None

            # 2. Outstanding grandchild evaluation from Phase 1
            if self._pending_grandchild is not None:
                grandchild = self._pending_grandchild["grandchild"]
                g_key = medal._state_key(grandchild.state)
                if g_key in self.cached_analyses:
                    g_analysis = self.cached_analyses[g_key]
                    self._resolve_child_reward(g_analysis, lookahead_status="cached_grandchild")
                    continue

                self._pending = EvaluationRequest(
                    grandchild, self._pending_grandchild["child"], [], [], "simulation"
                )
                return self._pending

            # 3. Frontier selection if needed
            while self._parent is None:
                if not self._root_cycle:
                    self.cycle += 1
                self._root_cycle = False
                with self._counted.phase("selection"):
                    parent, decisions = medal.select_node(self.root, self.config.exploration_c)
                if parent is None:
                    self._finish("frontier_exhausted")
                    return None
                self._emit(
                    "selection",
                    node_id=parent.node_id,
                    path=[0] + [hop["chosen_id"] for hop in decisions],
                    decisions=decisions,
                )
                parent.expanded = True
                self._remaining = list(parent.analysis.actions)
                if not self._remaining:
                    self._close_exhausted(parent)
                    if self.root.closed:
                        self._finish("frontier_exhausted")
                        return None
                    continue
                self._parent = parent

            parent = self._parent
            action = self._remaining.pop(0)
            if action.token_id in self.search_excluded_token_ids:
                raise RuntimeError("Excluded termination token reached search expansion.")

            # Expand child state s_i
            with self._counted.phase("expansion"):
                child = medal.expand_node(
                    parent, action, len(self.nodes), self.prompt_len, self.mask_id, self.config.init_tokens
                )
                self.nodes.append(child)

            self._emit(
                "expansion",
                parent_id=parent.node_id,
                child_id=child.node_id,
                depth=child.depth,
                action=medal.action_record(action, self.tokenizer),
            )
            self.simulations_started += 1
            self._emit("simulation_started", child_id=child.node_id, simulations_started=self.simulations_started)

            # Check if child is already in analysis cache
            c_key = medal._state_key(child.state)
            if c_key in self.cached_analyses:
                child_analysis = self.cached_analyses[c_key]
                self._prepare_grandchild_or_finalize(child, parent, child_analysis)
                continue

            self._pending = EvaluationRequest(child, parent, [], [], "simulation")
            return self._pending
        return None

    def _prepare_grandchild_or_finalize(
        self, child: medal.Node, parent: medal.Node, child_analysis: medal.StateAnalysis
    ):
        """Prepare grandchild lookahead or finalize child if terminal, actionless, cached, or at cap."""
        self.cached_analyses[medal._state_key(child.state)] = child_analysis
        child.analysis = child_analysis

        remaining_masks = int((child.state[0, self.prompt_len:] == self.mask_id).sum().item())

        # Case A: Terminal child (0 remaining masks -> H(child) = 0, second-step gain = 0, |M(g*)| = 0 -> CMER = 0)
        if remaining_masks == 0:
            reward, common_mask_count = self._compute_two_step_reward(
                parent.analysis, child_analysis, None
            )
            self._apply_evaluation_result(
                child,
                parent,
                child_analysis,
                reward,
                grandchild_entropy=0.0,
                lookahead_available=True,
                lookahead_status="terminal_child",
                common_mask_count=common_mask_count,
            )
            return

        # Case B: Non-terminal child with no legal actions -> fallback to reward-consistent 1-step reward
        if not child_analysis.actions:
            reward, common_mask_count = self._compute_one_step_fallback(
                parent.analysis, child_analysis, child.state
            )
            self._apply_evaluation_result(
                child,
                parent,
                child_analysis,
                reward,
                grandchild_entropy=None,
                lookahead_available=False,
                lookahead_status="no_legal_actions",
                common_mask_count=common_mask_count,
            )
            return

        # Choose highest-confidence action from child's action set
        greedy_action = child_analysis.actions[0]
        # Construct virtual grandchild state (never attached to tree)
        grandchild = medal.expand_node(
            child, greedy_action, -1, self.prompt_len, self.mask_id, self.config.init_tokens + 1
        )
        child.children.pop()

        g_key = medal._state_key(grandchild.state)
        g_remaining_masks = int((grandchild.state[0, self.prompt_len:] == self.mask_id).sum().item())
        if g_remaining_masks == 0 and g_key not in self.cached_analyses:
            zero_entropies = torch.zeros(grandchild.state.shape[1] - self.prompt_len)
            self.cached_analyses[g_key] = medal.StateAnalysis(zero_entropies, (), g_key, 0.0)
        if g_remaining_masks == 0 and self.reward_mode == "swd" and g_key not in self.cached_swd_states:
            self._cache_swd_from_distribution(g_key, torch.empty(0, dtype=torch.long), None)

        # Case C1: Grandchild is already cached (or terminal with 0 masks) -> resolve with 0 extra NFEs
        if g_key in self.cached_analyses:
            g_analysis = self.cached_analyses[g_key]
            grandchild.analysis = g_analysis
            reward, common_mask_count = self._compute_two_step_reward(
                parent.analysis, child_analysis, g_analysis, grandchild.state
            )
            self._apply_evaluation_result(
                child,
                parent,
                child_analysis,
                reward,
                grandchild_entropy=g_analysis.total_entropy,
                lookahead_available=True,
                lookahead_status="cached_grandchild",
                common_mask_count=common_mask_count,
            )
            return

        # Case C2: Grandchild is uncached (requires 1 new model forward pass)
        # If at hard model-call cap, cannot dispatch another forward; fallback to reward-consistent 1-step reward
        if self._at_cap():
            reward, common_mask_count = self._compute_one_step_fallback(
                parent.analysis, child_analysis, child.state
            )
            self._apply_evaluation_result(
                child,
                parent,
                child_analysis,
                reward,
                grandchild_entropy=None,
                lookahead_available=False,
                lookahead_status="budget_boundary_fallback",
                common_mask_count=common_mask_count,
            )
            return

        # If at an intermediate checkpoint budget b, snapshot hypothetical 1-step boundary state
        # on an isolated copy before dispatching call b + 1 for the grandchild.
        if self.checkpoint_budgets:
            for b in self.checkpoint_budgets:
                if b not in self.checkpoints and self.model_calls == b:
                    self.checkpoints[b] = self._snapshot_boundary_checkpoint(b, child, parent, child_analysis)
                    self._emit(
                        "search_checkpoint",
                        budget=b,
                        model_calls=b,
                        simulations_completed=self.simulations_completed + 1,
                        stop_reason="model_call_budget",
                        boundary_hypothetical=True,
                        selected_node_id=self.checkpoints[b]["selected_node_id"],
                        selected_depth=self.checkpoints[b]["selected_depth"],
                    )

        # Store pending lookahead data for next_evaluation()
        self._pending_grandchild = {
            "child": child,
            "parent": parent,
            "child_analysis": child_analysis,
            "grandchild": grandchild,
            "greedy_action": greedy_action,
        }

    def _resolve_child_reward(
        self, g_analysis: medal.StateAnalysis, *, lookahead_status: str = "evaluated_grandchild"
    ):
        """Apply grandchild analysis to compute 2-step reward and backpropagate to child."""
        data = self._pending_grandchild
        child, parent = data["child"], data["parent"]
        grandchild = data["grandchild"]
        child_analysis = data["child_analysis"]
        self._pending_grandchild = None

        grandchild.analysis = g_analysis
        self.cached_analyses[medal._state_key(grandchild.state)] = g_analysis

        grandchild_entropy = g_analysis.total_entropy
        reward, common_mask_count = self._compute_two_step_reward(
            parent.analysis, child_analysis, g_analysis, grandchild.state
        )
        self._apply_evaluation_result(
            child,
            parent,
            child_analysis,
            reward,
            grandchild_entropy=grandchild_entropy,
            lookahead_available=True,
            lookahead_status=lookahead_status,
            common_mask_count=common_mask_count,
        )

    def _apply_evaluation_result(
        self,
        child: medal.Node,
        parent: medal.Node,
        child_analysis: medal.StateAnalysis,
        reward: float,
        grandchild_entropy: Optional[float],
        *,
        lookahead_available: bool = True,
        lookahead_status: str = "evaluated_grandchild",
        common_mask_count: Optional[int] = None,
    ):
        """Commit child reward, update statistics, check checkpoints and finalize."""
        child.analysis = child_analysis
        self.cached_analyses[medal._state_key(child.state)] = child_analysis
        child.immediate_reward = reward
        self.simulations_completed += 1
        self._record_evaluated(child)

        rig_step1 = medal.information_gain(
            parent.analysis.total_entropy, child_analysis.total_entropy, self.config.epsilon
        )
        rig_step2 = (
            medal.information_gain(child_analysis.total_entropy, grandchild_entropy, self.config.epsilon)
            if (lookahead_available and grandchild_entropy is not None)
            else 0.0
        )
        masked_count = int((child.state[0, self.prompt_len:] == self.mask_id).sum().item())

        evaluation = {
            "entropy_before": parent.analysis.total_entropy,
            "entropy_child": child_analysis.total_entropy,
            "entropy_grandchild": grandchild_entropy,
            "rig_step1": rig_step1,
            "rig_step2": rig_step2,
            "reward_lambda": self.reward_lambda,
            "reward_beta": self.reward_beta,
            "reward_mode": self.reward_mode,
            "common_mask_count": common_mask_count,
            "lookahead_available": lookahead_available,
            "lookahead_status": lookahead_status,
            "reward": reward,
            "immediate_gain": reward,
            "masked_count": masked_count,
            "mean_token_entropy": child_analysis.total_entropy / max(1, masked_count),
            "mean_confidence_margin": child_analysis.mean_margin,
            "signal_nfe": self.model_calls,
        }
        self._emit(
            "evaluation",
            parent_id=parent.node_id,
            child_id=child.node_id,
            depth=child.depth,
            action=medal.action_record(child.action, self.tokenizer),
            **evaluation,
        )

        with self._counted.phase("backpropagation"):
            updates = medal.backpropagate(child, reward)
        self._emit("backpropagation", child_id=child.node_id, reward=reward, updates=updates)
        self._check_checkpoint()

        if child.depth == self.config.init_tokens:
            key = medal._state_key(child.state)
            duplicate = key in self.candidates
            self.candidates.setdefault(key, child)
            self._emit(
                "candidate",
                node_id=child.node_id,
                duplicate=duplicate,
                candidate_count=len(self.candidates),
                score=reward,
            )

        self._pending = None
        if self._at_cap() or self._at_simulation_cap():
            self._finish("model_call_budget" if self._at_cap() else "simulation_budget")
            return
        if not self._remaining:
            self._close_exhausted(parent)
            self._parent = None
            if self.root.closed:
                self._finish("frontier_exhausted")

    def submit_evaluation(
        self,
        request: EvaluationRequest,
        *,
        logits: Optional[torch.Tensor] = None,
        analysis: Optional[medal.StateAnalysis] = None,
    ):
        analysis = self._resolve_evaluation_payload(request, logits=logits, analysis=analysis)

        child, parent = request.node, request.parent

        # Case 1: Root initialization
        if parent is None:
            self.root.analysis = analysis
            self.cached_analyses[medal._state_key(self.root.state)] = analysis
            self._record_evaluated(self.root)
            self._emit(
                "initialization",
                node_id=0,
                entropy=analysis.total_entropy,
                actions=[medal.action_record(a, self.tokenizer) for a in analysis.actions],
            )
            self._pending = None
            self._check_checkpoint()
            return

        # Case 2: Grandchild evaluation (resolves 2-step reward)
        if self._pending_grandchild is not None and request.node is self._pending_grandchild["grandchild"]:
            self._resolve_child_reward(analysis, lookahead_status="evaluated_grandchild")
            return

        # Case 3: Child evaluation (Phase 1)
        self._prepare_grandchild_or_finalize(child, parent, analysis)
        self._pending = None

    @torch.inference_mode()
    def step(self):
        start = len(self.trace)
        while not self.done:
            request = self.next_evaluation()
            if request is None:
                break
            self.record_external_forward(request, invocation_id=self.model_calls + 1, batch_size=1)
            prediction = self.model(request.node.state.to(medal.model_device(self.model)))
            self.submit_evaluation(request, logits=prediction.logits[0])
            del prediction
            if request.parent is not None and self._pending_grandchild is None:
                break
        return self.trace[start:]


class E2ABMEDALSearch(_E2RewardMixin, ABMEDALSearch):
    """Two-step lookahead AB-MEDAL search with greedy continuation lookahead and caching."""

    def __init__(
        self,
        *args,
        reward_mode: str = "fig",
        reward_lambda: float = 1.0,
        sim_cap_multiplier: int = DEFAULT_SIM_CAP_MULTIPLIER,
        reward_beta: float = DEFAULT_SWD_BETA,
        swd_topk: Optional[int] = DEFAULT_SWD_TOPK,
        **kwargs,
    ):
        _validate_e2_params(
            reward_mode,
            reward_lambda,
            sim_cap_multiplier,
            reward_beta=reward_beta,
            swd_topk=swd_topk,
        )
        super().__init__(*args, **kwargs)
        self.reward_mode = reward_mode
        self.reward_lambda = float(reward_lambda)
        self.reward_beta = float(reward_beta)
        self.swd_topk = swd_topk
        self.sim_cap_multiplier = sim_cap_multiplier
        self.cached_analyses: Dict[Tuple[int, ...], medal.StateAnalysis] = {}
        self.cached_swd_states: Dict[Tuple[int, ...], SWDStateData] = {}
        self._pending_grandchild: Optional[Dict[str, Any]] = None

    def _snapshot_checkpoint(self, budget: int, stop_reason: str = "model_call_budget") -> Dict[str, Any]:
        evaluated_source = [n for n in self.nodes if n.evaluation_order is not None]
        evaluated = copy.deepcopy(evaluated_source)
        root_actions = len(self.root.analysis.actions) if self.root.analysis else 0
        snapshot = _build_snapshot_from_evaluated(
            evaluated,
            copy.deepcopy(self.trace),
            budget=budget,
            actual_model_calls=min(self.model_calls, budget),
            simulations_completed=self.simulations_completed,
            sim_cap_multiplier=self.sim_cap_multiplier,
            init_tokens=self.config.init_tokens,
            num_candidates=self.config.num_candidates,
            root_action_count=root_actions,
            stop_reason=stop_reason or "model_call_budget",
        )
        snapshot.update(
            posteriors={
                node_id: {k: copy.deepcopy(p) for k, p in models.items()}
                for node_id, models in self.posteriors.items()
            },
            remaining_actions={
                node_id: list(actions) for node_id, actions in self.remaining_actions.items()
            },
            decision_history={
                node_id: list(history) for node_id, history in self.decision_history.items()
            },
        )
        return snapshot

    def _snapshot_boundary_checkpoint(
        self,
        budget: int,
        child: medal.Node,
        parent: medal.Node,
        path: list,
        decisions: list,
        child_analysis: medal.StateAnalysis,
    ) -> Dict[str, Any]:
        """Snapshot checkpoint `budget` with a hypothetical 1-step fallback backup on an isolated copy.

        Never mutates the continuing live tree or its Student-t posteriors.
        """
        nodes_copy = copy.deepcopy(self.nodes)
        by_id = {n.node_id: n for n in nodes_copy}
        child_copy = by_id[child.node_id]
        parent_copy = by_id[parent.node_id]
        path_copy = [by_id[n.node_id] for n in path]
        trace_copy = copy.deepcopy(self.trace)
        posteriors_copy = {
            node_id: {k: copy.deepcopy(p) for k, p in models.items()}
            for node_id, models in self.posteriors.items()
        }
        remaining_copy = {
            node_id: list(actions) for node_id, actions in self.remaining_actions.items()
        }
        decisions_copy = {
            node_id: list(history) for node_id, history in self.decision_history.items()
        }

        reward, common_mask_count = self._compute_one_step_fallback(
            parent_copy.analysis, child_analysis, child_copy.state
        )
        rig_step1 = medal.information_gain(
            parent_copy.analysis.total_entropy, child_analysis.total_entropy, self.config.epsilon
        )
        sim_order = self.simulations_completed + 1
        child_copy.analysis = child_analysis
        child_copy.immediate_reward = reward
        child_copy.evaluation_order = sim_order

        if child_copy.node_id not in posteriors_copy:
            posteriors_copy[child_copy.node_id] = {
                name: self.config.new_posterior() for name in ("node", "GEN", "CONT")
            }
            remaining_copy[child_copy.node_id] = (
                list(child_analysis.actions) if child_copy.depth < self.config.init_tokens else []
            )
            decisions_copy[child_copy.node_id] = []

        trace_copy.append({
            "cycle": self.cycle,
            "stage": "node_evaluated",
            "search_model_calls": budget,
            "trace_schema_version": "search_trace_v1",
            "node_id": child_copy.node_id,
            "parent_id": parent_copy.node_id,
            "depth": child_copy.depth,
            "evaluation_order": sim_order,
            "state_hash": medal.candidate_value_record(child_copy)["state_hash"],
            "immediate_reward": reward,
            "action": None if child_copy.action is None else asdict(child_copy.action),
        })

        masked_count = int((child_copy.state[0, self.prompt_len:] == self.mask_id).sum().item())
        trace_copy.append({
            "cycle": self.cycle,
            "stage": "evaluation",
            "search_model_calls": budget,
            "parent_id": parent_copy.node_id,
            "child_id": child_copy.node_id,
            "depth": child_copy.depth,
            "action": medal.action_record(child_copy.action, self.tokenizer),
            "entropy_before": parent_copy.analysis.total_entropy,
            "entropy_child": child_analysis.total_entropy,
            "entropy_grandchild": None,
            "rig_step1": rig_step1,
            "rig_step2": 0.0,
            "reward_lambda": self.reward_lambda,
            "reward_beta": self.reward_beta,
            "reward_mode": self.reward_mode,
            "common_mask_count": common_mask_count,
            "lookahead_available": False,
            "lookahead_status": "budget_boundary_fallback",
            "reward": reward,
            "immediate_gain": reward,
            "masked_count": masked_count,
            "mean_token_entropy": child_analysis.total_entropy / max(1, masked_count),
            "mean_confidence_margin": child_analysis.mean_margin,
            "signal_nfe": budget,
        })

        recipients = [(child_copy.node_id, "node"), (parent_copy.node_id, "GEN")]
        recipients += [(n.node_id, "node") for n in path_copy[1:]]
        recipients += [(n.node_id, "CONT") for n in path_copy[:-1]]
        posterior_updates = [
            {"node_id": node_id, "model": name, **posteriors_copy[node_id][name].update(reward)}
            for node_id, name in recipients
        ]
        visits = medal.backpropagate(child_copy, reward)

        trace_copy.append({
            "cycle": self.cycle,
            "stage": "backpropagation",
            "search_model_calls": budget,
            "child_id": child_copy.node_id,
            "reward": reward,
            "posterior_updates": posterior_updates,
            "updates": visits,
        })

        evaluated_copy = [n for n in nodes_copy if n.evaluation_order is not None]
        root_actions = len(self.root.analysis.actions) if self.root.analysis else 0
        snapshot = _build_snapshot_from_evaluated(
            evaluated_copy,
            trace_copy,
            budget=budget,
            actual_model_calls=budget,
            simulations_completed=sim_order,
            sim_cap_multiplier=self.sim_cap_multiplier,
            init_tokens=self.config.init_tokens,
            num_candidates=self.config.num_candidates,
            root_action_count=root_actions,
            stop_reason="model_call_budget",
        )
        snapshot.update(
            posteriors=posteriors_copy,
            remaining_actions=remaining_copy,
            decision_history=decisions_copy,
        )
        return snapshot

    def _check_checkpoint(self, stop_reason: Optional[str] = None):
        if not self.checkpoint_budgets:
            return
        for b in self.checkpoint_budgets:
            if b in self.checkpoints:
                continue
            if self.model_calls >= b:
                reason = stop_reason or "model_call_budget"
            elif self.simulations_completed >= self.sim_cap_multiplier * b:
                reason = stop_reason or "simulation_budget"
            else:
                continue
            self.checkpoints[b] = self._snapshot_checkpoint(b, stop_reason=reason)
            self._emit(
                "search_checkpoint",
                budget=b,
                model_calls=self.checkpoints[b]["model_calls"],
                simulations_completed=self.simulations_completed,
                stop_reason=reason,
                selected_node_id=self.checkpoints[b]["selected_node_id"],
                selected_depth=self.checkpoints[b]["selected_depth"],
            )

    def next_evaluation(self) -> Optional[EvaluationRequest]:
        while not self.done:
            if self._pending is not None:
                return self._pending

            # 1. Root initialization
            if self.root.analysis is None:
                self.cycle += 1
                self._root_cycle = True
                self._pending = EvaluationRequest(self.root, None, [], [], "tree_analysis")
                return self._pending

            if self._at_cap() or self._at_simulation_cap():
                self._finish("model_call_budget" if self._at_cap() else "simulation_budget")
                return None

            # 2. Outstanding grandchild evaluation from Phase 1
            if self._pending_grandchild is not None:
                grandchild = self._pending_grandchild["grandchild"]
                g_key = medal._state_key(grandchild.state)
                if g_key in self.cached_analyses:
                    g_analysis = self.cached_analyses[g_key]
                    self._resolve_child_reward(g_analysis, lookahead_status="cached_grandchild")
                    continue

                self._pending = EvaluationRequest(
                    grandchild,
                    self._pending_grandchild["child"],
                    self._pending_grandchild["path"],
                    self._pending_grandchild["decisions"],
                    "simulation",
                )
                return self._pending

            if not getattr(self, "_root_cycle", False) and self._round is None:
                self.cycle += 1
            self._root_cycle = False

            # 3. Frontier traversal & decision
            with self._counted.phase("selection"):
                parent, path, decisions = self._select()
            if parent is None:
                self._finish("frontier_exhausted")
                return None

            self._emit("selection", node_id=parent.node_id, path=[n.node_id for n in path], decisions=decisions)

            with self._counted.phase("expansion"):
                action = self.remaining_actions[parent.node_id].pop(0)
                if action.token_id in self.search_excluded_token_ids:
                    raise RuntimeError("Excluded termination token reached search expansion.")
                parent.expanded = True
                child = medal.expand_node(
                    parent, action, len(self.nodes), self.prompt_len, self.mask_id, self.config.init_tokens
                )
                self.nodes.append(child)

            self._emit(
                "expansion",
                parent_id=parent.node_id,
                child_id=child.node_id,
                depth=child.depth,
                action=medal.action_record(action, self.tokenizer),
            )
            self.simulations_started += 1
            for order, decision in enumerate(decisions if self.config.search_policy == "gaussian" else []):
                self._emit(
                    "adaptive_decision",
                    traversal_id=self.simulations_started,
                    decision_order=order,
                    decision_node_id=decision["node_id"],
                    decision_type=decision["choice"],
                    child_id=child.node_id,
                    **decision,
                )

            c_key = medal._state_key(child.state)
            if c_key in self.cached_analyses:
                child_analysis = self.cached_analyses[c_key]
                self._prepare_grandchild_or_finalize(child, parent, path, decisions, child_analysis)
                continue

            self._pending = EvaluationRequest(child, parent, path, decisions, "simulation")
            return self._pending
        return None

    def _prepare_grandchild_or_finalize(
        self,
        child: medal.Node,
        parent: medal.Node,
        path: list,
        decisions: list,
        child_analysis: medal.StateAnalysis,
    ):
        self.cached_analyses[medal._state_key(child.state)] = child_analysis
        child.analysis = child_analysis

        remaining_masks = int((child.state[0, self.prompt_len:] == self.mask_id).sum().item())

        # Case A: Terminal child (0 remaining masks -> H(child) = 0, second-step gain = 0, |M(g*)| = 0 -> CMER = 0)
        if remaining_masks == 0:
            reward, common_mask_count = self._compute_two_step_reward(
                parent.analysis, child_analysis, None
            )
            self._apply_evaluation_result(
                child,
                parent,
                path,
                decisions,
                child_analysis,
                reward,
                grandchild_entropy=0.0,
                lookahead_available=True,
                lookahead_status="terminal_child",
                common_mask_count=common_mask_count,
            )
            return

        # Case B: Non-terminal child with no legal actions -> fallback to reward-consistent 1-step reward
        if not child_analysis.actions:
            reward, common_mask_count = self._compute_one_step_fallback(
                parent.analysis, child_analysis, child.state
            )
            self._apply_evaluation_result(
                child,
                parent,
                path,
                decisions,
                child_analysis,
                reward,
                grandchild_entropy=None,
                lookahead_available=False,
                lookahead_status="no_legal_actions",
                common_mask_count=common_mask_count,
            )
            return

        # Greedy second-step action (forced GEN on new child; does NOT update child's GEN/CONT posteriors)
        greedy_action = child_analysis.actions[0]
        grandchild = medal.expand_node(
            child, greedy_action, -1, self.prompt_len, self.mask_id, self.config.init_tokens + 1
        )
        child.children.pop()

        g_key = medal._state_key(grandchild.state)
        g_remaining_masks = int((grandchild.state[0, self.prompt_len:] == self.mask_id).sum().item())
        if g_remaining_masks == 0 and g_key not in self.cached_analyses:
            zero_entropies = torch.zeros(grandchild.state.shape[1] - self.prompt_len)
            self.cached_analyses[g_key] = medal.StateAnalysis(zero_entropies, (), g_key, 0.0)
        if g_remaining_masks == 0 and self.reward_mode == "swd" and g_key not in self.cached_swd_states:
            self._cache_swd_from_distribution(g_key, torch.empty(0, dtype=torch.long), None)

        # Case C1: Grandchild is already cached (or terminal with 0 masks) -> resolve with 0 extra NFEs
        if g_key in self.cached_analyses:
            g_analysis = self.cached_analyses[g_key]
            grandchild.analysis = g_analysis
            reward, common_mask_count = self._compute_two_step_reward(
                parent.analysis, child_analysis, g_analysis, grandchild.state
            )
            self._apply_evaluation_result(
                child,
                parent,
                path,
                decisions,
                child_analysis,
                reward,
                grandchild_entropy=g_analysis.total_entropy,
                lookahead_available=True,
                lookahead_status="cached_grandchild",
                common_mask_count=common_mask_count,
            )
            return

        # Case C2: Grandchild is uncached (requires 1 new model forward pass)
        if self._at_cap():
            reward, common_mask_count = self._compute_one_step_fallback(
                parent.analysis, child_analysis, child.state
            )
            self._apply_evaluation_result(
                child,
                parent,
                path,
                decisions,
                child_analysis,
                reward,
                grandchild_entropy=None,
                lookahead_available=False,
                lookahead_status="budget_boundary_fallback",
                common_mask_count=common_mask_count,
            )
            return

        if self.checkpoint_budgets:
            for b in self.checkpoint_budgets:
                if b not in self.checkpoints and self.model_calls == b:
                    self.checkpoints[b] = self._snapshot_boundary_checkpoint(
                        b, child, parent, path, decisions, child_analysis
                    )
                    self._emit(
                        "search_checkpoint",
                        budget=b,
                        model_calls=b,
                        simulations_completed=self.simulations_completed + 1,
                        stop_reason="model_call_budget",
                        boundary_hypothetical=True,
                        selected_node_id=self.checkpoints[b]["selected_node_id"],
                        selected_depth=self.checkpoints[b]["selected_depth"],
                    )

        self._pending_grandchild = {
            "child": child,
            "parent": parent,
            "path": path,
            "decisions": decisions,
            "child_analysis": child_analysis,
            "grandchild": grandchild,
            "greedy_action": greedy_action,
        }

    def _resolve_child_reward(
        self, g_analysis: medal.StateAnalysis, *, lookahead_status: str = "evaluated_grandchild"
    ):
        data = self._pending_grandchild
        child, parent = data["child"], data["parent"]
        grandchild = data["grandchild"]
        child_analysis = data["child_analysis"]
        path, decisions = data["path"], data["decisions"]
        self._pending_grandchild = None

        grandchild.analysis = g_analysis
        self.cached_analyses[medal._state_key(grandchild.state)] = g_analysis

        grandchild_entropy = g_analysis.total_entropy
        reward, common_mask_count = self._compute_two_step_reward(
            parent.analysis, child_analysis, g_analysis, grandchild.state
        )
        self._apply_evaluation_result(
            child,
            parent,
            path,
            decisions,
            child_analysis,
            reward,
            grandchild_entropy=grandchild_entropy,
            lookahead_available=True,
            lookahead_status=lookahead_status,
            common_mask_count=common_mask_count,
        )

    def _apply_evaluation_result(
        self,
        child: medal.Node,
        parent: medal.Node,
        path: list,
        decisions: list,
        child_analysis: medal.StateAnalysis,
        reward: float,
        grandchild_entropy: Optional[float],
        *,
        lookahead_available: bool = True,
        lookahead_status: str = "evaluated_grandchild",
        common_mask_count: Optional[int] = None,
    ):
        child.analysis = child_analysis
        self.cached_analyses[medal._state_key(child.state)] = child_analysis
        child.immediate_reward = reward
        self._register(child)

        self.simulations_completed += 1
        self._record_evaluated(child)

        rig_step1 = medal.information_gain(
            parent.analysis.total_entropy, child_analysis.total_entropy, self.config.epsilon
        )
        rig_step2 = (
            medal.information_gain(child_analysis.total_entropy, grandchild_entropy, self.config.epsilon)
            if (lookahead_available and grandchild_entropy is not None)
            else 0.0
        )
        masked_count = int((child.state[0, self.prompt_len:] == self.mask_id).sum().item())

        evaluation = {
            "entropy_before": parent.analysis.total_entropy,
            "entropy_child": child_analysis.total_entropy,
            "entropy_grandchild": grandchild_entropy,
            "rig_step1": rig_step1,
            "rig_step2": rig_step2,
            "reward_lambda": self.reward_lambda,
            "reward_beta": self.reward_beta,
            "reward_mode": self.reward_mode,
            "common_mask_count": common_mask_count,
            "lookahead_available": lookahead_available,
            "lookahead_status": lookahead_status,
            "reward": reward,
            "immediate_gain": reward,
            "masked_count": masked_count,
            "mean_token_entropy": child_analysis.total_entropy / max(1, masked_count),
            "mean_confidence_margin": child_analysis.mean_margin,
            "signal_nfe": self.model_calls,
        }
        self._emit(
            "evaluation",
            parent_id=parent.node_id,
            child_id=child.node_id,
            depth=child.depth,
            action=medal.action_record(child.action, self.tokenizer),
            **evaluation,
        )

        with self._counted.phase("backpropagation"):
            posterior_updates, visits = self._backup(child, parent, path, reward)
        self._emit(
            "backpropagation",
            child_id=child.node_id,
            reward=reward,
            posterior_updates=posterior_updates,
            updates=visits,
        )
        self._check_checkpoint()

        if child.depth == self.config.init_tokens:
            key = medal._state_key(child.state)
            duplicate = key in self.candidates
            self.candidates.setdefault(key, child)
            self._emit(
                "candidate",
                node_id=child.node_id,
                duplicate=duplicate,
                candidate_count=len(self.candidates),
                score=child.mean_reward,
            )

        record = {
            "iteration": self.simulations_completed,
            "parent_id": parent.node_id,
            "child_id": child.node_id,
            "path": [n.node_id for n in path],
            "decisions": decisions,
            "reward": reward,
            "posterior_updates": posterior_updates,
            "model_calls": self.model_calls,
            "sequence_nfes": self._counted.sequence_nfes,
        }
        self.history.append(record)
        self._emit("iteration", **record)

        self._pending = None
        if self._at_cap() or self._at_simulation_cap():
            self._finish("model_call_budget" if self._at_cap() else "simulation_budget")
            return
        if self.config.search_policy == "gaussian":
            self._close_exhausted(child)
        elif not self.remaining_actions[parent.node_id]:
            self._close_exhausted(parent)
            self._round = None
        if self.root.closed:
            self._finish("frontier_exhausted")

    def submit_evaluation(
        self,
        request: EvaluationRequest,
        *,
        logits: Optional[torch.Tensor] = None,
        analysis: Optional[medal.StateAnalysis] = None,
    ):
        analysis = self._resolve_evaluation_payload(request, logits=logits, analysis=analysis)

        child, parent = request.node, request.parent

        if parent is None:
            self.root.analysis = analysis
            self.cached_analyses[medal._state_key(self.root.state)] = analysis
            self._register(self.root)
            self._record_evaluated(self.root)
            self._emit(
                "initialization",
                node_id=0,
                entropy=analysis.total_entropy,
                actions=[medal.action_record(a, self.tokenizer) for a in analysis.actions],
                policy=self.config.search_policy,
                priors=asdict(self.config.new_posterior()),
            )
            self._pending = None
            self._check_checkpoint()
            return

        if self._pending_grandchild is not None and request.node is self._pending_grandchild["grandchild"]:
            self._resolve_child_reward(analysis, lookahead_status="evaluated_grandchild")
            return

        self._prepare_grandchild_or_finalize(child, parent, request.path, request.decisions, analysis)
        self._pending = None

    @torch.inference_mode()
    def step(self):
        start = len(self.trace)
        while not self.done:
            request = self.next_evaluation()
            if request is None:
                break
            request.attempted = True
            with self._counted.phase(request.phase):
                if self.reward_mode == "swd":
                    analysis, positions, probabilities, _ = medal._distribution(
                        self._counted,
                        request.node.state,
                        self.prompt_len,
                        self.mask_id,
                        self.config,
                        search_excluded_token_ids=self.search_excluded_token_ids,
                    )
                    self._cache_swd_from_distribution(analysis.state_key, positions, probabilities)
                    del probabilities
                else:
                    analysis = medal.analyze_state(
                        self._counted,
                        request.node.state,
                        self.prompt_len,
                        self.mask_id,
                        self.config,
                        search_excluded_token_ids=self.search_excluded_token_ids,
                    )
            self.submit_evaluation(request, logits=None, analysis=analysis)
            if request.parent is not None and self._pending_grandchild is None:
                break
        return self.trace[start:]
