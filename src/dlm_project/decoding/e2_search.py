"""Two-step lookahead Fractional Information Gain (FIG) MCTS search for Experiment 2 (E2).

Implements E2MEDALSearch and E2ABMEDALSearch:
For each child state s_i, selects highest-confidence action a*_i to construct grandchild g_i,
evaluating both states to compute the 2-step lookahead reward:
    R_FIG(s, s_i) = (H(s) - H(g_i)) / max(H(s), epsilon)

Evaluated grandchild states are cached in `cached_analyses` and linked into the tree,
avoiding redundant model forwards when visited or continued.
"""

from collections import Counter
import copy
from dataclasses import asdict, dataclass
import math
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import torch

from dlm_project.decoding import mcts_decoding as medal
from dlm_project.decoding.ab_medal import ABMEDALConfig, ABMEDALSearch, EvaluationRequest
from dlm_project.decoding.medal_batch import CooperativeMEDALSearch


def compute_fig_reward(parent_entropy: float, grandchild_entropy: float, epsilon: float = 1e-8) -> float:
    """Compute 2-step lookahead Fractional Information Gain reward."""
    denom = max(parent_entropy, epsilon)
    gain = (parent_entropy - grandchild_entropy) / denom
    if not math.isfinite(gain):
        raise ValueError(f"Non-finite FIG reward: parent={parent_entropy}, grandchild={grandchild_entropy}")
    return gain


class E2MEDALSearch(CooperativeMEDALSearch):
    """Two-step lookahead MEDAL search with cooperative batching and state caching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_analyses: Dict[Tuple[int, ...], medal.StateAnalysis] = {}
        self._pending_grandchild: Optional[Dict[str, Any]] = None

    def next_evaluation(self) -> Optional[EvaluationRequest]:
        if self._pending is not None:
            return self._pending
        if self.done:
            return None

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
                # Cache hit on grandchild: resolve immediately without extra NFE
                g_analysis = self.cached_analyses[g_key]
                self._resolve_child_reward(g_analysis)
                return self.next_evaluation()

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
            return self.next_evaluation()

        self._pending = EvaluationRequest(child, parent, [], [], "simulation")
        return self._pending

    def _prepare_grandchild_or_finalize(self, child: medal.Node, parent: medal.Node, child_analysis: medal.StateAnalysis):
        """Prepare grandchild lookahead or finalize child immediately if fully committed."""
        self.cached_analyses[medal._state_key(child.state)] = child_analysis
        child.analysis = child_analysis

        remaining_masks = int((child.state[0, self.prompt_len:] == self.mask_id).sum().item())
        if remaining_masks == 0 or not child_analysis.actions:
            # Child has 0 masks: H(g_i) = 0.0, FIG reward is (H(s) - 0) / max(H(s), eps)
            reward = compute_fig_reward(parent.analysis.total_entropy, 0.0, self.config.epsilon)
            self._apply_evaluation_result(child, parent, child_analysis, reward, 0.0)
            return

        # Choose highest-confidence action from child's action set
        greedy_action = child_analysis.actions[0]
        # Construct grandchild lookahead state for simulation (do not link into tree children)
        grandchild = medal.expand_node(
            child, greedy_action, -1, self.prompt_len, self.mask_id, self.config.init_tokens + 1
        )
        child.children.pop()

        # Store pending lookahead data
        self._pending_grandchild = {
            "child": child,
            "parent": parent,
            "child_analysis": child_analysis,
            "grandchild": grandchild,
            "greedy_action": greedy_action,
        }

    def _resolve_child_reward(self, g_analysis: medal.StateAnalysis):
        """Apply grandchild analysis to compute R_FIG and backpropagate to child."""
        data = self._pending_grandchild
        child, parent = data["child"], data["parent"]
        grandchild = data["grandchild"]
        child_analysis = data["child_analysis"]
        self._pending_grandchild = None

        grandchild.analysis = g_analysis
        self.cached_analyses[medal._state_key(grandchild.state)] = g_analysis

        grandchild_entropy = g_analysis.total_entropy
        reward = compute_fig_reward(parent.analysis.total_entropy, grandchild_entropy, self.config.epsilon)
        self._apply_evaluation_result(child, parent, child_analysis, reward, grandchild_entropy)

    def _apply_evaluation_result(
        self, child: medal.Node, parent: medal.Node, child_analysis: medal.StateAnalysis, reward: float, grandchild_entropy: float
    ):
        """Commit child reward, update statistics, check checkpoints and finalize."""
        child.analysis = child_analysis
        self.cached_analyses[medal._state_key(child.state)] = child_analysis
        child.immediate_reward = reward
        self.simulations_completed += 1
        self._record_evaluated(child)

        evaluation = {
            "entropy_before": parent.analysis.total_entropy,
            "entropy_child": child_analysis.total_entropy,
            "entropy_grandchild": grandchild_entropy,
            "reward": reward,
            "immediate_gain": reward,
            "masked_count": int((child.state[0, self.prompt_len:] == self.mask_id).sum().item()),
            "mean_token_entropy": child_analysis.total_entropy / max(1, (child.state[0, self.prompt_len:] == self.mask_id).sum().item()),
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

    def submit_evaluation(self, request: EvaluationRequest, *, logits: Optional[torch.Tensor] = None, analysis: Optional[medal.StateAnalysis] = None):
        if request is not self._pending or not request.attempted:
            raise ValueError("Expected the current dispatched evaluation request.")
        if (logits is None) == (analysis is None):
            raise ValueError("Supply exactly one of logits and analysis.")

        if analysis is None:
            analysis = medal.distribution_from_logits(
                logits,
                request.node.state,
                self.prompt_len,
                self.mask_id,
                self.config,
                search_excluded_token_ids=self.search_excluded_token_ids,
            )[0]

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
            self._resolve_child_reward(analysis)
            return

        # Case 3: Child evaluation (Phase 1)
        # Check budget boundary: if budget is exhausted right now, fallback to 1-step immediate reward
        if self._at_cap():
            reward = medal.information_gain(parent.analysis.total_entropy, analysis.total_entropy, self.config.epsilon)
            self._apply_evaluation_result(child, parent, analysis, reward, analysis.total_entropy)
            return

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


class E2ABMEDALSearch(ABMEDALSearch):
    """Two-step lookahead AB-MEDAL search with greedy continuation lookahead and caching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_analyses: Dict[Tuple[int, ...], medal.StateAnalysis] = {}
        self._pending_grandchild: Optional[Dict[str, Any]] = None

    def next_evaluation(self) -> Optional[EvaluationRequest]:
        if self._pending is not None:
            return self._pending
        if self.done:
            return None

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
                self._resolve_child_reward(g_analysis)
                return self.next_evaluation()

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
            return self.next_evaluation()

        self._pending = EvaluationRequest(child, parent, path, decisions, "simulation")
        return self._pending

    def _prepare_grandchild_or_finalize(
        self, child: medal.Node, parent: medal.Node, path: list, decisions: list, child_analysis: medal.StateAnalysis
    ):
        self.cached_analyses[medal._state_key(child.state)] = child_analysis
        child.analysis = child_analysis

        remaining_masks = int((child.state[0, self.prompt_len:] == self.mask_id).sum().item())
        if remaining_masks == 0 or not child_analysis.actions:
            reward = compute_fig_reward(parent.analysis.total_entropy, 0.0, self.config.epsilon)
            self._apply_evaluation_result(child, parent, path, decisions, child_analysis, reward, 0.0)
            return

        greedy_action = child_analysis.actions[0]
        grandchild = medal.expand_node(
            child, greedy_action, -1, self.prompt_len, self.mask_id, self.config.init_tokens + 1
        )
        child.children.pop()

        self._pending_grandchild = {
            "child": child,
            "parent": parent,
            "path": path,
            "decisions": decisions,
            "child_analysis": child_analysis,
            "grandchild": grandchild,
            "greedy_action": greedy_action,
        }

    def _resolve_child_reward(self, g_analysis: medal.StateAnalysis):
        data = self._pending_grandchild
        child, parent = data["child"], data["parent"]
        grandchild = data["grandchild"]
        child_analysis = data["child_analysis"]
        path, decisions = data["path"], data["decisions"]
        self._pending_grandchild = None

        grandchild.analysis = g_analysis
        self.cached_analyses[medal._state_key(grandchild.state)] = g_analysis

        grandchild_entropy = g_analysis.total_entropy
        reward = compute_fig_reward(parent.analysis.total_entropy, grandchild_entropy, self.config.epsilon)
        self._apply_evaluation_result(child, parent, path, decisions, child_analysis, reward, grandchild_entropy)

    def _apply_evaluation_result(
        self,
        child: medal.Node,
        parent: medal.Node,
        path: list,
        decisions: list,
        child_analysis: medal.StateAnalysis,
        reward: float,
        grandchild_entropy: float,
    ):
        child.analysis = child_analysis
        self.cached_analyses[medal._state_key(child.state)] = child_analysis
        child.immediate_reward = reward
        self._register(child)

        self.simulations_completed += 1
        self._record_evaluated(child)

        evaluation = {
            "entropy_before": parent.analysis.total_entropy,
            "entropy_child": child_analysis.total_entropy,
            "entropy_grandchild": grandchild_entropy,
            "reward": reward,
            "immediate_gain": reward,
            "masked_count": int((child.state[0, self.prompt_len:] == self.mask_id).sum().item()),
            "mean_token_entropy": child_analysis.total_entropy / max(1, (child.state[0, self.prompt_len:] == self.mask_id).sum().item()),
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

    def submit_evaluation(self, request: EvaluationRequest, *, logits: Optional[torch.Tensor] = None, analysis: Optional[medal.StateAnalysis] = None):
        if request is not self._pending or not request.attempted:
            raise ValueError("Expected the current dispatched evaluation request.")
        if (logits is None) == (analysis is None):
            raise ValueError("Supply exactly one of logits and analysis.")

        if analysis is None:
            analysis = medal.distribution_from_logits(
                logits,
                request.node.state,
                self.prompt_len,
                self.mask_id,
                self.config,
                search_excluded_token_ids=self.search_excluded_token_ids,
            )[0]

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
            self._resolve_child_reward(analysis)
            return

        if self._at_cap():
            reward = medal.information_gain(parent.analysis.total_entropy, analysis.total_entropy, self.config.epsilon)
            self._apply_evaluation_result(
                child, parent, request.path, request.decisions, analysis, reward, analysis.total_entropy
            )
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
