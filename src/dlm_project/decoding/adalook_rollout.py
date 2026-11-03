"""AdaLook-inspired MEDAL evaluation, not the paper's block/ETE decoder.

Temporary hypotheses never become outer tree nodes. A complete sibling group is
scored before one backup per child. Every extra round must fit the next budget
checkpoint. Serial and cross-question batched execution drive the same generator.
The global masked suffix substitutes for ETE's block frontier. Restart candidates
use MEDAL's retained actions. Pruned outer actions keep their last valid score.
"""

from dataclasses import dataclass
import math
from statistics import pvariance

import torch


def score(probability, mass, config):
    return (config.adalook_alpha * math.log(max(probability, config.epsilon))
            + math.log(max(mass, config.epsilon)))


def eligible(analysis, config, mask_id, excluded):
    if analysis.positions is None:
        return torch.zeros(0, dtype=torch.bool)
    keep = (analysis.confidences >= config.adalook_confidence_threshold) & (analysis.proposed_tokens != mask_id)
    for token in excluded:
        keep &= analysis.proposed_tokens != token
    return keep


@dataclass
class Request:
    node: object
    parent: object
    phase: str
    attempted: bool = False


@dataclass
class Hypothesis:
    origin: object
    state: torch.Tensor
    analysis: object
    committed_mass: float = 0.0

    def mass(self):
        return self.committed_mass + self.analysis.confidence_mass


class ConfidenceRefinement:
    """Shared temporary-hypothesis evaluator; never selects or backs up tree nodes."""

    def _boundary(self):
        s = self.search
        limits = [b for b in s.checkpoint_budgets if b > s.model_calls]
        if s.config.max_model_calls is not None:
            limits.append(s.config.max_model_calls)
        return min(limits, default=math.inf)

    def _trigger(self, h):
        a, cfg = h.analysis, self.search.config
        return (a.confidences is not None and len(a.confidences) >= cfg.adalook_min_masks
                and a.confidences.double().mean().item() < cfg.adalook_trigger_threshold)

    def _refine(self, children, analyses):
        from .mcts_decoding import Node
        s, cfg = self.search, self.search.config
        active = [Hypothesis(n, n.state.clone(), a) for n, a in zip(children, analyses)]
        values = {h.origin.node_id: score(h.origin.action.probability, h.mass(), cfg) for h in active}
        depth, extra, reason = 1, 0, 'depth_limit'
        boundary = self._boundary()
        branch_events = []
        while depth < cfg.adalook_max_depth:
            scores = [score(h.origin.action.probability, h.mass(), cfg) for h in active]
            variance = pvariance(scores) if len(scores) > 1 else 0.0
            s._emit('adalook_gate', rollout_depth=depth, score_variance=variance,
                    threshold=cfg.adalook_variance_threshold, scores=scores,
                    origin_ids=[h.origin.node_id for h in active])
            if len(active) <= 1:
                reason = 'single_branch'
                break
            if variance >= cfg.adalook_variance_threshold:
                reason = 'score_variance'
                break
            triggers = [self._trigger(h) for h in active]
            plans = []
            if all(triggers):
                best = max(active, key=lambda h: score(h.origin.action.probability, h.mass(), cfg)
                    + math.log(max(h.analysis.confidences.double().mean().item(), cfg.epsilon)))
                for action in best.analysis.actions:
                    state = best.state.clone()
                    state[0, s.prompt_len + action.output_position] = action.token_id
                    # Restart from the same evaluated state; no duplicate forward
                    # and no invented confidence mass for exploratory commitments.
                    plans.append((best, state, best.committed_mass))
                kind = 'all_retrigger'
            else:
                stable = [h for h, trigger in zip(active, triggers) if not trigger]
                kind = 'mixed_prune' if any(triggers) else 'stable'
                if len(stable) == 1:
                    reason = 'single_stable_branch'
                    branch_events.append(kind)
                    s._emit('adalook_branch', action=kind, retained_origins=[stable[0].origin.node_id])
                    break
                for h in stable:
                    keep = eligible(h.analysis, cfg, s.mask_id, s.search_excluded_token_ids)
                    if not keep.any():
                        continue
                    state = h.state.clone()
                    state[0, s.prompt_len + h.analysis.positions[keep]] = h.analysis.proposed_tokens[keep]
                    mass = h.committed_mass + h.analysis.confidences[keep].double().sum().item()
                    plans.append((h, state, mass))
            branch_events.append(kind)
            s._emit('adalook_branch', action=kind, retained_origins=[h.origin.node_id for h, _, _ in plans])
            if not plans:
                reason = 'no_progress'
                break
            required = sum(bool((state[0, s.prompt_len:] == s.mask_id).any()) for _, state, _ in plans)
            if s.model_calls + required > boundary:
                reason = 'round_budget'
                break
            next_active = []
            for h, state, mass in plans:
                if (state[0, s.prompt_len:] == s.mask_id).any():
                    temporary = Node(-1, state)
                    analysis = yield Request(temporary, h.origin, 'adalook_rollout')
                    extra += 1
                else:
                    from .mcts_decoding import StateAnalysis
                    analysis = StateAnalysis(torch.zeros(state.shape[1] - s.prompt_len), (), tuple(state[0].tolist()))
                next_active.append(Hypothesis(h.origin, state, analysis, mass))
            active = next_active
            depth += 1
            # Several restart alternatives can belong to one outer action.
            for origin in {h.origin.node_id for h in active}:
                values[origin] = max(score(h.origin.action.probability, h.mass(), cfg)
                                     for h in active if h.origin.node_id == origin)
            s._emit('adalook_round', rollout_depth=depth, extra_nfes=extra,
                    origin_scores={str(k): v for k, v in values.items()})
            if s.model_calls == boundary:
                reason = 'checkpoint_or_budget'
                break
        final_scores = list(values.values())
        metadata = dict(rollout_depth=depth, extra_nfes=extra, stop_reason=reason,
                        score_variance=pvariance(final_scores) if len(final_scores) > 1 else 0.0,
                        branch_actions=branch_events)
        return values, metadata


class AdaptiveDriver(ConfidenceRefinement):
    def __init__(self, search):
        self.search = search
        self.pending = None
        self.generator = self._run()
        self.exhausted = False

    def next_evaluation(self):
        if self.pending is None and not self.exhausted:
            self._advance()
        return self.pending

    def _advance(self, analysis=None):
        try:
            self.pending = self.generator.send(analysis)
        except StopIteration:
            self.pending = None
            self.exhausted = True

    def record_external_forward(self, request, invocation_id, batch_size):
        from .mcts_decoding import BudgetExhausted
        s = self.search
        if request is not self.pending or request.attempted:
            raise ValueError("Expected current undispatched AdaLook request.")
        if s._at_cap():
            raise BudgetExhausted("Search forward cap reached.")
        request.attempted = True
        s._counted.calls += 1
        s._counted.counts[request.phase] += 1
        s._emit('forward', phase=request.phase, invocation_id=invocation_id,
                batch_size=batch_size, node_id=request.node.node_id,
                status='attempted', **s._counted.costs())

    def submit_evaluation(self, request, logits):
        from .mcts_decoding import distribution_from_logits
        s = self.search
        if request is not self.pending or not request.attempted:
            raise ValueError("Expected current dispatched AdaLook request.")
        analysis = distribution_from_logits(logits, request.node.state, s.prompt_len,
            s.mask_id, s.config, search_excluded_token_ids=s.search_excluded_token_ids)[0]
        s._emit('adalook_forward_result', phase=request.phase, node_id=request.node.node_id)
        self.pending = None
        self._advance(analysis)

    def _finish_group(self, parent, children, analyses, values, metadata):
        from .mcts_decoding import apply_child_analysis, backpropagate, action_record, _state_key
        s = self.search
        s._emit('adalook_group', parent_id=parent.node_id, child_ids=[n.node_id for n in children], **metadata)
        for child, analysis in zip(children, analyses):
            evaluation = apply_child_analysis(parent, child, analysis, s.prompt_len, s.mask_id, s.config,
                                              signal_nfe=s.model_calls)
            child.immediate_reward = values[child.node_id]
            evaluation.update(reward=child.immediate_reward, **{f'adalook_{k}': v for k, v in metadata.items()})
            s.simulations_completed += 1
            s._record_evaluated(child)
            s._emit('evaluation', parent_id=parent.node_id, child_id=child.node_id,
                    depth=child.depth, action=action_record(child.action, s.tokenizer), **evaluation)
            updates = backpropagate(child, child.immediate_reward)
            s._emit('backpropagation', child_id=child.node_id, reward=child.immediate_reward, updates=updates)
            if child.depth == s.config.init_tokens:
                key = _state_key(child.state)
                duplicate = key in s.candidates
                s.candidates.setdefault(key, child)
                s._emit('candidate', node_id=child.node_id, duplicate=duplicate,
                        candidate_count=len(s.candidates), score=child.immediate_reward)

    def _run(self):
        from .mcts_decoding import select_node, expand_node, action_record
        s = self.search
        s.cycle += 1
        s.root.analysis = yield Request(s.root, None, 'tree_analysis')
        s._record_evaluated(s.root)
        s._emit('initialization', node_id=0, entropy=s.root.analysis.total_entropy,
                actions=[action_record(a, s.tokenizer) for a in s.root.analysis.actions],
                reward_mode=s.config.reward_mode, assumptions=s._assumptions())
        s._check_checkpoint()
        while not s.done:
            if s._at_cap() or s._at_simulation_cap():
                s._finish('model_call_budget' if s._at_cap() else 'simulation_budget')
                break
            parent, decisions = select_node(s.root, s.config.exploration_c)
            if parent is None:
                s._finish('frontier_exhausted')
                break
            s._emit('selection', node_id=parent.node_id,
                    path=[0] + [d['chosen_id'] for d in decisions], decisions=decisions)
            parent.expanded = True
            remaining = list(parent.analysis.actions)
            while remaining and not s._at_cap() and not s._at_simulation_cap():
                boundary = self._boundary()
                children, analyses = [], []
                while remaining and s.model_calls < boundary:
                    if s.config.max_simulations is not None and s.simulations_started >= s.config.max_simulations:
                        break
                    action = remaining.pop(0)
                    child = expand_node(parent, action, len(s.nodes), s.prompt_len, s.mask_id, s.config.init_tokens)
                    s.nodes.append(child)
                    s._emit('expansion', parent_id=parent.node_id, child_id=child.node_id,
                            depth=child.depth, action=action_record(action, s.tokenizer))
                    s.simulations_started += 1
                    s._emit('simulation_started', child_id=child.node_id, simulations_started=s.simulations_started)
                    analysis = yield Request(child, parent, 'simulation')
                    children.append(child)
                    analyses.append(analysis)
                if not children:
                    break
                # A checkpoint reached by base evaluations must be finalized now.
                if s.model_calls < boundary and not remaining:
                    values, metadata = yield from self._refine(children, analyses)
                else:
                    values = {n.node_id: score(n.action.probability, a.confidence_mass, s.config)
                              for n, a in zip(children, analyses)}
                    metadata = dict(rollout_depth=1, extra_nfes=0, stop_reason='base_budget',
                                    score_variance=pvariance(values.values()) if len(values) > 1 else 0.0,
                                    branch_actions=[])
                self._finish_group(parent, children, analyses, values, metadata)
                s._check_checkpoint()
            s._close_exhausted(parent)
            if s._at_cap() or s._at_simulation_cap():
                s._finish('model_call_budget' if s._at_cap() else 'simulation_budget')
            elif s.root.closed:
                s._finish('frontier_exhausted')
            s.cycle += 1
