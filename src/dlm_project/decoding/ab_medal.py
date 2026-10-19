"""Adaptive MEDAL with identical Student-t predictive models for GEN and CONT.

States, action filtering, immediate information gain and final candidate ranking
come from the repository's MEDAL reconstruction. There are no search rollouts.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
import math
from time import perf_counter

import numpy as np
import torch

from dlm_project.decoding import mcts_decoding as medal


@dataclass
class GaussianPosterior:
    """Normal--inverse-chi-squared parameters; m is the posterior location."""

    m: float = 0.0
    kappa: float = 1.0
    nu: float = 1.0
    tau2: float = 0.1

    def __post_init__(self):
        for name in ("m", "kappa", "nu", "tau2"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or (name != "m" and value <= 0)):
                raise ValueError(f"Invalid Gaussian posterior {name}: {value}.")

    @property
    def predictive_scale(self):
        scale = math.sqrt(self.tau2 * (1 + 1 / self.kappa))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Non-finite or non-positive predictive scale.")
        return scale

    def sample(self, rng):
        value = self.m + self.predictive_scale * rng.standard_t(self.nu)
        if not math.isfinite(value):
            raise ValueError("Non-finite posterior-predictive sample.")
        return float(value)

    def update(self, reward):
        if isinstance(reward, bool) or not math.isfinite(reward):
            raise ValueError("Posterior rewards must be finite numbers.")
        before = asdict(self)
        # All right-hand sides use the OLD parameters, including old m and nu.
        try:
            updated = GaussianPosterior(
                (self.kappa * self.m + reward) / (self.kappa + 1),
                self.kappa + 1,
                self.nu + 1,
                (self.nu * self.tau2 + self.kappa / (self.kappa + 1)
                 * (reward - self.m) ** 2) / (self.nu + 1),
            )
        except OverflowError as error:
            raise ValueError("Non-finite posterior update.") from error
        self.m, self.kappa, self.nu, self.tau2 = (
            updated.m, updated.kappa, updated.nu, updated.tau2)
        return {"before": before, "after": asdict(self), "reward": reward}


@dataclass(frozen=True)
class ABMEDALConfig(medal.MCTSConfig):
    prior_m: float = 0.0
    prior_kappa: float = 1.0
    prior_nu: float = 1.0
    prior_tau2: float = 0.1
    search_policy: str = "gaussian"

    def __post_init__(self):
        super().__post_init__()
        self.new_posterior()
        if self.search_policy not in ("gaussian", "medal_equivalent"):
            raise ValueError("search_policy must be gaussian or medal_equivalent.")
        if self.stop_mode not in ("candidates", "candidates_or_budget"):
            raise ValueError(
                "Adaptive MEDAL requires candidates or candidates_or_budget stop mode.")

    def new_posterior(self):
        return GaussianPosterior(self.prior_m, self.prior_kappa, self.prior_nu, self.prior_tau2)


class _SearchModel(medal._SearchModel):
    """Count sequences at the actual forward as well as model invocations."""

    def __init__(self, *args):
        super().__init__(*args)
        self.sequence_nfes = 0
        self.sequence_counts = dict.fromkeys(self.counts, 0)

    def __call__(self, ids, **kwargs):
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Adaptive MEDAL requires batch size 1.")
        if self.cap is not None and self.calls >= self.cap:
            raise medal.BudgetExhausted("Search forward cap reached.")
        self.sequence_nfes += ids.shape[0]
        self.sequence_counts[self.current_phase] += ids.shape[0]
        return super().__call__(ids, **kwargs)

    def costs(self):
        return {**super().costs(), "search_sequence_nfes": self.sequence_nfes,
                **{f"{phase}_sequence_nfes": n for phase, n in self.sequence_counts.items()}}


def action_path(node):
    path = []
    while node.parent is not None:
        path.append({"output_position": node.action.output_position, "token_id": node.action.token_id})
        node = node.parent
    return list(reversed(path))


def tree_statistics(search):
    """Cached structural metrics shared by both runners; no inference or draws."""
    evaluated = [n for n in search.nodes if n.immediate_reward is not None]
    child_counts = Counter(n.parent.node_id for n in evaluated)
    leaves = [n for n in evaluated if n.node_id not in child_counts]
    candidates = []
    for node in search.candidates.values():
        path = action_path(node)
        commitments = sorted((a["output_position"], a["token_id"]) for a in path)
        candidates.append({"node_id": node.node_id, "depth": node.depth, "action_path": path,
                           "commitment_set": [list(pair) for pair in commitments]})
    pairs = []
    for left, right in combinations(candidates, 2):
        common = 0
        for a, b in zip(left["action_path"], right["action_path"]):
            if a != b:
                break
            common += 1
        a, b = (set(map(tuple, row["commitment_set"])) for row in (left, right))
        pairs.append({"candidate_ids": [left["node_id"], right["node_id"]],
                      "shared_prefix_length": common, "jaccard_distance": 1 - len(a & b) / len(a | b)})
    shared_prefix = None
    if pairs:
        shared_prefix = min(pair["shared_prefix_length"] for pair in pairs)
    return {
        "expanded_parents": len(child_counts), "evaluated_children": len(evaluated),
        "unique_states": len({medal._state_key(n.state) for n in search.nodes}),
        "branch_depths": [{"node_id": n.node_id, "depth": n.depth} for n in leaves],
        "depth_histogram": dict(sorted(Counter(str(n.depth) for n in evaluated).items(), key=lambda p: int(p[0]))),
        "mean_evaluated_depth": sum(n.depth for n in evaluated) / len(evaluated) if evaluated else 0.0,
        "max_evaluated_depth": max((n.depth for n in evaluated), default=0),
        "branching_factors": dict(sorted(child_counts.items())),
        "branching_factor_distribution": dict(sorted(Counter(child_counts.values()).items())),
        "mean_branching_factor": len(evaluated) / len(child_counts) if child_counts else 0.0,
        "visited_once_fraction": sum(n.visits == 1 for n in evaluated) / len(evaluated) if evaluated else None,
        "candidates": candidates,
        "candidate_diversity": {"available": bool(pairs), "pairs": pairs,
                                "shared_prefix_length": shared_prefix,
                                "mean_jaccard_distance": sum(p["jaccard_distance"] for p in pairs) / len(pairs) if pairs else None},
    }


def tree_record(search):
    """Serialize the full tree, including separate adaptive metadata if present."""
    rows = []
    for node in search.nodes:
        row = {"node_id": node.node_id, "parent_id": node.parent.node_id if node.parent else None,
               "children": [c.node_id for c in node.children], "depth": node.depth,
               "visits": node.visits, "reward_sum": node.reward_sum, "mean_reward": node.mean_reward,
               "immediate_reward": node.immediate_reward, "expanded": node.expanded, "closed": node.closed,
               "candidate": node in search.candidates.values(),
               "action": medal.action_record(node.action, search.tokenizer) if node.action else None,
               "action_path": action_path(node),
               "state": medal.state_record(node.state, search.prompt_len, search.mask_id, search.tokenizer),
               "entropy": node.analysis.total_entropy if node.analysis else None,
               "cached_actions": [medal.action_record(a, search.tokenizer) for a in node.analysis.actions] if node.analysis else []}
        if hasattr(search, "posteriors") and node.node_id in search.posteriors:
            row["posteriors"] = {name: asdict(p) for name, p in search.posteriors[node.node_id].items()}
            row["remaining_actions"] = [medal.action_record(a, search.tokenizer) for a in search.remaining_actions[node.node_id]]
            row["decisions"] = search.decision_history[node.node_id]
        rows.append(row)
    return {"nodes": rows, "statistics": tree_statistics(search)}


class ABMEDALSearch(medal.MCTSSearch):
    """One GEN evaluation per step; every traversal starts again at the root."""

    def __init__(self, model, prompt_ids, mask_id, gen_length, config=None,
                 tokenizer=None, on_event=None, *, seed=1):
        config = config if isinstance(config, ABMEDALConfig) else ABMEDALConfig(**(config or {}))
        super().__init__(model, prompt_ids, mask_id, gen_length, config, tokenizer, on_event, seed=seed)
        self._counted = _SearchModel(model, config.max_model_calls, self._emit)
        self.rng = np.random.default_rng(seed)
        self.posteriors, self.remaining_actions, self.decision_history = {}, {}, {}
        self.history = []
        self.initialization_seconds = 0.0
        self._round = None

    def _register(self, node):
        if node.node_id not in self.posteriors:
            self.posteriors[node.node_id] = {name: self.config.new_posterior() for name in ("node", "GEN", "CONT")}
            self.remaining_actions[node.node_id] = list(node.analysis.actions) if node.depth < self.config.init_tokens else []
            self.decision_history[node.node_id] = []

    def _close_exhausted(self, node):
        if self.config.search_policy == "medal_equivalent":
            return super()._close_exhausted(node)
        while node is not None:
            node.closed = node.depth >= self.config.init_tokens or (
                node.analysis is not None and not self.remaining_actions[node.node_id]
                and all(child.closed for child in node.children))
            node = node.parent

    def _traverse(self):
        node, path, decisions = self.root, [self.root], []
        while not node.closed:
            models = self.posteriors[node.node_id]
            children = [c for c in node.children if not c.closed]
            gen_available = bool(self.remaining_actions[node.node_id])
            cont_available = bool(children)
            if not gen_available and not cont_available:
                self._close_exhausted(node)
                return None, path, decisions
            scores = {}
            forced = not (gen_available and cont_available)
            if forced:
                choice = "GEN" if gen_available else "CONT"
            else:
                scores = {name: models[name].sample(self.rng) for name in ("GEN", "CONT")}
                choice = "GEN" if scores["GEN"] > scores["CONT"] else "CONT"
            decision = {"node_id": node.node_id, "choice": choice, "forced": forced,
                        "gen_available": gen_available, "cont_available": cont_available, "scores": scores}
            if choice == "CONT":
                child_scores = [{"node_id": c.node_id, "score": self.posteriors[c.node_id]["node"].sample(self.rng)} for c in children]
                chosen = max(child_scores, key=lambda item: (item["score"], -item["node_id"]))["node_id"]
                decision.update(child_scores=child_scores, chosen_id=chosen)
            decisions.append(decision)
            self.decision_history[node.node_id].append({"iteration": self.simulations_completed + 1, **decision})
            if choice == "GEN":
                return node, path, decisions
            node = next(c for c in children if c.node_id == decision["chosen_id"])
            path.append(node)
        return None, path, decisions

    def _select(self):
        if self.config.search_policy == "gaussian":
            return self._traverse()
        # Debugging mode preserves MEDAL's sibling rounds, including UCB only
        # between rounds. A step still evaluates just one sibling.
        if self._round is None:
            parent, decisions = medal.select_node(self.root, self.config.exploration_c)
            if parent is None:
                return None, [], []
            path, node = [], parent
            while node is not None:
                path.append(node)
                node = node.parent
            self._round = (parent, list(reversed(path)), decisions)
        return self._round

    def _backup(self, child, parent, path, reward):
        recipients = [(child.node_id, "node"), (parent.node_id, "GEN")]
        recipients += [(node.node_id, "node") for node in path[1:]]
        recipients += [(node.node_id, "CONT") for node in path[:-1]]
        if len(set(recipients)) != len(recipients):
            raise RuntimeError("A posterior would be updated twice for one child.")
        updates = [{"node_id": node_id, "model": name,
                    **self.posteriors[node_id][name].update(reward)} for node_id, name in recipients]
        return updates, medal.backpropagate(child, reward)

    def _finish(self, reason):
        with self._counted.phase("candidate_evaluation"):
            for node in self.candidates.values():
                self._emit("candidate_evaluation", node_id=node.node_id, reward=node.mean_reward, model_calls=0)
            self.stop_reason = reason
        # Wall times stay out of deterministic event histories.
        self._emit("termination", stop_reason=reason, candidate_count=len(self.candidates),
                   target_reached=len(self.candidates) >= self.config.num_candidates,
                   audit=tree_statistics(self))
        if (self.config.stop_mode == "candidates"
                and len(self.candidates) < self.config.num_candidates):
            raise medal.SearchIncompleteError(self.result().stats)

    @torch.inference_mode()
    def step(self):
        if self.done:
            return []
        started = perf_counter()
        try:
            try:
                return self._step()
            finally:
                self.initialization_seconds += perf_counter() - started
        except medal.SearchIncompleteError as error:
            error.stats = self.result().stats
            raise

    def _step(self):
        start = len(self.trace)
        if self._round is None:
            self.cycle += 1
        if self.root.analysis is None:
            with self._counted.phase("tree_analysis"):
                self.root.analysis = medal.analyze_state(
                    self._counted, self.root.state, self.prompt_len, self.mask_id,
                    self.config, excluded_token_ids=self.excluded_token_ids)
                self._register(self.root)
            self._emit("initialization", node_id=0, entropy=self.root.analysis.total_entropy,
                       actions=[medal.action_record(a, self.tokenizer) for a in self.root.analysis.actions],
                       policy=self.config.search_policy, priors=asdict(self.config.new_posterior()))
        if self._at_cap() or self._at_simulation_cap():
            self._finish("model_call_budget" if self._at_cap() else "simulation_budget")
            return self.trace[start:]
        with self._counted.phase("selection"):
            parent, path, decisions = self._select()
        if parent is None:
            self._finish("frontier_exhausted")
            return self.trace[start:]
        self._emit("selection", node_id=parent.node_id, path=[n.node_id for n in path], decisions=decisions)
        with self._counted.phase("expansion"):
            action = self.remaining_actions[parent.node_id].pop(0)
            parent.expanded = True
            child = medal.expand_node(parent, action, len(self.nodes), self.prompt_len, self.mask_id, self.config.init_tokens)
            self.nodes.append(child)
        self._emit("expansion", parent_id=parent.node_id, child_id=child.node_id, depth=child.depth,
                   action=medal.action_record(action, self.tokenizer))
        self.simulations_started += 1
        try:
            with self._counted.phase("simulation"):
                evaluation = medal.evaluate_action(
                    self._counted, parent, child, self.prompt_len, self.mask_id,
                    self.config, self.excluded_token_ids)
        except medal.BudgetExhausted:
            child.closed = True
            self._emit("simulation_interrupted", child_id=child.node_id)
            self._finish("model_call_budget")
            return self.trace[start:]
        self._register(child)
        self.simulations_completed += 1
        self._emit("evaluation", parent_id=parent.node_id, child_id=child.node_id, depth=child.depth,
                   action=medal.action_record(action, self.tokenizer), **evaluation)
        with self._counted.phase("backpropagation"):
            posterior_updates, visits = self._backup(child, parent, path, evaluation["reward"])
        self._emit("backpropagation", child_id=child.node_id, reward=evaluation["reward"],
                   posterior_updates=posterior_updates, updates=visits)
        if child.depth == self.config.init_tokens:
            key = medal._state_key(child.state)
            duplicate = key in self.candidates
            self.candidates.setdefault(key, child)
            self._emit("candidate", node_id=child.node_id, duplicate=duplicate,
                       candidate_count=len(self.candidates), score=child.mean_reward)
        record = {"iteration": self.simulations_completed, "parent_id": parent.node_id, "child_id": child.node_id,
                  "path": [n.node_id for n in path], "decisions": decisions, "reward": evaluation["reward"],
                  "posterior_updates": posterior_updates, "model_calls": self.model_calls,
                  "sequence_nfes": self._counted.sequence_nfes, "audit": tree_statistics(self)}
        self.history.append(record)
        self._emit("iteration", **record)
        # The target wins when it is reached at exactly the NFE cap.
        if len(self.candidates) >= self.config.num_candidates:
            self._finish("candidate_quota")
            return self.trace[start:]
        if self._at_cap() or self._at_simulation_cap():
            self._finish("model_call_budget" if self._at_cap() else "simulation_budget")
            return self.trace[start:]
        if self.config.search_policy == "gaussian":
            self._close_exhausted(child)
        elif not self.remaining_actions[parent.node_id]:
            self._close_exhausted(parent)
            self._round = None
        if self.root.closed:
            self._finish("frontier_exhausted")
        return self.trace[start:]

    def result(self):
        result = super().result()
        counts = Counter(d["choice"] for rows in self.decision_history.values() for d in rows)
        filtering = self._counted.seconds["candidate_filtering"]
        times = {"selection": self._counted.seconds["selection"],
                 "candidate_generation": self._counted.seconds["tree_analysis"] + filtering,
                 "expansion": self._counted.seconds["expansion"],
                 "reward_evaluation": self._counted.seconds["simulation"] - filtering,
                 "backpropagation": self._counted.seconds["backpropagation"],
                 "candidate_selection": self._counted.seconds["candidate_evaluation"],
                 "full_rollout": 0.0}
        times["bookkeeping_trace"] = self.initialization_seconds - sum(times.values())
        result.stats.update(implementation_version="ab_medal_gaussian_v1", search_policy=self.config.search_policy,
                            evaluation_variant="Adaptive MEDAL (immediate-child IG)",
                            iterations=self.simulations_completed, decision_counts=dict(counts),
                            forced_decisions=sum(d["forced"] for rows in self.decision_history.values() for d in rows),
                            target_reached=len(self.candidates) >= self.config.num_candidates,
                            search_sequence_nfes=self._counted.sequence_nfes,
                            initialization_wall_seconds=self.initialization_seconds,
                            timing_seconds=times, **tree_statistics(self))
        return result
