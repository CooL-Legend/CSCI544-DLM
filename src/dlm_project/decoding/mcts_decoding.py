"""Inspectable, one-step-entropy MCTS for a frozen masked language model.

Read in order: analyze_state -> select_node -> expand_node -> evaluate_action
-> backpropagate -> MCTSSearch.step. All search and completion logic lives here.

This adapts MEDAL's confidence filtering (Eq. 5) and information gain (Eq. 9).
Evaluation commits ONE token and observes the child's remaining-mask entropy;
it does not sample a full completion. Final completion is greedy. Search states
are CPU int64 tensors [1, prompt_length + output_length]. Output positions in
actions and traces are zero-based and exclude the prompt. No reference answer
or Sudoku validator is accepted by the decoder.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Callable

import torch


@dataclass(frozen=True)
class MCTSConfig:
    """Validated search settings: Lc, C, K1, K2 and the search-only call budget."""
    init_tokens: int = 2
    num_candidates: int = 3
    top_k_tokens: int = 2
    top_k_actions: int = 2
    max_model_calls: int = 32
    margin_gamma: float = 10.0
    exploration_c: float = 0.01
    epsilon: float = 1e-8

    def __post_init__(self):
        for name in ("init_tokens", "num_candidates", "top_k_tokens",
                     "top_k_actions", "max_model_calls"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"mcts.{name} must be a positive integer.")
        for name in ("margin_gamma", "exploration_c", "epsilon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"mcts.{name} must be finite.")
            if value < 0 or (name == "epsilon" and not 0 < value < 1):
                raise ValueError(f"Invalid mcts.{name}: {value}.")

    def validate_length(self, gen_length):
        if type(gen_length) is not int or not 1 <= self.init_tokens <= gen_length:
            raise ValueError("MCTS requires 1 <= init_tokens <= gen_length.")


def resolve_decoding(decoding=None, generation=None):
    """Resolve YAML settings; an absent section preserves baseline decoding."""
    if decoding is not None and not isinstance(decoding, dict):
        raise ValueError("decoding must be a mapping.")
    options = dict(decoding or {})
    if set(options) - {"method", "seed", "mcts"}:
        raise ValueError("Unknown decoding setting.")
    method = options.get("method", "baseline")
    if not isinstance(method, str) or method not in {"baseline", "confidence", "mcts_entropy"}:
        raise ValueError("decoding.method must be baseline, confidence, or mcts_entropy.")
    seed = options.get("seed", 1)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("decoding.seed must be an integer in [0, 2**63).")
    try:
        config = MCTSConfig(**options.get("mcts", {}))
    except TypeError as error:
        raise ValueError(f"Invalid decoding.mcts settings: {error}") from error
    if generation is not None and method != "baseline":
        length = generation["gen_length"]
        if (generation["steps"] != length or generation["block_length"] != length
                or generation["temperature"] != 0 or generation["remasking"] != "low_confidence"):
            raise ValueError("Confidence/MCTS decoding requires steps = block_length = gen_length, "
                             "temperature = 0, and remasking = low_confidence.")
        if method == "mcts_entropy":
            config.validate_length(length)
    return {"method": method, "seed": seed, "mcts": asdict(config)}


@dataclass(frozen=True)
class Action:
    """Commit token_id at a zero-based output position; confidence is Eq. 5."""
    output_position: int
    token_id: int
    confidence_score: float
    probability: float = 0.0
    entropy: float = 0.0
    margin: float = 0.0


@dataclass
class StateAnalysis:
    """Compact CPU statistics; never retains vocabulary logits/probabilities."""
    entropies: torch.Tensor  # [output_length]; committed positions are zero.
    actions: tuple[Action, ...]
    state_key: tuple[int, ...]

    @property
    def total_entropy(self):
        return self.entropies.double().sum().item()


@dataclass(eq=False)
class Node:
    """CPU state [1, P+L], cached analysis and accumulated MCTS visit statistics."""
    node_id: int
    state: torch.Tensor
    parent: "Node | None" = None
    action: Action | None = None
    depth: int = 0
    children: list["Node"] = field(default_factory=list)
    visits: int = 0
    reward_sum: float = 0.0
    analysis: StateAnalysis | None = None
    expanded: bool = False
    closed: bool = False

    @property
    def mean_reward(self):
        return self.reward_sum / self.visits if self.visits else 0.0


@dataclass
class SearchResult:
    selected: Node
    candidates: list[Node]
    stats: dict
    trace: list[dict]


@dataclass
class CompletionResult:
    token_ids: torch.Tensor
    model_calls: int
    trace: list[dict]


def model_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _state_key(state):
    return tuple(state[0].tolist())


def _action_order(action):
    return (-action.confidence_score, action.output_position, action.token_id)


def _token_topk(probabilities, k, mask_id):
    """Top-k per row, resolving boundary ties by smaller token ID.

    Usually only k+1 values per row leave the GPU. Only a tied cutoff needs a
    scan; we avoid sorting the entire LLaDA vocabulary for every position.
    """
    legal = probabilities.clone()
    legal[:, mask_id] = -torch.inf
    k = min(k, legal.shape[-1] - 1)
    count = min(k + 1, legal.shape[-1] - 1)
    values, indices = torch.topk(legal, count, dim=-1)
    value_rows, index_rows = values.cpu().tolist(), indices.cpu().tolist()
    result = []
    for row, (scores, ids) in enumerate(zip(value_rows, index_rows)):
        pairs = list(zip(scores[:k], ids[:k]))
        if count > k and scores[k - 1] == scores[k]:
            cutoff = scores[k - 1]
            pairs = [(p, token) for p, token in pairs if p > cutoff]
            tied = torch.nonzero(legal[row] == cutoff, as_tuple=False).flatten()
            pairs += [(cutoff, token) for token in tied[:k - len(pairs)].cpu().tolist()]
        result.append(sorted(pairs, key=lambda pair: (-pair[0], pair[1])))
    return result


@torch.inference_mode()
def analyze_state(model, state, prompt_len, mask_id, config, *, force_forward=False):
    """One forward pass on [1, P+L]; return entropies [L] and top-K2 actions.

    Entropy and the top-two margin use the full model distribution. The mask
    token is excluded only from possible commitments. Prompt masks, if any,
    cannot become actions. A completed state normally needs no call;
    force_forward=True preserves the one-forward-per-child evaluation contract
    when Lc equals the entire output length.
    """
    if state.ndim != 2 or state.shape[0] != 1 or not 0 <= prompt_len <= state.shape[1]:
        raise ValueError("Expected one sequence [1, P+L] and a valid prompt length.")
    state = state.detach().cpu()
    output = state[0, prompt_len:]
    positions = torch.nonzero(output == mask_id, as_tuple=False).flatten()
    entropies = torch.zeros(output.numel(), dtype=torch.float32)
    if positions.numel() == 0 and not force_forward:
        return StateAnalysis(entropies, (), _state_key(state))

    device = model_device(model)
    prediction = model(state.to(device))
    if positions.numel() == 0:
        return StateAnalysis(entropies, (), _state_key(state))
    logits = prediction.logits[0, (positions + prompt_len).to(device)].float()
    del prediction
    if not 0 <= mask_id < logits.shape[-1] or logits.shape[-1] < 2:
        raise ValueError("The model vocabulary must contain the mask ID and a non-mask token.")
    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("Non-finite model probabilities; cannot evaluate entropy.")
    # entr(0) = 0: exact Shannon entropy in nats, including zero probabilities.
    h = torch.special.entr(probabilities).sum(dim=-1)
    top_two = probabilities.topk(2, dim=-1).values
    margins = top_two[:, 0] - top_two[:, 1]
    factors = torch.exp(-h) * torch.sigmoid(config.margin_gamma * margins)
    token_rows = _token_topk(probabilities, config.top_k_tokens, mask_id)
    h_cpu, margins_cpu, factors_cpu = h.cpu(), margins.cpu().tolist(), factors.cpu().tolist()
    entropies[positions] = h_cpu
    actions = []
    for row, position in enumerate(positions.tolist()):
        for probability, token_id in token_rows[row]:
            actions.append(Action(position, token_id, probability * factors_cpu[row],
                                  probability, h_cpu[row].item(), margins_cpu[row]))
    actions.sort(key=_action_order)
    return StateAnalysis(entropies, tuple(actions[:config.top_k_actions]), _state_key(state))


# 1. SELECTION: choose a frontier node using accumulated search statistics.
def select_node(root, exploration_c):
    """Return (frontier node or None, UCB decisions along the traversed path)."""
    node, decisions = root, []
    while node.expanded and not node.closed:
        eligible = [child for child in node.children if not child.closed]
        if not eligible:
            return None, decisions
        scores, siblings = {}, []
        for child in node.children:
            bonus = (exploration_c * math.sqrt(math.log(max(1, node.visits)) / child.visits)
                     if child.visits else math.inf)
            ucb = child.mean_reward + bonus
            scores[child.node_id] = ucb
            siblings.append({"node_id": child.node_id, "visits": child.visits,
                             "mean_reward": child.mean_reward, "eligible": not child.closed,
                             "unvisited": child.visits == 0,
                             "exploration_bonus": bonus if math.isfinite(bonus) else None,
                             "ucb": ucb if math.isfinite(ucb) else None})
        chosen = min(eligible, key=lambda child: (-scores[child.node_id],
                                                 *_action_order(child.action), child.node_id))
        decisions.append({"parent_id": node.node_id, "chosen_id": chosen.node_id,
                          "siblings": siblings})
        node = chosen
    return (None if node.closed else node), decisions


# 2. EXPANSION: each child commits exactly one action to an independent copy.
def expand_node(parent, action, node_id, prompt_len, mask_id, init_tokens):
    """Create one child on CPU; action positions are relative to the output."""
    if not 0 <= action.output_position < parent.state.shape[1] - prompt_len:
        raise ValueError("Action position is outside the output.")
    position = prompt_len + action.output_position
    if parent.state[0, position].item() != mask_id or action.token_id == mask_id:
        raise ValueError("An action must replace an output mask with a non-mask token.")
    state = parent.state.clone()
    state[0, position] = action.token_id
    child = Node(node_id, state, parent, action, parent.depth + 1,
                 closed=parent.depth + 1 >= init_tokens)
    parent.children.append(child)
    return child


def information_gain(before, after, epsilon):
    """MEDAL Eq. 9, with a guard for an effectively certain parent state."""
    return (before - after) / before if before > epsilon else 0.0


# 3. EVALUATION: observe one post-action distribution, without more commitments.
def evaluate_action(model, parent, child, prompt_len, mask_id, config):
    """Evaluate a newly expanded child once; cache its statistics for reuse."""
    if parent.analysis is None or child.analysis is not None:
        raise ValueError("Evaluation requires a cached parent and an unevaluated child.")
    child.analysis = analyze_state(model, child.state, prompt_len, mask_id, config, force_forward=True)
    before, after = parent.analysis.total_entropy, child.analysis.total_entropy
    removed = parent.analysis.entropies[child.action.output_position].item()
    return {"entropy_before": before, "entropy_after": after,
            "committed_position_entropy": removed,
            "reward": information_gain(before, after, config.epsilon),
            "context_gain": before - removed - after,
            "entropies_before": parent.analysis.entropies.tolist(),
            "entropies_after": child.analysis.entropies.tolist()}


# 4. BACKPROPAGATION: update tree values; no gradients or weight updates occur.
def backpropagate(node, reward):
    """Update the child and all ancestors once; return before/after snapshots."""
    if not math.isfinite(reward):
        raise ValueError("Search rewards must be finite.")
    updates = []
    while node is not None:
        before = {"visits": node.visits, "reward_sum": node.reward_sum,
                  "mean_reward": node.mean_reward}
        node.visits += 1
        node.reward_sum += reward
        updates.append({"node_id": node.node_id, "before": before,
                        "after": {"visits": node.visits, "reward_sum": node.reward_sum,
                                  "mean_reward": node.mean_reward}})
        node = node.parent
    return updates


def action_record(action, tokenizer=None):
    record = asdict(action)
    record["decoded_token"] = (tokenizer.decode([action.token_id], skip_special_tokens=False)
                               if tokenizer is not None else str(action.token_id))
    return record


def state_record(state, prompt_len, mask_id, tokenizer=None):
    """A token-position view: visible masks and exact IDs, not repaired text."""
    ids = state[0, prompt_len:].tolist()
    committed = []
    for position, token in enumerate(ids):
        if token != mask_id:
            text = tokenizer.decode([token], skip_special_tokens=False) if tokenizer is not None else str(token)
            committed.append({"output_position": position, "token_id": token, "decoded_token": text})
    tokens = {item["output_position"]: repr(item["decoded_token"]) for item in committed}
    return {"output_ids": ids, "committed": committed, "remaining_masks": ids.count(mask_id),
            "partial_output": " ".join(tokens.get(i, f"[M{i}]") for i in range(len(ids)))}


class MCTSSearch:
    """Stateful search. step() exposes one cycle; run() resumes the same tree.

    The constructor copies prompt_ids [1, P] to CPU and appends L masks.
    The first step evaluates the root. Every new child costs one forward pass.
    on_event, when provided, receives each JSON-serializable event immediately.
    """

    def __init__(self, model, prompt_ids, mask_id, gen_length, config=None,
                 tokenizer=None, on_event: Callable[[dict], None] | None = None):
        self.config = config if isinstance(config, MCTSConfig) else MCTSConfig(**(config or {}))
        self.config.validate_length(gen_length)
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.dtype != torch.long:
            raise ValueError("prompt_ids must be an int64 tensor with shape [1, P].")
        self.model, self.mask_id, self.tokenizer = model, mask_id, tokenizer
        self.prompt_len = prompt_ids.shape[1]
        state = torch.full((1, self.prompt_len + gen_length), mask_id, dtype=torch.long)
        state[:, :self.prompt_len] = prompt_ids.detach().cpu()
        self.root = Node(0, state)
        self.nodes = [self.root]
        self.candidates = {}
        self.trace = []
        self.on_event = on_event
        self.model_calls = 0
        self.cycle = 0
        self.stop_reason = None

    @property
    def done(self):
        return self.stop_reason is not None

    def _emit(self, stage, **details):
        event = {"cycle": self.cycle, "stage": stage, "search_model_calls": self.model_calls,
                 **details}
        self.trace.append(event)
        if self.on_event is not None:
            self.on_event(event)

    def _score(self, node):
        return information_gain(self.root.analysis.total_entropy, node.analysis.total_entropy,
                                self.config.epsilon)

    def _close_exhausted(self, node):
        while node is not None:
            node.closed = node.depth >= self.config.init_tokens or (
                node.expanded and all(child.closed for child in node.children))
            node = node.parent

    def _finish_if_needed(self):
        if len(self.candidates) >= self.config.num_candidates:
            reason = "candidate_quota"
        elif self.model_calls >= self.config.max_model_calls:
            reason = "model_call_budget"
        elif self.root.closed:
            reason = "frontier_exhausted"
        else:
            return
        self.stop_reason = reason
        result = self.result()
        self._emit("termination", **result.stats,
                   selected_state=state_record(result.selected.state, self.prompt_len,
                                               self.mask_id, self.tokenizer))

    @torch.inference_mode()
    def step(self):
        """Return ordered events for one selection/expansion/evaluation/backup cycle.

        Finish the retained sibling set unless the call budget is exhausted.
        Candidate quotas are checked after the cycle, so a cycle can collect
        more than C candidates. Calling step() after termination returns [].
        """
        if self.done:
            return []
        start = len(self.trace)
        self.cycle += 1
        if self.root.analysis is None:
            self.model_calls += 1
            self.root.analysis = analyze_state(self.model, self.root.state, self.prompt_len,
                                               self.mask_id, self.config)
            self._emit("initialization", node_id=0, entropy=self.root.analysis.total_entropy,
                       model_device=str(model_device(self.model)),
                       search_state_device=str(self.root.state.device),
                       entropies=self.root.analysis.entropies.tolist(),
                       state=state_record(self.root.state, self.prompt_len, self.mask_id, self.tokenizer),
                       actions=[action_record(a, self.tokenizer) for a in self.root.analysis.actions])
            self._finish_if_needed()
            if self.done:
                return self.trace[start:]

        parent, decisions = select_node(self.root, self.config.exploration_c)
        if parent is None:
            self.root.closed = True
            self._finish_if_needed()
            return self.trace[start:]
        self._emit("selection", node_id=parent.node_id,
                   path=[self.root.node_id] + [hop["chosen_id"] for hop in decisions],
                   decisions=decisions, visits=parent.visits, mean_reward=parent.mean_reward)
        parent.expanded = True
        for action in parent.analysis.actions:
            if self.model_calls >= self.config.max_model_calls:
                break
            child = expand_node(parent, action, len(self.nodes), self.prompt_len,
                                self.mask_id, self.config.init_tokens)
            self.nodes.append(child)
            self._emit("expansion", parent_id=parent.node_id, child_id=child.node_id,
                       depth=child.depth, action=action_record(action, self.tokenizer))
            self.model_calls += 1
            evaluation = evaluate_action(self.model, parent, child, self.prompt_len,
                                         self.mask_id, self.config)
            self._emit("evaluation", parent_id=parent.node_id, child_id=child.node_id,
                       depth=child.depth, action=action_record(action, self.tokenizer),
                       child_state=state_record(child.state, self.prompt_len, self.mask_id, self.tokenizer),
                       **evaluation)
            updates = backpropagate(child, evaluation["reward"])
            self._emit("backpropagation", child_id=child.node_id,
                       reward=evaluation["reward"], updates=updates)
            if child.depth == self.config.init_tokens:
                key = _state_key(child.state)
                duplicate = key in self.candidates
                self.candidates.setdefault(key, child)
                self._emit("candidate", node_id=child.node_id, duplicate=duplicate,
                           candidate_count=len(self.candidates), score=self._score(child))
        self._close_exhausted(parent)
        self._finish_if_needed()
        return self.trace[start:]

    def result(self):
        """Snapshot the current best initialization; does not consume model calls."""
        if self.root.analysis is None:
            raise RuntimeError("Call step() or run() before requesting a result.")
        candidates = list(self.candidates.values())
        pool = candidates or [node for node in self.nodes if node.analysis is not None]
        deepest = max(node.depth for node in pool)
        pool = [node for node in pool if node.depth == deepest]
        selected = min(pool, key=lambda node: (-self._score(node), node.node_id))
        fallback = self.done and len(candidates) < self.config.num_candidates
        stats = {"finished": self.done, "stop_reason": self.stop_reason,
                 "search_model_calls": self.model_calls, "cycles": self.cycle,
                 "requested_depth": self.config.init_tokens, "selected_depth": selected.depth,
                 "max_depth": max(node.depth for node in self.nodes),
                 "candidate_target": self.config.num_candidates, "candidate_count": len(candidates),
                 "candidate_ids": [node.node_id for node in candidates],
                 "selected_node_id": selected.node_id, "selected_score": self._score(selected),
                 "fallback_used": fallback,
                 "fallback_reason": ("insufficient_target_candidates" if candidates else "deepest_available_state")
                 if fallback else None,
                 "budget_exhausted": self.model_calls >= self.config.max_model_calls}
        return SearchResult(selected, candidates, stats, list(self.trace))

    def run(self):
        """Continue this tree until quota, budget, or frontier exhaustion."""
        while not self.done:
            self.step()
        return self.result()


@torch.inference_mode()
def complete_confidence(model, state, prompt_len, mask_id, config=None,
                        initial_analysis=None, tokenizer=None, on_event=None):
    """Greedily fill remaining masks in a COPY of [1, P+L], one token per step.

    Reuses the selected search node's cached analysis for the first commitment.
    model_calls counts only actual completion-phase forward passes. Prompt
    tokens and commitments remain fixed, even if a prompt token is mask_id.
    """
    config = config or MCTSConfig()
    if state.ndim != 2 or state.shape[0] != 1 or state.dtype != torch.long:
        raise ValueError("state must be an int64 tensor with shape [1, P+L].")
    if not 0 <= prompt_len <= state.shape[1]:
        raise ValueError("Invalid prompt length.")
    x = state.detach().cpu().clone()
    analysis, calls, trace = initial_analysis, 0, []
    if analysis is not None and analysis.state_key != _state_key(x):
        raise ValueError("Cached analysis belongs to a different partial state.")
    while (x[0, prompt_len:] == mask_id).any():
        reused = analysis is not None
        if analysis is None:
            analysis = analyze_state(model, x, prompt_len, mask_id, config)
            calls += 1
        if not analysis.actions:
            raise RuntimeError("No legal commitment for a masked output position.")
        action = analysis.actions[0]
        x[0, prompt_len + action.output_position] = action.token_id
        event = {"stage": "completion", "step": len(trace) + 1,
                 "completion_model_calls": calls, "used_cached_analysis": reused,
                 "remaining_masks": (x[0, prompt_len:] == mask_id).sum().item(),
                 "action": action_record(action, tokenizer)}
        trace.append(event)
        if on_event is not None:
            on_event(event)
        analysis = None
    return CompletionResult(x, calls, trace)
