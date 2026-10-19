"""Documented reconstruction of MEDAL Algorithms 1 and 2.

Immediate parent-to-child information gain, cached candidate ranking,
and literal adjusted-score softmax completion. The interpretation
choices are exported in RECONSTRUCTION_ASSUMPTIONS; author-code parity is not
claimed. Trees retain compact CPU states and never accept reference answers.
"""

from dataclasses import asdict, dataclass, field
import math
import hashlib
from contextlib import contextmanager, nullcontext
from time import perf_counter
from typing import Callable

import torch


MCTS_EXCLUDED_TOKENS = ("<|endoftext|>", "<|eot_id|>")


@dataclass(frozen=True)
class MCTSConfig:
    """Lc, C, K1, K2 and optional cap covering root and child forwards."""
    init_tokens: int = 20
    num_candidates: int = 3
    top_k_tokens: int = 3
    top_k_actions: int = 5
    max_model_calls: int | None = None
    max_simulations: int | None = None
    margin_gamma: float = 10.0
    exploration_c: float = 0.01
    epsilon: float = 1e-8
    stop_mode: str = "candidates"

    def __post_init__(self):
        if self.stop_mode not in ("candidates", "budget"):
            raise ValueError("mcts.stop_mode must be candidates or budget.")
        for name in ("init_tokens", "num_candidates", "top_k_tokens",
                     "top_k_actions"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"mcts.{name} must be a positive integer.")
        if self.max_model_calls is not None and (type(self.max_model_calls) is not int or self.max_model_calls < 1):
            raise ValueError("mcts.max_model_calls must be null or a positive integer.")
        if self.max_simulations is not None and (type(self.max_simulations) is not int or self.max_simulations < 1):
            raise ValueError("mcts.max_simulations must be null or a positive integer.")
        if self.stop_mode == "budget" and self.max_model_calls is None and self.max_simulations is None:
            raise ValueError("budget mode requires a finite max_model_calls or max_simulations.")
        for name in ("margin_gamma", "exploration_c", "epsilon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"mcts.{name} must be finite.")
            if value < 0 or (name == "epsilon" and not 0 < value < 1):
                raise ValueError(f"Invalid mcts.{name}: {value}.")

    def validate_length(self, gen_length):
        if type(gen_length) is not int or not 1 <= self.init_tokens < gen_length:
            raise ValueError("MCTS requires 1 <= init_tokens < gen_length.")


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
        if generation["steps"] != length or generation["block_length"] != length:
            raise ValueError("Confidence/MCTS decoding requires steps = block_length = gen_length.")
        if generation["remasking"] != "low_confidence":
            raise ValueError("Confidence/MCTS decoding requires remasking = low_confidence.")
        if method == "confidence" and generation["temperature"] != 0:
            raise ValueError("Greedy confidence requires temperature = 0.")
        if method == "mcts_entropy":
            if generation["temperature"] <= 0 or not math.isfinite(generation["temperature"]):
                raise ValueError("MEDAL completion requires positive temperature (default 1.0).")
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
    mean_margin: float | None = None

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
    immediate_reward: float | None = None

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


def _search_excluded_token_ids(tokenizer):
    """Resolve terminal tokens that tree search must not count as initialization."""
    if tokenizer is None or not callable(getattr(tokenizer, "get_vocab", None)):
        return ()
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, dict):
        return ()
    return tuple(sorted({int(vocabulary[token]) for token in MCTS_EXCLUDED_TOKENS
                         if token in vocabulary}))


def _token_topk(probabilities, k, mask_id, excluded_token_ids=()):
    """Top-k per row, resolving boundary ties by smaller token ID.

    Usually only k+1 values per row leave the GPU. Only a tied cutoff needs a
    scan; we avoid sorting the entire LLaDA vocabulary for every position.
    """
    legal = probabilities.clone()
    excluded = {mask_id, *(token for token in excluded_token_ids
                           if 0 <= token < legal.shape[-1])}
    legal[:, list(excluded)] = -torch.inf
    legal_count = legal.shape[-1] - len(excluded)
    if legal_count < 1:
        raise ValueError("No legal token remains after applying the MCTS search filter.")
    k = min(k, legal_count)
    count = min(k + 1, legal_count)
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


def _distribution(model, state, prompt_len, mask_id, config, *, legacy_entropy=False,
                  excluded_token_ids=()):
    """One forward; full-vocabulary probabilities stay local to this step."""
    if (state.ndim != 2 or state.shape[0] != 1 or state.dtype != torch.long
            or not 0 <= prompt_len <= state.shape[1]):
        raise ValueError("Expected one int64 sequence [1, P+L] and a valid prompt length.")
    state = state.detach().cpu()
    positions = torch.nonzero(state[0, prompt_len:] == mask_id, as_tuple=False).flatten()
    entropies = torch.zeros(state.shape[1] - prompt_len)
    if not positions.numel():
        return StateAnalysis(entropies, (), _state_key(state)), positions, None, None
    device = model_device(model)
    prediction = model(state.to(device))
    logits = prediction.logits[0, (positions + prompt_len).to(device)].float()
    del prediction
    if not 0 <= mask_id < logits.shape[-1] or logits.shape[-1] < 2:
        raise ValueError("The model vocabulary must contain the mask ID and a non-mask token.")
    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("Non-finite model probabilities; cannot evaluate entropy.")
    h = (torch.special.entr(probabilities).sum(-1) if legacy_entropy else
         -(probabilities * torch.log(probabilities + config.epsilon)).sum(-1))
    top_two = probabilities.topk(2, dim=-1).values
    margins = top_two[:, 0] - top_two[:, 1]
    factors = torch.exp(-h) * torch.sigmoid(config.margin_gamma * margins)
    # Root filtering is already in tree_analysis. Split child filtering out for
    # the demo's stage report; the existing simulation timer stays inclusive.
    timing = (model.phase("candidate_filtering")
              if isinstance(model, _SearchModel) and model.current_phase == "simulation" else nullcontext())
    with timing:
        token_rows = _token_topk(probabilities, config.top_k_tokens, mask_id,
                                 excluded_token_ids)
        h_cpu, margins_cpu, factors_cpu = h.cpu(), margins.cpu().tolist(), factors.cpu().tolist()
        entropies[positions] = h_cpu
        actions = []
        for row, position in enumerate(positions.tolist()):
            for probability, token_id in token_rows[row]:
                actions.append(Action(position, token_id, probability * factors_cpu[row],
                                      probability, h_cpu[row].item(), margins_cpu[row]))
        actions.sort(key=_action_order)
    return (StateAnalysis(entropies, tuple(actions[:config.top_k_actions]), _state_key(state), margins.mean().item()),
            positions, probabilities, factors)


@torch.inference_mode()
def analyze_state(model, state, prompt_len, mask_id, config, *, force_forward=False,
                  legacy_entropy=False, excluded_token_ids=()):
    """Return compact CPU analysis; completed states need no forward.

    force_forward remains accepted for source compatibility, but no longer
    causes unnecessary inference on a completed state.
    """
    return _distribution(model, state, prompt_len, mask_id, config,
                         legacy_entropy=legacy_entropy,
                         excluded_token_ids=excluded_token_ids)[0]


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


IMPLEMENTATION_VERSION = "medal_reconstruction_v2"
PAPER_URL = "https://aclanthology.org/2026.findings-eacl.180.pdf"
RECONSTRUCTION_ASSUMPTIONS = {
    "simulation": "one forward on the immediate child; cache its entropy and actions",
    "reward": "Eq9 gain from cached parent entropy to immediate child entropy",
    "candidate_evaluation": "rank cached mean tree rewards; no additional model forwards",
    "completion": "fresh partial-state completion; literal softmax(s/temperature) over all legal position-token actions",
    "decomposition": "inline three subtasks; disjoint two-shot demonstrations",
    "quota": "stop immediately at first C distinct depth-Lc states",
    "constants": "gamma=10 and exploration_c=0.01 are repository defaults, not verified author settings",
}


def derive_seed(seed, phase, identifier=0):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63).")
    raw = hashlib.sha256(f"{IMPLEMENTATION_VERSION}:{seed}:{phase}:{identifier}".encode()).digest()
    return int.from_bytes(raw[:8], "big") % 2**63


class BudgetExhausted(RuntimeError):
    """Internal interruption before a forward exceeding the cap."""


class SearchIncompleteError(RuntimeError):
    def __init__(self, stats):
        self.stats = stats
        super().__init__(f"MEDAL initialization incomplete: {stats['stop_reason']}; "
                         f"collected {stats['candidate_count']}/{stats['candidate_target']} candidates, "
                         f"evaluated {stats['evaluated_candidates']}. Use explicit budget mode for an ablation.")


class _SearchModel:
    """One invocation counter, including failed child evaluations; no model reload."""
    def __init__(self, model, cap, emit):
        self.model, self.cap, self.emit = model, cap, emit
        self.calls = 0
        self.current_phase = "tree_analysis"
        self.counts = {name: 0 for name in ("tree_analysis", "simulation", "candidate_evaluation")}
        self.seconds = dict.fromkeys((*self.counts, "selection", "candidate_filtering",
                                     "expansion", "backpropagation"), 0.0)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        if self.cap is not None and self.calls >= self.cap:
            raise BudgetExhausted("Search forward cap reached.")
        self.calls += 1
        self.counts[self.current_phase] += 1
        try:
            return self.model(*args, **kwargs)
        finally:
            self.emit("forward", phase=self.current_phase, **self.costs())

    def costs(self):
        return {f"{phase}_model_calls": count for phase, count in self.counts.items()}

    @contextmanager
    def phase(self, name):
        previous = self.current_phase
        self.current_phase = name
        device = model_device(self.model)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = perf_counter()
        try:
            yield
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self.seconds[name] += perf_counter() - started
            self.current_phase = previous


@dataclass
class RolloutResult:
    token_ids: torch.Tensor
    gains: list[float]
    mean_reward: float
    seed: int
    model_calls: int
    initial_analysis: StateAnalysis
    trace: list[dict]


_MULTINOMIAL_MAX_CATEGORIES = 2**24


def _sample_action(analysis, positions, probabilities, factors, mask_id, generator,
                   policy, temperature):
    if policy == "native":
        # Top action's position is globally best; sample over the full legal vocabulary.
        position = analysis.actions[0].output_position
        row = positions.tolist().index(position)
        weights = probabilities[row].clone()
        weights[mask_id] = 0
        if weights.sum() <= 0:
            raise ValueError("No probability mass on legal tokens.")
        token = torch.multinomial(weights, 1, generator=generator).item()
    else:
        scores = probabilities * factors[:, None]
        scores[:, mask_id] = -torch.inf
        weights = torch.softmax(scores.flatten() / temperature, dim=0)
        if weights.numel() > _MULTINOMIAL_MAX_CATEGORIES:
            # Same joint distribution, using two draws below PyTorch's limit:
            # P(position, token) = P(position) * P(token | position).
            weights = weights.view_as(scores)
            row = torch.multinomial(weights.sum(dim=1), 1, generator=generator).item()
            token = torch.multinomial(weights[row], 1, generator=generator).item()
        else:
            flat = torch.multinomial(weights, 1, generator=generator).item()
            row, token = divmod(flat, probabilities.shape[1])
        position = positions[row].item()
    probability = probabilities[row, token].item()
    top_two = probabilities[row].topk(2).values
    margin = (top_two[0] - top_two[1]).item()
    return Action(position, token, probability * factors[row].item(), probability,
                  analysis.entropies[position].item(), margin)


@torch.inference_mode()
def rollout(model, state, prompt_len, mask_id, config, *, seed, policy="native",
            temperature=1.0, parent_analysis=None, on_event=None, tokenizer=None):
    """Completion helper; MCTS search evaluates immediate children without calling it."""
    if policy not in ("native", "adjusted_softmax") or temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Invalid rollout policy or nonpositive temperature.")
    derive_seed(seed, "validate_rollout")
    x = state.detach().cpu().clone()
    generator = torch.Generator(device=model_device(model)).manual_seed(seed)
    calls, gains, trace = 0, [], []
    analysis, positions, probabilities, factors = _distribution(model, x, prompt_len, mask_id, config)
    calls += int(positions.numel() > 0)
    initial_analysis = analysis
    if parent_analysis is not None:
        gains.append(information_gain(parent_analysis.total_entropy, analysis.total_entropy, config.epsilon))
    while positions.numel():
        signals = {"masked_count": positions.numel(),
                   "mean_token_entropy": analysis.total_entropy / positions.numel(),
                   "mean_confidence_margin": analysis.mean_margin,
                   "signal_nfe": getattr(model, "calls", calls), "signal_extra_nfes": 0}
        action = _sample_action(analysis, positions, probabilities, factors, mask_id,
                                generator, policy, temperature)
        x[0, prompt_len + action.output_position] = action.token_id
        before = analysis.total_entropy
        analysis, positions, probabilities, factors = _distribution(model, x, prompt_len, mask_id, config)
        calls += int(positions.numel() > 0)
        gain = information_gain(before, analysis.total_entropy, config.epsilon)
        if not math.isfinite(gain):
            raise ValueError("Non-finite rollout information gain.")
        gains.append(gain)
        event = {"stage": "rollout_step", "step": len(trace) + 1, "seed": seed,
                 **signals,
                 "policy": policy, "model_calls": calls, "remaining_masks": positions.numel(),
                 "entropy_before": before, "entropy_after": analysis.total_entropy,
                 "gain": gain, "action": action_record(action, tokenizer)}
        trace.append(event)
        if on_event:
            on_event(event)
    return RolloutResult(x, gains, sum(gains) / len(gains) if gains else 0.0,
                         seed, calls, initial_analysis, trace)


# 3. SIMULATION: evaluate only the immediate child, without further commitments.
@torch.inference_mode()
def evaluate_action(model, parent, child, prompt_len, mask_id, config,
                    excluded_token_ids=()):
    """Use the cached parent and one child forward to compute the edge reward."""
    if parent.analysis is None or child.analysis is not None:
        raise ValueError("Evaluation requires a cached parent and an unevaluated child.")
    analysis = analyze_state(model, child.state, prompt_len, mask_id, config,
                             excluded_token_ids=excluded_token_ids)
    before, after = parent.analysis.total_entropy, analysis.total_entropy
    reward = information_gain(before, after, config.epsilon)
    if not math.isfinite(reward):
        raise ValueError("Non-finite child information gain.")
    child.analysis, child.immediate_reward = analysis, reward
    remaining = int((child.state[0, prompt_len:] == mask_id).sum().item())
    removed = parent.analysis.entropies[child.action.output_position].item()
    return {"entropy_before": before, "entropy_after": after,
            "immediate_gain": reward, "reward": reward, "evaluation_model_calls": int(remaining > 0),
            "masked_count": remaining, "mean_token_entropy": after / remaining if remaining else 0.0,
            "mean_confidence_margin": analysis.mean_margin,
            "signal_nfe": getattr(model, "calls", None), "signal_extra_nfes": 0,
            "committed_position_entropy": removed, "context_gain": before - removed - after,
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
    """One resumable tree; step() evaluates immediate children and backs up gains."""
    def __init__(self, model, prompt_ids, mask_id, gen_length, config=None,
                 tokenizer=None, on_event: Callable[[dict], None] | None = None, *, seed=1):
        self.config = config if isinstance(config, MCTSConfig) else MCTSConfig(**(config or {}))
        self.config.validate_length(gen_length)
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.dtype != torch.long:
            raise ValueError("prompt_ids must be an int64 tensor with shape [1, P].")
        derive_seed(seed, "validation")
        self.seed = seed
        self.model, self.mask_id, self.tokenizer = model, mask_id, tokenizer
        self.excluded_token_ids = _search_excluded_token_ids(tokenizer)
        self.prompt_len = prompt_ids.shape[1]
        state = torch.full((1, self.prompt_len + gen_length), mask_id, dtype=torch.long)
        state[:, :self.prompt_len] = prompt_ids.detach().cpu()
        self.root = Node(0, state)
        self.nodes, self.candidates = [self.root], {}
        self.trace, self.on_event = [], on_event
        self.cycle, self.stop_reason = 0, None
        self.simulations_started = self.simulations_completed = 0
        self._counted = _SearchModel(model, self.config.max_model_calls, self._emit)

    @property
    def model_calls(self):
        return self._counted.calls

    @property
    def done(self):
        return self.stop_reason is not None

    def _emit(self, stage, **details):
        event = {"cycle": self.cycle, "stage": stage, "search_model_calls": self.model_calls, **details}
        self.trace.append(event)
        if self.on_event:
            self.on_event(event)

    def _close_exhausted(self, node):
        while node is not None:
            node.closed = node.depth >= self.config.init_tokens or (
                node.expanded and all(child.closed for child in node.children))
            node = node.parent

    def _at_cap(self):
        return self.config.max_model_calls is not None and self.model_calls >= self.config.max_model_calls

    def _at_simulation_cap(self):
        return self.config.max_simulations is not None and self.simulations_completed >= self.config.max_simulations

    def _finish(self, reason):
        with self._counted.phase("candidate_evaluation"):
            for node in self.candidates.values():
                self._emit("candidate_evaluation", phase="candidate_evaluation", node_id=node.node_id,
                           reward=node.mean_reward, immediate_gain=node.immediate_reward, model_calls=0,
                           state=state_record(node.state, self.prompt_len, self.mask_id, self.tokenizer))
            self.stop_reason = reason
            result = self.result()
        result.stats["candidate_evaluation_latency_seconds"] = self._counted.seconds["candidate_evaluation"]
        self._emit("termination", **{k:v for k,v in result.stats.items() if not k.endswith("_latency_seconds")},
                   selected_state=state_record(result.selected.state, self.prompt_len, self.mask_id, self.tokenizer))
        if self.config.stop_mode == "candidates" and result.stats["fallback_used"]:
            raise SearchIncompleteError(result.stats)

    @torch.inference_mode()
    def step(self):
        if self.done:
            return []
        start = len(self.trace)
        self.cycle += 1
        if self.root.analysis is None:
            with self._counted.phase("tree_analysis"):
                self.root.analysis = analyze_state(self._counted, self.root.state, self.prompt_len,
                                                   self.mask_id, self.config,
                                                   excluded_token_ids=self.excluded_token_ids)
            self._emit("initialization", node_id=0, entropy=self.root.analysis.total_entropy,
                       masked_count=self.root.state.shape[1] - self.prompt_len,
                       mean_token_entropy=self.root.analysis.total_entropy / (self.root.state.shape[1] - self.prompt_len),
                       mean_confidence_margin=self.root.analysis.mean_margin,
                       signal_nfe=self.model_calls, signal_extra_nfes=0,
                       implementation_version=IMPLEMENTATION_VERSION, assumptions=RECONSTRUCTION_ASSUMPTIONS,
                       search_excluded_tokens=list(MCTS_EXCLUDED_TOKENS),
                       search_excluded_token_ids=list(self.excluded_token_ids),
                       model_device=str(model_device(self.model)), search_state_device="cpu",
                       entropies=self.root.analysis.entropies.tolist(),
                       state=state_record(self.root.state, self.prompt_len, self.mask_id, self.tokenizer),
                       actions=[action_record(a, self.tokenizer) for a in self.root.analysis.actions])
        if self._at_cap():
            self._finish("model_call_budget")
            return self.trace[start:]
        if self._at_simulation_cap():
            self._finish("simulation_budget")
            return self.trace[start:]
        with self._counted.phase("selection"):
            parent, decisions = select_node(self.root, self.config.exploration_c)
        if parent is None:
            self._finish("frontier_exhausted")
            return self.trace[start:]
        self._emit("selection", node_id=parent.node_id,
                   path=[0] + [hop["chosen_id"] for hop in decisions], decisions=decisions)
        parent.expanded = True
        for action in parent.analysis.actions:
            with self._counted.phase("expansion"):
                child = expand_node(parent, action, len(self.nodes), self.prompt_len, self.mask_id, self.config.init_tokens)
                self.nodes.append(child)
            self._emit("expansion", parent_id=parent.node_id, child_id=child.node_id,
                       depth=child.depth, action=action_record(action, self.tokenizer))
            self.simulations_started += 1
            self._emit("simulation_started", child_id=child.node_id,
                       simulations_started=self.simulations_started)
            try:
                with self._counted.phase("simulation"):
                    evaluation = evaluate_action(self._counted, parent, child, self.prompt_len, self.mask_id,
                                                 self.config, self.excluded_token_ids)
            except BudgetExhausted:
                child.closed = True
                self._emit("simulation_interrupted", phase="simulation", child_id=child.node_id)
                self._finish("model_call_budget")
                return self.trace[start:]
            self.simulations_completed += 1
            self._emit("evaluation", parent_id=parent.node_id, child_id=child.node_id,
                       depth=child.depth, action=action_record(action, self.tokenizer),
                       child_state=state_record(child.state, self.prompt_len, self.mask_id, self.tokenizer), **evaluation)
            with self._counted.phase("backpropagation"):
                updates = backpropagate(child, evaluation["reward"])
            self._emit("backpropagation", child_id=child.node_id, reward=evaluation["reward"], updates=updates)
            if child.depth == self.config.init_tokens:
                key = _state_key(child.state)
                duplicate = key in self.candidates
                self.candidates.setdefault(key, child)
                self._emit("candidate", node_id=child.node_id, duplicate=duplicate,
                           candidate_count=len(self.candidates), score=evaluation["reward"])
            if self.config.stop_mode == "candidates" and len(self.candidates) >= self.config.num_candidates:
                self._finish("candidate_quota")
                return self.trace[start:]
            if self._at_cap():
                self._finish("model_call_budget")
                return self.trace[start:]
            if self._at_simulation_cap():
                self._finish("simulation_budget")
                return self.trace[start:]
        self._close_exhausted(parent)
        if self.root.closed:
            self._finish("frontier_exhausted")
        return self.trace[start:]

    def result(self):
        """Read-only snapshot; provisional frontier selection during inspection."""
        if self.root.analysis is None:
            raise RuntimeError("Call step() or run() before requesting a result.")
        candidates = list(self.candidates.values())
        evaluated = [node for node in self.nodes if node.immediate_reward is not None]
        if candidates:
            selected = max(candidates, key=lambda n: n.mean_reward)
        else:
            selected = min(evaluated or [self.root], key=lambda n: (-n.depth, -n.mean_reward, n.node_id))
        score = selected.mean_reward
        complete = len(candidates) >= self.config.num_candidates
        fallback = self.done and not complete
        parents = len({node.parent.node_id for node in evaluated})
        stats = {"implementation_version": IMPLEMENTATION_VERSION, "paper_url": PAPER_URL,
                 "reconstruction_assumptions": dict(RECONSTRUCTION_ASSUMPTIONS),
                 "evaluation_variant": "budget-limited MEDAL reconstruction (immediate-child IG)" if self.config.stop_mode == "budget"
                 else "MEDAL reconstruction (immediate-child IG)",
                 "search_excluded_tokens": list(MCTS_EXCLUDED_TOKENS),
                 "search_excluded_token_ids": list(self.excluded_token_ids),
                 "finished": self.done, "stop_reason": self.stop_reason, "stop_mode": self.config.stop_mode,
                 "search_model_calls": self.model_calls, **self._counted.costs(),
                 **{f"{phase}_latency_seconds": seconds for phase, seconds in self._counted.seconds.items()},
                 "cycles": self.cycle, "requested_depth": self.config.init_tokens, "selected_depth": selected.depth,
                 "search_invocations": 1, "mcts_simulations": self.simulations_completed,
                 "simulations_per_invocation": [self.simulations_completed],
                 "simulations_started": self.simulations_started,
                 "simulations_interrupted": self.simulations_started - self.simulations_completed,
                 "requested_simulation_budget": self.config.max_simulations,
                 "simulation_decoding_steps": 0,
                 "average_search_depth": sum(n.depth for n in evaluated) / len(evaluated) if evaluated else 0.0,
                 "expanded_nodes": len({_state_key(n.state) for n in evaluated}),
                 "unique_tree_states": len({_state_key(n.state) for n in self.nodes}),
                 "max_depth": max(n.depth for n in self.nodes), "candidate_target": self.config.num_candidates,
                 "candidate_count": len(candidates), "evaluated_candidates": len(candidates),
                 "evaluated_children": len(evaluated), "expanded_parents": parents,
                 "realized_branching_factor": len(evaluated) / parents if parents else 0.0,
                 "candidate_ids": [n.node_id for n in candidates], "selected_node_id": selected.node_id,
                 "selected_score": score, "fallback_used": fallback,
                 "fallback_reason": self.stop_reason if fallback else None, "budget_exhausted": self._at_cap(),
                 "candidate_evaluations": [{"node_id": n.node_id, "reward": n.mean_reward,
                     "immediate_gain": n.immediate_reward, "model_calls": 0,
                     **state_record(n.state, self.prompt_len, self.mask_id, self.tokenizer)} for n in candidates]}
        return SearchResult(selected, candidates, stats, list(self.trace))

    def run(self):
        while not self.done:
            self.step()
        result = self.result()
        if self.config.stop_mode == "candidates" and result.stats["fallback_used"]:
            raise SearchIncompleteError(result.stats)
        return result


@torch.inference_mode()
def complete_medal(model, state, prompt_len, mask_id, config=None, *, seed=1,
                   temperature=1.0, tokenizer=None, on_event=None):
    """Fresh literal adjusted-score softmax, without top-K filtering or cache reuse."""
    events = []
    class CountCompletion:
        calls = 0
        def __getattr__(self, name):
            return getattr(model, name)
        def __call__(self, *args, **kwargs):
            self.calls += 1
            try:
                return model(*args, **kwargs)
            finally:
                emit({"stage": "forward", "model_calls": self.calls, "seed": seed})
    def emit(event):
        events.append(event)
        if on_event:
            on_event(event)
    counted = CountCompletion()
    result = rollout(counted, state, prompt_len, mask_id, config or MCTSConfig(), seed=seed,
                     policy="adjusted_softmax", temperature=temperature, tokenizer=tokenizer, on_event=emit)
    result.trace = events
    return result


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
            analysis = analyze_state(model, x, prompt_len, mask_id, config, legacy_entropy=True)
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
