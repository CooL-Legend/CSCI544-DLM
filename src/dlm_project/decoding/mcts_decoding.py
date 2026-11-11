"""Documented reconstruction of MEDAL Algorithms 1 and 2.

Immediate parent-to-child information gain, cached candidate ranking,
and literal adjusted-score softmax completion. The interpretation
choices are exported in RECONSTRUCTION_ASSUMPTIONS; author-code parity is not
claimed. Optional AdaLook-inspired confidence rewards and temporary adaptive
rollout are explicitly configured; trees never accept reference answers.
"""

import copy
from dataclasses import asdict, dataclass, field
import math
import hashlib
from contextlib import contextmanager, nullcontext
from time import perf_counter
from typing import Callable

import torch


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
    reward_mode: str = "ig"
    adalook_alpha: float = 0.1
    adalook_confidence_threshold: float = 0.8
    adalook_max_depth: int = 2
    adalook_variance_threshold: float = 0.01
    adalook_trigger_threshold: float = 0.2
    adalook_min_masks: int = 2

    def __post_init__(self):
        if self.reward_mode not in ("ig", "adalook_one_step", "adalook_adaptive"):
            raise ValueError("Unknown MCTS reward_mode.")
        if self.reward_mode == "adalook_adaptive" and self.stop_mode != "budget":
            raise ValueError("Adaptive AdaLook requires budget stopping.")
        for name in ("adalook_max_depth", "adalook_min_masks"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("adalook_alpha", "adalook_confidence_threshold", "adalook_variance_threshold", "adalook_trigger_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}.")
        if not 0 < self.adalook_confidence_threshold <= 1 or self.adalook_trigger_threshold > 1:
            raise ValueError("AdaLook confidence thresholds must be probabilities.")
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
    positions: torch.Tensor | None = None
    confidences: torch.Tensor | None = None
    proposed_tokens: torch.Tensor | None = None
    confidence_mass: float = 0.0

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
    evaluation_order: int | None = None
    ig_reward: float | None = None

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


CANDIDATE_SELECTION_VERSION = "deepest_with_c_distinct_mean_q_v2"


def select_candidate_depth(distinct_counts, limit):
    """Deepest depth with >= limit distinct states, else the most-populated depth.

    distinct_counts maps depth -> number of distinct evaluated states. Fallback
    ties prefer the deeper depth. Shared by live search and offline audits.
    """
    if not distinct_counts or type(limit) is not int or limit < 1:
        raise ValueError("Depth selection requires distinct-state counts and a positive limit.")
    qualified = [depth for depth, count in distinct_counts.items() if count >= limit]
    if qualified:
        return max(qualified)
    return max(distinct_counts, key=lambda depth: (distinct_counts[depth], depth))


def select_deepest_candidates(nodes, limit=3):
    """Distinct states at the deepest depth holding >= limit of them; mean-Q ranking.

    limit is C. Depth d* is the deepest depth with at least C distinct evaluated
    states; when no depth qualifies, d* is the depth with the most distinct
    states (deeper wins ties). States at d* use first-occurrence deduplication
    and are ranked by backed-up mean reward. The analyzed root is the only
    eligible zero-visit state. This helper does not mutate the tree, pool
    transpositions, or invoke the model.
    """
    eligible = [n for n in nodes if n.analysis is not None
                and (n.parent is None or n.immediate_reward is not None)]
    if not eligible or type(limit) is not int or limit < 1:
        raise ValueError("Selection requires evaluated nodes and a positive shortlist limit.")
    by_depth = {}
    for node in sorted(eligible, key=lambda n: n.node_id):
        by_depth.setdefault(node.depth, {}).setdefault(_state_key(node.state), node)
    depth = select_candidate_depth({d: len(states) for d, states in by_depth.items()}, limit)
    candidates = list(by_depth[depth].values())
    shortlist = sorted(candidates, key=lambda n: (-n.mean_reward, n.node_id))[:limit]
    return candidates, shortlist


def candidate_value_record(node):
    return {"node_id": node.node_id, "depth": node.depth, "visits": node.visits,
            "reward_sum": node.reward_sum, "q_value": node.mean_reward,
            "reward": node.mean_reward, "immediate_gain": node.ig_reward if node.ig_reward is not None else node.immediate_reward,
            "state_hash": hashlib.sha256(repr(_state_key(node.state)).encode()).hexdigest()}


def _action_order(action):
    return (-action.confidence_score, action.output_position, action.token_id)


def _token_topk(probabilities, k, mask_id, excluded_token_ids=()):
    """Top-k per row, resolving boundary ties by smaller token ID.

    Usually only k+1 values per row leave the GPU. Only a tied cutoff needs a
    scan; we avoid sorting the entire LLaDA vocabulary for every position.
    """
    legal = probabilities.clone()
    excluded = set(excluded_token_ids) | {mask_id}
    if any(type(token) is not int or not 0 <= token < legal.shape[-1] for token in excluded):
        raise ValueError("Excluded search token IDs must be integers within the model vocabulary.")
    available = legal.shape[-1] - len(excluded)
    if not available:
        raise ValueError("Search exclusions leave no legal token.")
    legal[:, sorted(excluded)] = -torch.inf
    k = min(k, available)
    count = min(k + 1, available)
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
                  search_excluded_token_ids=()):
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
    timing = (model.phase("candidate_filtering")
              if isinstance(model, _SearchModel) and model.current_phase == "simulation" else nullcontext())
    return distribution_from_logits(logits, state, prompt_len, mask_id, config, masked_only=True,
                                    legacy_entropy=legacy_entropy,
                                    search_excluded_token_ids=search_excluded_token_ids,
                                    filtering_timer=timing)


def distribution_from_logits(logits, state, prompt_len, mask_id, config, *,
                             legacy_entropy=False, search_excluded_token_ids=(),
                             filtering_timer=None, masked_only=False):
    """Shared serial/batched analysis; no model call and no retained logits."""
    state = state.detach().cpu()
    positions = torch.nonzero(state[0, prompt_len:] == mask_id, as_tuple=False).flatten()
    entropies = torch.zeros(state.shape[1] - prompt_len)
    if not positions.numel():
        return StateAnalysis(entropies, (), _state_key(state)), positions, None, None
    logits = logits.float() if masked_only else logits[(positions + prompt_len).to(logits.device)].float()
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
    with filtering_timer if filtering_timer is not None else nullcontext():
        token_rows = _token_topk(probabilities, config.top_k_tokens, mask_id,
                                search_excluded_token_ids)
        h_cpu, margins_cpu, factors_cpu = h.cpu(), margins.cpu().tolist(), factors.cpu().tolist()
        entropies[positions] = h_cpu
        actions = []
        for row, position in enumerate(positions.tolist()):
            for probability, token_id in token_rows[row]:
                actions.append(Action(position, token_id, probability * factors_cpu[row],
                                      probability, h_cpu[row].item(), margins_cpu[row]))
        actions.sort(key=_action_order)
    analysis = StateAnalysis(entropies, tuple(actions[:config.top_k_actions]), _state_key(state), margins.mean().item())
    if config.reward_mode != "ig":
        confidence, tokens = probabilities.max(dim=-1)
        legal = tokens != mask_id
        for token in search_excluded_token_ids:
            legal &= tokens != token
        analysis.positions = positions.cpu()
        analysis.confidences = confidence.detach().cpu()
        analysis.proposed_tokens = tokens.detach().cpu()
        eligible = legal & (confidence >= config.adalook_confidence_threshold)
        analysis.confidence_mass = confidence[eligible].double().sum().item()
    return analysis, positions, probabilities, factors


@torch.inference_mode()
def analyze_state(model, state, prompt_len, mask_id, config, *, force_forward=False,
                  legacy_entropy=False, search_excluded_token_ids=()):
    """Return compact CPU analysis; completed states need no forward.

    force_forward remains accepted for source compatibility, but no longer
    causes unnecessary inference on a completed state.
    """
    return _distribution(model, state, prompt_len, mask_id, config,
                         legacy_entropy=legacy_entropy,
                         search_excluded_token_ids=search_excluded_token_ids)[0]


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
        self.counts = {name: 0 for name in ("tree_analysis", "simulation", "candidate_evaluation", "adalook_rollout")}
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
def evaluate_action(model, parent, child, prompt_len, mask_id, config, *,
                    search_excluded_token_ids=()):
    """Use the cached parent and one child forward to compute the edge reward."""
    if parent.analysis is None or child.analysis is not None:
        raise ValueError("Evaluation requires a cached parent and an unevaluated child.")
    analysis = analyze_state(model, child.state, prompt_len, mask_id, config,
                             search_excluded_token_ids=search_excluded_token_ids)
    return apply_child_analysis(parent, child, analysis, prompt_len, mask_id, config,
                                signal_nfe=getattr(model, "calls", None))


def apply_child_analysis(parent, child, analysis, prompt_len, mask_id, config, *, signal_nfe=None):
    """Apply a completed evaluation using the same IG rule for every executor."""
    if parent.analysis is None or child.analysis is not None:
        raise ValueError("Evaluation requires a cached parent and an unevaluated child.")
    before, after = parent.analysis.total_entropy, analysis.total_entropy

    gain = information_gain(before, after, config.epsilon)
    reward = gain
    if config.reward_mode != "ig":
        from dlm_project.decoding.adalook_rollout import score
        reward = score(child.action.probability, analysis.confidence_mass, config)

    if not math.isfinite(reward):
        raise ValueError("Non-finite child reward.")
    child.analysis, child.immediate_reward = analysis, reward
    child.ig_reward = gain
    remaining = int((child.state[0, prompt_len:] == mask_id).sum().item())
    removed = parent.analysis.entropies[child.action.output_position].item()
    return {"entropy_before": before, "entropy_after": after,
            "immediate_gain": information_gain(before, after, config.epsilon),
            "reward": reward, "reward_method": config.reward_mode,
            "reward_alpha": config.adalook_alpha, "reward_confidence_threshold": config.adalook_confidence_threshold,
            "reward_confidence_mass": analysis.confidence_mass,
            "evaluation_model_calls": int(remaining > 0),
            "masked_count": remaining, "mean_token_entropy": after / remaining if remaining else 0.0,
            "mean_confidence_margin": analysis.mean_margin,
            "signal_nfe": signal_nfe, "signal_extra_nfes": 0,
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
                 tokenizer=None, on_event: Callable[[dict], None] | None = None, *, seed=1,
                 search_excluded_token_ids=(), checkpoint_budgets=None):
        self.config = config if isinstance(config, MCTSConfig) else MCTSConfig(**(config or {}))
        self.config.validate_length(gen_length)
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.dtype != torch.long:
            raise ValueError("prompt_ids must be an int64 tensor with shape [1, P].")
        derive_seed(seed, "validation")
        if any(type(token) is not int or token < 0 for token in search_excluded_token_ids):
            raise ValueError("Search exclusions must be nonnegative integer token IDs.")
        self.search_excluded_token_ids = tuple(sorted(set(search_excluded_token_ids)))
        self.seed = seed
        self.model, self.mask_id, self.tokenizer = model, mask_id, tokenizer
        self.prompt_len = prompt_ids.shape[1]
        state = torch.full((1, self.prompt_len + gen_length), mask_id, dtype=torch.long)
        state[:, :self.prompt_len] = prompt_ids.detach().cpu()
        self.root = Node(0, state)
        self.nodes, self.candidates = [self.root], {}
        self.trace, self.on_event = [], on_event
        self.cycle, self.stop_reason = 0, None
        self.simulations_started = self.simulations_completed = 0
        self.checkpoints = {}
        self.checkpoint_budgets = tuple(checkpoint_budgets) if checkpoint_budgets is not None else getattr(self.config, "checkpoint_budgets", ())
        self._counted = _SearchModel(model, self.config.max_model_calls, self._emit)
        self._adalook_driver = None

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

    def _snapshot_checkpoint(self, budget, stop_reason="model_call_budget"):
        evaluated_source = [n for n in self.nodes if n.evaluation_order is not None]
        evaluated = copy.deepcopy(evaluated_source)
        candidates, shortlist = select_deepest_candidates(evaluated, limit=self.config.num_candidates)
        selected = shortlist[0]
        distinct_counts = {}
        for n in evaluated:
            distinct_counts.setdefault(n.depth, set()).add(_state_key(n.state))
        counts = {d: len(states) for d, states in distinct_counts.items()}

        selection_record = {
            "stage": "search_selection",
            "trace_schema_version": "search_trace_v1",
            "requested_depth": self.config.init_tokens,
            "requested_budget": budget,
            "candidate_ids": [n.node_id for n in candidates],
            "shortlist_ids": [n.node_id for n in shortlist],
            "selected_node_id": selected.node_id,
            "selected_depth": selected.depth,
            "stop_reason": stop_reason or "model_call_budget",
            "fallback": selected.depth < self.config.init_tokens,
            "candidate_selection_version": CANDIDATE_SELECTION_VERSION,
            "root_action_count": len(self.root.analysis.actions) if self.root.analysis else 0,
            "node_statistics": [{"node_id": n.node_id, "visits": n.visits,
                                 "reward_sum": n.reward_sum, "q_value": n.mean_reward}
                                for n in evaluated]
        }

        return {
            "budget": budget,
            "model_calls": min(self.model_calls, budget),
            "nodes": evaluated,
            "candidates": candidates,
            "shortlist": shortlist,
            "selected": selected,
            "selection": selection_record,
            "trace": copy.deepcopy(self.trace),
            "stop_reason": stop_reason or "model_call_budget",
            "distinct_candidates_by_depth": counts,
            "max_depth_reached": max(distinct_counts.keys(), default=0),
            "selected_depth": selected.depth,
            "selection_depth_gap": max(distinct_counts.keys(), default=0) - selected.depth,
            "selected_depth_candidate_count": counts.get(selected.depth, 0),
            "shortlist_node_ids": [n.node_id for n in shortlist],
            "selected_node_id": selected.node_id,
            "selected_q": selected.mean_reward,
            "selected_state_hash": hashlib.sha256(repr(_state_key(selected.state)).encode()).hexdigest(),
            "reached_target_depth": self.config.init_tokens in distinct_counts,
            "selection_below_target": selected.depth < self.config.init_tokens,
            "fallback_used": selected.depth < self.config.init_tokens,
        }

    def _check_checkpoint(self, stop_reason=None):
        if not self.checkpoint_budgets:
            return
        if self.model_calls in self.checkpoint_budgets and self.model_calls not in self.checkpoints:
            self.checkpoints[self.model_calls] = self._snapshot_checkpoint(self.model_calls, stop_reason=stop_reason)
            self._emit("search_checkpoint", budget=self.model_calls,
                       selected_node_id=self.checkpoints[self.model_calls]["selected_node_id"],
                       selected_depth=self.checkpoints[self.model_calls]["selected_depth"])

    def _record_evaluated(self, node):
        """Compact, model-free identity of each successfully evaluated state."""
        node.evaluation_order = self.simulations_completed if node.parent else 0
        self._emit("node_evaluated", trace_schema_version="search_trace_v1",
                   node_id=node.node_id, parent_id=node.parent.node_id if node.parent else None,
                   depth=node.depth, evaluation_order=node.evaluation_order,
                   state_hash=candidate_value_record(node)["state_hash"],
                   immediate_reward=node.immediate_reward,
                   immediate_gain=node.ig_reward, reward_mode=self.config.reward_mode,
                   action=None if node.action is None else asdict(node.action))

    def _close_exhausted(self, node):
        while node is not None:
            node.closed = node.depth >= self.config.init_tokens or (
                node.expanded and all(child.closed for child in node.children))
            node = node.parent

    def _at_cap(self):
        return self.config.max_model_calls is not None and self.model_calls >= self.config.max_model_calls

    def _at_simulation_cap(self):
        return self.config.max_simulations is not None and self.simulations_completed >= self.config.max_simulations

    def _assumptions(self):
        assumptions = dict(RECONSTRUCTION_ASSUMPTIONS)
        if self.config.reward_mode != "ig":
            assumptions.update(reward=self.config.reward_mode,
                simulation="MEDAL actions; AdaLook-inspired confidence scoring; temporary adaptive rollout only when enabled",
                adalook="MCTS hybrid, not a reproduction of the ETE/block decoder; global masked frontier")
        if self.config.stop_mode == "budget":
            assumptions.update(
                quota="stop at the forward cap or exhausted frontier; no candidate quota",
                candidate_evaluation=("deepest depth with at least C distinct evaluated states (else the most "
                                      "populated depth, deeper on ties); first-occurrence deduplication; "
                                      "cached mean rewards"),
            )
        return assumptions

    def _finish(self, reason):
        with self._counted.phase("candidate_evaluation"):
            candidates = (select_deepest_candidates(self.nodes, self.config.num_candidates)[0]
                          if self.config.stop_mode == "budget" else self.candidates.values())
            for node in candidates:
                self._emit("candidate_evaluation", phase="candidate_evaluation", node_id=node.node_id,
                           reward=node.mean_reward, immediate_gain=node.ig_reward,
                           visits=node.visits, reward_sum=node.reward_sum, q_value=node.mean_reward, model_calls=0,
                           state=state_record(node.state, self.prompt_len, self.mask_id, self.tokenizer))
            self.stop_reason = reason
            result = self.result()
        for b in self.checkpoint_budgets:
            if b not in self.checkpoints:
                self.checkpoints[b] = self._snapshot_checkpoint(b, stop_reason=reason)
        result.stats["candidate_evaluation_latency_seconds"] = self._counted.seconds["candidate_evaluation"]
        self._emit("search_selection", trace_schema_version="search_trace_v1",
                   requested_depth=self.config.init_tokens, requested_budget=self.config.max_model_calls,
                   candidate_ids=result.stats["candidate_ids"],
                   shortlist_ids=[row["node_id"] for row in result.stats["candidate_shortlist"]],
                   selected_node_id=result.selected.node_id, selected_depth=result.selected.depth,
                   stop_reason=reason, fallback=result.selected.depth < self.config.init_tokens,
                   candidate_selection_version=result.stats["candidate_selection_version"],
                   root_action_count=len(self.root.analysis.actions),
                   node_statistics=[{"node_id": n.node_id, "visits": n.visits,
                                     "reward_sum": n.reward_sum, "q_value": n.mean_reward}
                                    for n in self.nodes if n.evaluation_order is not None])
        self._emit("termination", **{k:v for k,v in result.stats.items() if "seconds" not in k},
                   selected_state=state_record(result.selected.state, self.prompt_len, self.mask_id, self.tokenizer))
        if self.config.stop_mode == "candidates" and result.stats["fallback_used"]:
            raise SearchIncompleteError(result.stats)

    @torch.inference_mode()
    def step(self):
        if self.config.reward_mode == "adalook_adaptive":
            start = len(self.trace)
            cycle = self.cycle
            while not self.done:
                request = self.next_evaluation()
                if request is None:
                    break
                self.record_external_forward(request, invocation_id=self.model_calls + 1, batch_size=1)
                with self._counted.phase(request.phase):
                    output = self.model(request.node.state.to(model_device(self.model)))
                self.submit_evaluation(request, logits=output.logits[0])
                del output
                if self.cycle > cycle:
                    break
            return self.trace[start:]
        if self.done:
            return []
        start = len(self.trace)
        self.cycle += 1
        if self.root.analysis is None:
            with self._counted.phase("tree_analysis"):
                self.root.analysis = analyze_state(self._counted, self.root.state, self.prompt_len,
                                                   self.mask_id, self.config,
                                                   search_excluded_token_ids=self.search_excluded_token_ids)
            self._record_evaluated(self.root)
            self._emit("initialization", node_id=0, entropy=self.root.analysis.total_entropy,
                       masked_count=self.root.state.shape[1] - self.prompt_len,
                       mean_token_entropy=self.root.analysis.total_entropy / (self.root.state.shape[1] - self.prompt_len),
                       mean_confidence_margin=self.root.analysis.mean_margin,
                       signal_nfe=self.model_calls, signal_extra_nfes=0,
                       implementation_version=IMPLEMENTATION_VERSION, assumptions=self._assumptions(),
                       model_device=str(model_device(self.model)), search_state_device="cpu",
                       entropies=self.root.analysis.entropies.tolist(),
                       state=state_record(self.root.state, self.prompt_len, self.mask_id, self.tokenizer),
                       actions=[action_record(a, self.tokenizer) for a in self.root.analysis.actions])
            self._check_checkpoint()
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
            if action.token_id in self.search_excluded_token_ids:
                raise RuntimeError("Excluded termination token reached search expansion.")
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
                                                 self.config, search_excluded_token_ids=self.search_excluded_token_ids)
            except BudgetExhausted:
                child.closed = True
                self._emit("simulation_interrupted", phase="simulation", child_id=child.node_id)
                self._finish("model_call_budget")
                return self.trace[start:]
            self.simulations_completed += 1
            self._record_evaluated(child)
            self._emit("evaluation", parent_id=parent.node_id, child_id=child.node_id,
                       depth=child.depth, action=action_record(action, self.tokenizer),
                       child_state=state_record(child.state, self.prompt_len, self.mask_id, self.tokenizer), **evaluation)
            with self._counted.phase("backpropagation"):
                updates = backpropagate(child, evaluation["reward"])
            self._emit("backpropagation", child_id=child.node_id, reward=evaluation["reward"], updates=updates)
            self._check_checkpoint()
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
        target_candidates = list(self.candidates.values())
        candidates = target_candidates
        evaluated = [node for node in self.nodes if node.immediate_reward is not None]
        if self.config.stop_mode == "budget":
            candidates, shortlist = select_deepest_candidates(self.nodes, self.config.num_candidates)
            selected = shortlist[0]
        elif candidates:
            selected = max(candidates, key=lambda n: n.mean_reward)
        else:
            selected = min(evaluated or [self.root], key=lambda n: (-n.depth, -n.mean_reward, n.node_id))
        score = selected.mean_reward
        complete = len(target_candidates) >= self.config.num_candidates
        fallback = self.done and not complete
        parents = len({node.parent.node_id for node in evaluated})
        stats = {"implementation_version": IMPLEMENTATION_VERSION, "paper_url": PAPER_URL,
                 "reconstruction_assumptions": self._assumptions(),
                 "evaluation_variant": "budget-limited MEDAL reconstruction (immediate-child IG)" if self.config.stop_mode == "budget"
                 else "MEDAL reconstruction (immediate-child IG)",
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
                 "max_depth": max((n.depth for n in evaluated), default=0), "candidate_target": self.config.num_candidates,
                 "candidate_count": len(candidates), "evaluated_candidates": len(candidates),
                 "target_depth_candidate_count": len(target_candidates),
                 "evaluated_children": len(evaluated), "expanded_parents": parents,
                 "realized_branching_factor": len(evaluated) / parents if parents else 0.0,
                 "candidate_ids": [n.node_id for n in candidates], "selected_node_id": selected.node_id,
                 "selected_score": score, "fallback_used": fallback,
                 "fallback_reason": self.stop_reason if fallback else None, "budget_exhausted": self._at_cap(),
                 "candidate_evaluations": [{**candidate_value_record(n), "model_calls": 0,
                     **state_record(n.state, self.prompt_len, self.mask_id, self.tokenizer)} for n in candidates]}
        # Additive metadata: preserve legacy fallback_used (quota shortfall) for
        # old callers. E0 distinguishes that from actually decoding below Lc.
        shortlist = sorted(candidates, key=lambda n: (-n.mean_reward, n.node_id))[:self.config.num_candidates]
        source = ("target_depth" if selected.depth == self.config.init_tokens
                  else "deepest_evaluated" if evaluated else "root")
        stats.update(search_excluded_token_ids=list(self.search_excluded_token_ids),
                     candidate_selection_version=CANDIDATE_SELECTION_VERSION if self.config.stop_mode == "budget" else "target_mean_q_v1",
                     candidate_shortlist=[candidate_value_record(n) for n in shortlist],
                     shortlist_size=len(shortlist), shortlist_underfilled=len(shortlist) < self.config.num_candidates,
                     selection_source=source, fallback_decode_used=source != "target_depth")
        groups = [e for e in self.trace if e['stage'] == 'adalook_group']
        if self.config.reward_mode != 'ig':
            stats['evaluation_variant'] = 'MEDAL + ' + self.config.reward_mode + ' (hybrid)'
        stats.update(reward_mode=self.config.reward_mode,
                     reward_version="ig" if self.config.reward_mode == 'ig' else "medal_adalook_hybrid_v1",
                     adalook_config={k: v for k, v in asdict(self.config).items() if k.startswith('adalook_')},
                     adalook_extra_nfes=self._counted.counts['adalook_rollout'],
                     adalook_groups=groups,
                     adalook_mean_depth=sum(g['rollout_depth'] for g in groups) / len(groups) if groups else 1.0,
                     selected_ig_reward=selected.ig_reward)
        return SearchResult(selected, candidates, stats, list(self.trace))

    def next_evaluation(self):
        from dlm_project.decoding.adalook_rollout import AdaptiveDriver
        if self._adalook_driver is None:
            self._adalook_driver = AdaptiveDriver(self)
        return self._adalook_driver.next_evaluation()

    def record_external_forward(self, request, *, invocation_id, batch_size):
        return self._adalook_driver.record_external_forward(request, invocation_id, batch_size)

    def submit_evaluation(self, request, *, logits):
        return self._adalook_driver.submit_evaluation(request, logits)

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
def complete_llada(model, state, prompt_len, mask_id, config=None, *, seed=1,
                   temperature=1.0, tokenizer=None, on_event=None):
    """LLaDA Gumbel proposals and low-confidence remasking from a partial state."""
    from dlm_project.decoding.common import decode_masked

    events = []
    def emit(event):
        events.append(event)
        if on_event is not None:
            on_event(event)
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
    counted = CountCompletion()
    length = state.shape[1] - prompt_len
    result = decode_masked(counted, state[:, :prompt_len], mask_id,
                           selection_rule="token_probability", gen_length=length,
                           steps=length, block_length=length, initial_state=state,
                           temperature=temperature, seed=seed,
                           on_event=lambda event: emit({**event, "model_calls": event["generation_nfes"], "seed": seed}))
    return CompletionResult(result.token_ids.detach().cpu(), counted.calls, events)


def resolve_completion_sampler(method):
    if method == "llada":
        return complete_llada
    if method == "medal":
        return complete_medal
    raise ValueError("completion_method must be 'llada' or 'medal'.")


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
