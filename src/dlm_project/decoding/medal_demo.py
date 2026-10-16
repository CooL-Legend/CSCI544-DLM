"""Single-prompt MEDAL demo: inspect initialization, then swap the completion sampler."""

import argparse
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from textwrap import fill
from time import perf_counter

import torch

from dlm_project.decoding.mcts_decoding import (
    MCTSConfig, MCTSSearch, SearchResult, complete_medal, derive_seed,
    information_gain, model_device,
)
from dlm_project.decoding.medal_prompting import get_gsm8k_demonstrations, gsm8k_prompt
from dlm_project.tasks.gsm8k.dataset import build_prompt


DEFAULT_QUESTION = (
    "A bookstore has 18 mystery books. It receives 3 boxes with 8 mystery books "
    "in each box, then sells 15 mystery books. How many mystery books are left?"
)
DEFAULT_INSTRUCTIONS = (
    "Solve this math problem in English.\n"
    "Use at most three short calculation lines.\n"
    "End with a final line containing exactly #### followed by a plain number.\n"
    "Do not include units, %, LaTeX, or any text after that number."
)


@dataclass
class InitializationResult:
    search: MCTSSearch
    result: SearchResult
    wall_seconds: float

    @property
    def state(self):
        """Selected prompt-plus-partial-output state; completion receives a copy."""
        return self.result.selected.state


@dataclass
class DecodingResult:
    token_ids: torch.Tensor
    model_calls: int
    wall_seconds: float
    method: str
    trace: list[dict]


def _sync(model):
    device = model_device(model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_demo_model(**settings):
    """Load once from a Drive/local directory or the pinned Hub checkpoint."""
    from dlm_project.models.llada import LLaDA

    started = perf_counter()
    wrapper = LLaDA(**settings)
    _sync(wrapper.model)
    print(f"Model ready: {wrapper.identity['model_path']} on {wrapper.device}", flush=True)
    print(f"Model loading: {perf_counter() - started:.2f}s (separate from decoding)", flush=True)
    return wrapper


def prepare_prompt(question, demonstrations_path, *, revision=None, cache_dir=None,
                   instructions=DEFAULT_INSTRUCTIONS):
    """Use the current two-shot prompt without loading a test set or scoring answers."""
    path = Path(demonstrations_path).expanduser()
    if revision is None:
        # Reuse the existing Drive demonstrations/revision, including offline.
        for source in (path, path.with_name("gsm8k_test.json")):
            if source.is_file():
                revision = json.loads(source.read_text(encoding="utf-8"))["dataset"]["revision"]
                break
    if revision is None:
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info("openai/gsm8k").sha
    demonstrations = get_gsm8k_demonstrations(path, revision=revision, cache_dir=cache_dir)
    return gsm8k_prompt(build_prompt(question, instructions), question, demonstrations)


def show_prompt(question):
    print(f"Prompt:\n{question.strip()}\n")
    print("Inline decomposition template (generated reasoning appears in the final output):")
    print("1. Identify the quantities and the quantity being asked for.")
    print("2. Calculate using the relevant quantities.")
    print("3. State the resulting quantity.")
    print("Planning NFE: 0; the model generates these steps within its answer.", flush=True)


@torch.inference_mode()
def run_initialization(model, prompt_ids, mask_id, *, gen_length=256, mcts=None,
                       seed=1, tokenizer=None):
    _sync(model)
    started = perf_counter()
    search = MCTSSearch(model, prompt_ids, mask_id, gen_length, mcts,
                        tokenizer=tokenizer, seed=seed)
    result = search.run()
    _sync(model)
    return InitializationResult(search, result, perf_counter() - started)


def candidate_summaries(initialization, *, tokenizer=None, limit=3):
    """Read cached statistics only; Q is the score used by the existing search."""
    search, result = initialization.search, initialization.result
    tokenizer = tokenizer or search.tokenizer
    root_entropy = search.root.analysis.total_entropy
    candidates = sorted(result.candidates, key=lambda node: -node.mean_reward)[:limit]
    rows = []
    for rank, node in enumerate(candidates, 1):
        actions, current = [], node
        while current.parent is not None:
            action = current.action
            text = (tokenizer.decode([action.token_id], skip_special_tokens=False)
                    if tokenizer is not None else str(action.token_id))
            actions.append({"position": action.output_position, "token_id": action.token_id,
                            "token": text, "confidence": action.confidence_score})
            current = current.parent
        entropy = node.analysis.total_entropy
        rows.append({"rank": rank, "node_id": node.node_id, "selected": node is result.selected,
                     "value": node.mean_reward, "depth": node.depth, "entropy": entropy,
                     "entropy_reduction": root_entropy - entropy,
                     "relative_entropy_reduction": information_gain(root_entropy, entropy, search.config.epsilon),
                     "tokens": sorted(actions, key=lambda action: action["position"])})
    return rows


def show_candidates(initialization, *, tokenizer=None, limit=3):
    rows = candidate_summaries(initialization, tokenizer=tokenizer, limit=limit)
    print(f"Top {len(rows)} of {len(initialization.result.candidates)} initialization candidates")
    print("Q = cached selection value (last-edge normalized entropy reduction).")
    print("Tokens: zero-based output position, decoded value, adjusted confidence s.")
    for row in rows:
        marker = " [selected]" if row["selected"] else ""
        print(f"\nC{row['rank']}{marker}: Q={row['value']:+.6f} | {row['depth']} unmasked tokens")
        tokens = "; ".join(f"{a['position']}:{a['token']!r} (s={a['confidence']:.4g})"
                           for a in row["tokens"])
        print(fill(tokens, width=110, initial_indent="  ", subsequent_indent="  "))
        print(f"  H={row['entropy']:.4f} nats | root-to-candidate reduction="
              f"{row['entropy_reduction']:+.4f} nats ({row['relative_entropy_reduction']:+.2%})")
    print(flush=True)


class _CountForwards:
    def __init__(self, model):
        self.model, self.calls = model, 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.model(*args, **kwargs)


def _resolve_sampler(method):
    if callable(method):
        return method
    if method == "medal":
        return complete_medal
    module_name, separator, function_name = method.partition(":")
    if not separator:
        raise ValueError("decoding method must be 'medal', 'module:function', or a callable.")
    return getattr(importlib.import_module(module_name), function_name)


@torch.inference_mode()
def run_completion(initialization, *, method="medal", seed=1, temperature=1.0,
                   tokenizer=None):
    """Run a sampler on a fresh copy; changing method never reruns initialization.

    A custom module:function accepts the complete_medal signature (without
    on_event) and returns an object with token_ids and an optional trace.
    Actual forwards are counted here, independent of the sampler's counters.
    """
    search = initialization.search
    sampler = _resolve_sampler(method)
    counted = _CountForwards(search.model)
    _sync(search.model)
    started = perf_counter()
    output = sampler(counted, initialization.state.clone(), search.prompt_len, search.mask_id,
                     search.config, seed=derive_seed(seed, "final_completion"),
                     temperature=temperature, tokenizer=tokenizer or search.tokenizer)
    _sync(search.model)
    seconds = perf_counter() - started
    ids = output.token_ids.detach().cpu()
    if ids.shape != initialization.state.shape or ids.dtype != torch.long:
        raise ValueError("The sampler must return int64 token_ids with shape [1, prompt + output length].")
    fixed = initialization.state != search.mask_id
    fixed[:, :search.prompt_len] = True
    if not torch.equal(ids[fixed], initialization.state[fixed]):
        raise ValueError("The sampler changed the prompt or an MCTS commitment.")
    if (ids[:, search.prompt_len:] == search.mask_id).any():
        raise ValueError("The sampler left masked output positions.")
    name = method if isinstance(method, str) else method.__name__
    return DecodingResult(ids, counted.calls, seconds, name, getattr(output, "trace", []))


def show_metrics(initialization, completion=None):
    """Only inference costs; reading/printing this report performs no forwards."""
    stats = initialization.result.stats
    phases = [
        ("Root analysis", stats["tree_analysis_model_calls"]),
        ("Selection", 0),
        ("Expansion", 0),
        ("Simulation / child evaluation", stats["simulation_model_calls"]),
        ("Backpropagation", 0),
        ("Candidate ranking (cached)", stats["candidate_evaluation_model_calls"]),
    ]
    for phase, count in phases:
        print(f"{phase:<32} {count:>6} NFE")
    search_nfe = stats["search_model_calls"]
    print(f"{'Initialization total':<32} {search_nfe:>6} NFE")
    print(f"Initialization wall clock: {initialization.wall_seconds:.3f}s | "
          f"{stats['mcts_simulations']} child evaluations")
    total_nfe, seconds = search_nfe, initialization.wall_seconds
    if completion is not None:
        total_nfe += completion.model_calls
        seconds += completion.wall_seconds
        print(f"Completion ({completion.method}): {completion.model_calls} NFE | {completion.wall_seconds:.3f}s")
        print("Final remaining-mask entropy: 0 nats (all output positions committed).")
    print(f"Total NFE: {total_nfe} | inference wall clock: {seconds:.3f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Drive/local snapshot or Hub ID; defaults to pinned LLaDA.")
    parser.add_argument("--family", choices=("llada", "illada"), default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    prompt_source = parser.add_mutually_exclusive_group()
    prompt_source.add_argument("--question", default=DEFAULT_QUESTION)
    prompt_source.add_argument("--prompt-file", type=Path, help="UTF-8 file containing the question.")
    parser.add_argument("--demonstrations-path", type=Path,
                        default=Path(".cache/medal/gsm8k_decomposition_train.json"))
    parser.add_argument("--dataset-revision", help="Optional immutable GSM8K revision for the two training examples.")
    parser.add_argument("--cache-dir", help="Optional Hugging Face dataset cache.")
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--init-tokens", type=int, default=20)
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--top-k-tokens", type=int, default=3)
    parser.add_argument("--top-k-actions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--decoding-method", default="medal", help="medal or importable module:function.")
    args = parser.parse_args()
    config = MCTSConfig(init_tokens=args.init_tokens, num_candidates=args.num_candidates,
                        top_k_tokens=args.top_k_tokens, top_k_actions=args.top_k_actions)
    config.validate_length(args.gen_length)
    _resolve_sampler(args.decoding_method)
    question = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.question
    if not question.strip():
        parser.error("The question must be nonempty.")

    wrapper = load_demo_model(model_path=args.model_path, family=args.family, device=args.device,
                              dtype=args.dtype, local_files_only=args.local_files_only)
    prompt, _ = prepare_prompt(question, args.demonstrations_path,
                               revision=args.dataset_revision, cache_dir=args.cache_dir)
    prompt_ids = wrapper.encode_prompt(prompt)
    show_prompt(question)

    print("\nRunning MCTS initialization...", flush=True)
    initialization = run_initialization(wrapper.model, prompt_ids, wrapper.mask_id,
                                         gen_length=args.gen_length, mcts=config,
                                         seed=args.seed, tokenizer=wrapper.tokenizer)
    show_candidates(initialization)
    show_metrics(initialization)

    print(f"\nCompleting the selected state with {args.decoding_method}...", flush=True)
    completion = run_completion(initialization, method=args.decoding_method, seed=args.seed,
                                temperature=args.temperature, tokenizer=wrapper.tokenizer)
    print("\nGenerated response:\n" + wrapper.tokenizer.decode(
        completion.token_ids[0, prompt_ids.shape[1]:].tolist(), skip_special_tokens=True))
    print()
    show_metrics(initialization, completion)


if __name__ == "__main__":
    main()
