"""Single-prompt MEDAL demo: inspect initialization, then swap the completion sampler."""

import argparse
from dataclasses import asdict, dataclass
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
from dlm_project.experiments.artifacts import save_json
from dlm_project.tasks.gsm8k.dataset import build_prompt
from dlm_project.tasks.gsm8k.evaluation import score_answer


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
    started = perf_counter()
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
    setup_seconds = perf_counter() - started
    prompt, metadata = gsm8k_prompt(build_prompt(question, instructions), question, demonstrations)
    metadata["setup_wall_seconds"] = setup_seconds
    return prompt, metadata


def prepare_demo_prompt(wrapper, question, demonstrations_path, **settings):
    """Time prompt rendering and tokenization separately from demonstration setup."""
    _sync(wrapper.model)
    started = perf_counter()
    prompt, metadata = prepare_prompt(question, demonstrations_path, **settings)
    prompt_ids = wrapper.encode_prompt(prompt)
    _sync(wrapper.model)
    metadata["prompt_wall_seconds"] = perf_counter() - started - metadata["setup_wall_seconds"]
    return prompt, prompt_ids, metadata


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


def show_metrics(initialization, completion=None, *, prompt_seconds=0.0,
                 setup_seconds=0.0, score=None, display=True):
    """Return JSON-ready metrics; reporting and answer scoring add no forwards."""
    stats = initialization.result.stats
    # Legacy simulation time includes filtering; move that portion to the
    # candidate-generation row, leaving the original stats compatible.
    filtering_seconds = stats["candidate_filtering_latency_seconds"]
    phases = [
        ("Selection", 0, stats["selection_latency_seconds"]),
        ("Candidate generation / filtering", stats["tree_analysis_model_calls"],
         stats["tree_analysis_latency_seconds"] + filtering_seconds),
        ("Child expansion", 0, stats["expansion_latency_seconds"]),
        ("Child evaluation / r_IG", stats["simulation_model_calls"],
         stats["simulation_latency_seconds"] - filtering_seconds),
        ("Full rollout (disabled)", 0, 0.0),
        ("Backpropagation", 0, stats["backpropagation_latency_seconds"]),
        ("Candidate ranking (cached)", stats["candidate_evaluation_model_calls"],
         stats["candidate_evaluation_latency_seconds"]),
    ]
    overhead = initialization.wall_seconds - sum(seconds for _, _, seconds in phases)
    phases.append(("Bookkeeping / trace overhead", 0, overhead))
    rows = [{"stage": "Prompt", "substage": "Preparation / tokenization", "nfe": 0,
             "wall_seconds": prompt_seconds}]
    rows += [{"stage": "Initialization", "substage": name, "nfe": nfe, "wall_seconds": seconds}
             for name, nfe, seconds in phases]
    decoding_nfe = completion.model_calls if completion is not None else 0
    decoding_seconds = completion.wall_seconds if completion is not None else 0.0
    if completion is not None:
        rows.append({"stage": "Decoding", "substage": f"Completion ({completion.method})",
                     "nfe": decoding_nfe, "wall_seconds": decoding_seconds})
    tree = {key: stats[key] for key in (
        "expanded_parents", "evaluated_children", "unique_tree_states", "average_search_depth",
        "realized_branching_factor", "selected_depth", "candidate_count", "candidate_target", "stop_reason")}
    tree["max_evaluated_depth"] = max((node.depth for node in initialization.search.nodes
                                       if node.immediate_reward is not None), default=0)
    total_nfe = stats["search_model_calls"] + decoding_nfe
    report = {"stages": rows, "initialization_nfe": stats["search_model_calls"],
              "initialization_wall_seconds": initialization.wall_seconds,
              "decoding_nfe": decoding_nfe, "decoding_wall_seconds": decoding_seconds,
              "total_nfe": total_nfe, "model_evaluations": total_nfe, "batch_size": 1,
              "total_wall_seconds": prompt_seconds + initialization.wall_seconds + decoding_seconds,
              "setup_wall_seconds": setup_seconds, "tree": tree, "score": score,
              "accuracy": float(score["exact_match"]) if score and score["exact_match"] is not None else None,
              "prompt_token_count": initialization.search.prompt_len,
              "timing_mode": "CUDA-synchronized" if model_device(initialization.search.model).type == "cuda" else "CPU"}
    if display:
        print(f"{'Stage':<16} {'Substage':<35} {'NFE':>6} {'Seconds':>10}")
        for row in rows:
            print(f"{row['stage']:<16} {row['substage']:<35} {row['nfe']:>6} {row['wall_seconds']:>10.6f}")
        print(f"Initialization total: {report['initialization_nfe']} NFE | {initialization.wall_seconds:.3f}s")
        print(f"Total NFE: {total_nfe} | pipeline wall clock: {report['total_wall_seconds']:.3f}s")
        print(f"Demonstration setup/cache: {setup_seconds:.3f}s (separate; model loading reported above)")
        print(f"Expanded parents: {tree['expanded_parents']} | evaluated children: {tree['evaluated_children']} | "
              f"unique states: {tree['unique_tree_states']} | model evaluations: {total_nfe} (batch size 1)")
        print(f"Evaluated depth (non-root): mean={tree['average_search_depth']:.3f}, "
              f"max={tree['max_evaluated_depth']} | average branching factor={tree['realized_branching_factor']:.3f}")
        print(f"Selected depth: {tree['selected_depth']} | candidates: {tree['candidate_count']}/{tree['candidate_target']} | "
              f"stop: {tree['stop_reason']}")
        if score is not None:
            accuracy = f"{report['accuracy']:.0%} (n=1)" if report["accuracy"] is not None else "unavailable (no reference)"
            print(f"Task accuracy: {accuracy} | parsed answer: {score['predicted_answer']}", flush=True)
    return report


def run_demo_batch(wrapper, demonstrations_path, manifest_path, output_dir, *, num_samples=10,
                   gen_length=256, mcts=None, seed=1, temperature=1.0, method="medal", cache_dir=None):
    """Optional ordered GSM8K test subset, reusing the demo's model and helpers."""
    from dlm_project.experiments.paths import create_run_directory
    from dlm_project.tasks.gsm8k.benchmark import example_seed
    from dlm_project.tasks.gsm8k.benchmark_dataset import get_manifest, select_examples

    config = mcts if isinstance(mcts, MCTSConfig) else MCTSConfig(**(mcts or {}))
    config.validate_length(gen_length)
    started = perf_counter()
    path = Path(demonstrations_path).expanduser()
    revision = json.loads(path.read_text())["dataset"]["revision"] if path.is_file() else None
    manifest = get_manifest(manifest_path, cache_dir=cache_dir, revision=revision)
    examples = select_examples(manifest, "smoke", num_samples)
    setup_seconds = perf_counter() - started
    directory, _ = create_run_directory({"output_dir": str(output_dir)})
    save_json(directory / "config.json", {"mcts": asdict(config), "gen_length": gen_length,
              "seed": seed, "temperature": temperature, "method": method if isinstance(method, str) else method.__name__,
              "dataset": manifest["dataset"], "model": getattr(wrapper, "identity", {}),
              "example_ids": [row["example_id"] for row in examples]})
    records = []

    def save_summary(status, error=None):
        count = len(records)
        metrics = [row["metrics"] for row in records]
        correct = sum(row["accuracy"] for row in metrics)
        stages = []
        for index, stage in enumerate(metrics[0]["stages"] if metrics else []):
            stages.append({**stage, "nfe": sum(row["stages"][index]["nfe"] for row in metrics),
                           "wall_seconds": sum(row["stages"][index]["wall_seconds"] for row in metrics)})
        summary = {"status": status, "requested": len(examples), "processed": count, "correct": int(correct),
                   "accuracy": correct / count if count else None, "stages": stages,
                   "total_nfe": sum(row["total_nfe"] for row in metrics),
                   "total_wall_seconds": sum(row["total_wall_seconds"] for row in metrics),
                   "mean_nfe": sum(row["total_nfe"] for row in metrics) / count if count else None,
                   "mean_wall_seconds": sum(row["total_wall_seconds"] for row in metrics) / count if count else None,
                   "totals_cover": "completed_examples",
                   "setup_wall_seconds": setup_seconds + sum(row["setup_wall_seconds"] for row in metrics),
                   "mean_tree": {key: sum(row["tree"][key] for row in metrics) / count if count else None
                                 for key in ("expanded_parents", "evaluated_children", "unique_tree_states",
                                             "average_search_depth", "realized_branching_factor")},
                   "max_evaluated_depth": max((row["tree"]["max_evaluated_depth"] for row in metrics), default=0),
                   "error": error}
        save_json(directory / "summary.json", summary)
        return summary

    save_summary("running")
    print(f"Batch results: {directory}", flush=True)
    try:
        with (directory / "predictions.jsonl").open("a", encoding="utf-8") as predictions:
            for example in examples:
                prompt, prompt_ids, metadata = prepare_demo_prompt(
                    wrapper, example["question"], path, revision=manifest["dataset"]["revision"], cache_dir=cache_dir)
                current_seed = example_seed(seed, example["row_id"])
                initial = run_initialization(wrapper.model, prompt_ids, wrapper.mask_id, gen_length=gen_length,
                                             mcts=config, seed=current_seed, tokenizer=wrapper.tokenizer)
                completion = run_completion(initial, method=method, seed=current_seed,
                                            temperature=temperature, tokenizer=wrapper.tokenizer)
                text = wrapper.tokenizer.decode(completion.token_ids[0, prompt_ids.shape[1]:].tolist(), skip_special_tokens=True)
                metrics = show_metrics(initial, completion, prompt_seconds=metadata["prompt_wall_seconds"],
                                       setup_seconds=metadata["setup_wall_seconds"],
                                       score=score_answer(text, example["reference_solution"]), display=False)
                record = {"example_id": example["example_id"], "seed": current_seed,
                          "question": example["question"], "raw_output": text, "metrics": metrics}
                predictions.write(json.dumps(record, allow_nan=False) + "\n")
                predictions.flush()
                records.append(record)
                summary = save_summary("running")
                print(f"{example['example_id']} | accuracy={summary['accuracy']:.3f} | "
                      f"NFE={metrics['total_nfe']} | {metrics['total_wall_seconds']:.2f}s", flush=True)
    except BaseException as error:
        save_summary("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     f"{type(error).__name__}: {error}")
        raise
    return directory, save_summary("completed")


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
