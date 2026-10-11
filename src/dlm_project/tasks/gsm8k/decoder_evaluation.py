"""Paired GSM8K evaluation shared by four separate search-free decoders."""

import argparse
import copy
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import torch

from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.config import load_config, validate_config
from dlm_project.experiments.manifests import digest, get_manifest, select_examples
from dlm_project.experiments.paths import create_run_directory
from dlm_project.experiments.sweep_report import percentile, wilson_interval, write_csv
from dlm_project.models.profiles import get_profile
from dlm_project.paths import PROJECT_ROOT
from dlm_project.tasks.gsm8k.dataset import build_prompt
from dlm_project.tasks.gsm8k.evaluation import score_answer

METHODS = ("random_mdm", "topk_margin", "confidence_guided", "best_of_five")


def resolve_settings(method, source, *, profile=None, num_samples=None):
    """Validate a single-method run and resolve smoke/milestone sample count."""
    if method not in METHODS:
        raise ValueError(f"Unknown decoder: {method}")
    config = copy.deepcopy(source)
    evaluation = config.setdefault("evaluation", {})
    evaluation["profile"] = profile or evaluation.get("profile", "smoke")
    if evaluation["profile"] not in ("smoke", "milestone"):
        raise ValueError("evaluation.profile must be smoke or milestone.")
    evaluation.setdefault("sample_seed", 1)
    evaluation.setdefault("warmup", True)
    if type(evaluation["sample_seed"]) is not int or not 0 <= evaluation["sample_seed"] < 2**63:
        raise ValueError("evaluation.sample_seed must be an integer in [0, 2**63).")
    if type(evaluation["warmup"]) is not bool:
        raise ValueError("evaluation.warmup must be boolean.")
    maximum = 20 if evaluation["profile"] == "smoke" else 100
    default = 3 if evaluation["profile"] == "smoke" else 50
    count = num_samples if num_samples is not None else config["data"].get("num_samples")
    count = default if count is None else count
    if type(count) is not int or not 1 <= count <= maximum:
        raise ValueError(f"Select 1–{maximum} examples from the {evaluation['profile']} pool.")
    config["data"]["num_samples"] = count
    if config["model"].get("family", "llada") != "llada":
        raise ValueError("These baselines use the original LLaDA model family.")
    config["model"].setdefault("family", "llada")
    config["model"].setdefault("model_path", get_profile("llada")["repo_id"])
    if not isinstance(config.get("prompt", {}).get("instructions"), str) or not config["prompt"]["instructions"].strip():
        raise ValueError("prompt.instructions must be nonempty.")
    generation = config["generation"]
    config["method"] = method
    options = config.setdefault("decoding", {})
    seed = options.setdefault("seed", 1)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("decoding.seed must be an integer in [0, 2**63).")
    options.setdefault("top_k_positions", 1)
    options.setdefault("margin_gamma", 10.0)
    options.setdefault("best_of_five", {}).setdefault("score_block_size", 32)
    k = options["top_k_positions"]
    block = options["best_of_five"]["score_block_size"]
    if type(k) is not int or not 1 <= k <= generation["gen_length"]:
        raise ValueError("decoding.top_k_positions must be 1–gen_length.")
    if type(block) is not int or not 1 <= block <= generation["gen_length"]:
        raise ValueError("decoding.best_of_five.score_block_size must be 1–gen_length.")
    if method == "topk_margin" and generation["steps"] * k < generation["gen_length"]:
        raise ValueError("Top-K Margin needs steps * top_k_positions >= gen_length.")
    if method in ("topk_margin", "confidence_guided") and (
            generation["temperature"] != 0 or generation["block_length"] != generation["gen_length"]):
        raise ValueError("Top-K Margin and confidence decoding need temperature=0 and one output block.")
    if method == "confidence_guided" and generation["steps"] != generation["gen_length"]:
        raise ValueError("Confidence decoding needs steps=gen_length (one commitment per forward).")
    if method == "topk_margin" and generation["steps"] != generation["gen_length"]:
        # This keeps the configured maximum and the actual stopping behavior clear.
        raise ValueError("Top-K Margin needs steps=gen_length; K>1 can finish early.")
    from dlm_project.decoding.mcts_decoding import MCTSConfig
    MCTSConfig(margin_gamma=options["margin_gamma"])
    # Shared configuration validation still sees baseline (MCTS is not enabled).
    baseline_config = {**config, "decoding": {"method": "baseline", "seed": seed}}
    validate_config(baseline_config)
    return config


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _load_model(config):
    from dlm_project.models.llada import LLaDA
    return LLaDA(**config["model"])


def _verify_model(model):
    pinned = get_profile("llada")
    if any(model.identity.get(key) != pinned[key] for key in ("repo_id", "revision")):
        raise ValueError("The decoder evaluation requires the pinned original LLaDA checkpoint.")


def _decode(method, wrapper, prompt_ids, config, seed):
    from dlm_project.decoding import best_of_five, confidence_guided, random_mdm, topk_margin
    generation, options = config["generation"], config["decoding"]
    model, mask_id = wrapper.model, wrapper.mask_id
    shared = {"gen_length": generation["gen_length"]}
    if method == "random_mdm":
        return random_mdm.decode(model, prompt_ids, mask_id, **shared,
                                 steps=generation["steps"], block_length=generation["block_length"],
                                 temperature=generation["temperature"], seed=seed)
    if method == "topk_margin":
        return topk_margin.decode(model, prompt_ids, mask_id, **shared,
                                  steps=generation["steps"], top_k_positions=options["top_k_positions"])
    if method == "confidence_guided":
        return confidence_guided.decode(model, prompt_ids, mask_id, **shared,
                                        margin_gamma=options["margin_gamma"])
    return best_of_five.decode(model, prompt_ids, mask_id, **shared,
                               steps=generation["steps"], block_length=generation["block_length"],
                               temperature=generation["temperature"], seed=seed,
                               score_block_size=options["best_of_five"]["score_block_size"])


@torch.inference_mode()
def _warmup(model, prompt, length):
    ids = model.encode_prompt(prompt)
    state = torch.full((1, ids.shape[1] + length), model.mask_id,
                       dtype=torch.long, device=model.device)
    state[:, :ids.shape[1]] = ids
    prediction = model.model(state)
    del prediction
    _sync(model.device)


def _read_records(directory):
    records = []
    for line_number, line in enumerate((directory / "predictions.jsonl").read_text().splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid predictions.jsonl line {line_number}.") from error
    return records


def generate_report(directory):
    """Rebuild simple one-method metrics offline from completed JSONL records."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    state = json.loads((directory / "run_state.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if digest(manifest) != config["run"]["manifest_hash"]:
        raise ValueError("Manifest hash differs from saved run configuration.")
    records = _read_records(directory)
    selected = set(config["run"]["example_ids"])
    seen = set()
    for row in records:
        if row["example_id"] not in selected or row["example_id"] in seen or row["method"] != config["method"]:
            raise ValueError("Unexpected or repeated prediction in saved run.")
        if row["total_nfes"] != row["generation_nfes"] + row["scoring_nfes"]:
            raise ValueError("NFE accounting does not reconcile.")
        seen.add(row["example_id"])
    count = len(records)
    correct = sum(bool(row["exact_match"]) for row in records)
    low, high = wilson_interval(correct, count)
    times = [row["latency_seconds"] for row in records]
    memory = [row["peak_gpu_memory_bytes"] for row in records if row["peak_gpu_memory_bytes"] is not None]
    summary = {"status": state["status"] if count == len(selected) else "incomplete",
               "method": config["method"], "profile": config["evaluation"]["profile"],
               "model_revision": get_profile("llada")["revision"],
               "dataset_revision": manifest["dataset"]["revision"],
               "requested_examples": len(selected), "evaluated_n": count, "correct": correct,
               "accuracy": correct / count if count else None,
               "accuracy_ci95_low": low, "accuracy_ci95_high": high,
               "parse_rate": sum(row["parse_success"] for row in records) / count if count else None,
               "mean_generation_nfes": mean(row["generation_nfes"] for row in records) if count else None,
               "mean_scoring_nfes": mean(row["scoring_nfes"] for row in records) if count else None,
               "mean_total_nfes": mean(row["total_nfes"] for row in records) if count else None,
               "median_latency_seconds": median(times) if count else None,
               "p95_latency_seconds": percentile(times, .95),
               "max_gpu_memory_bytes": max(memory) if memory else None,
               "setup": {key: state.get(key) for key in ("model_load_count", "model_loading_seconds",
                                                      "warmup_nfes", "warmup_seconds")},
               "error": state.get("error")}
    write_csv(directory / "results.csv", records, ("example_id", "method"))
    write_csv(directory / "summary.csv", [summary])
    save_json(directory / "summary.json", summary)
    lines = [f"# GSM8K · {config['method']}", "", f"Status: **{summary['status']}**. "
             f"Evaluated **{count}/{len(selected)}** selected questions.", "",
             "| Accuracy | Wilson 95% interval | Parse rate | Mean generation NFEs | "
             "Mean scoring NFEs | Mean total NFEs | Median / p95 seconds | Peak GPU bytes |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|",
             f"| {summary['accuracy']} | [{low}, {high}] | {summary['parse_rate']} | "
             f"{summary['mean_generation_nfes']} | {summary['mean_scoring_nfes']} | "
             f"{summary['mean_total_nfes']} | {summary['median_latency_seconds']} / "
             f"{summary['p95_latency_seconds']} | {summary['max_gpu_memory_bytes']} |", "",
             "Exact match uses the last strictly parsed `#### <number>` line; parse failures count as wrong. "
             "NFEs count actual model forwards (five generations plus self-scoring for Best-of-5). "
             "Latency synchronizes CUDA and excludes model loading, warm-up and artifact/report writing. "
             "Masked-block selection scores never use reference answers.", "",
             "[Resolved configuration](config.json) · [Frozen manifest](manifest.json) · "
             "[Per-example CSV](results.csv) · [Raw predictions](predictions.jsonl) · "
             "[Machine-readable summary](summary.json)", ""]
    if summary["status"] != "completed":
        lines += ["This run is incomplete. Missing generations have no prediction row and are not scored as incorrect.", ""]
    if summary["error"]:
        lines += [f"Execution error: {summary['error']}", ""]
    temporary = directory / "report.md.tmp"
    temporary.write_text("\n".join(lines))
    temporary.replace(directory / "report.md")
    return summary


def run_evaluation(method, source, *, manifest_path=None, output_dir=None, model_factory=None):
    """Load one model and evaluate selected frozen examples sequentially."""
    config = resolve_settings(method, source)
    root = Path(config.get("experiment_root", "results")).expanduser()
    root = root if root.is_absolute() else PROJECT_ROOT / root
    manifest_path = (Path(manifest_path) if manifest_path else
                     root / "manifests" / f"gsm8k-seed{config['evaluation']['sample_seed']}.json")
    manifest = get_manifest(manifest_path, "gsm8k", sample_seed=config["evaluation"]["sample_seed"],
                            cache_dir=config["data"].get("cache_dir"), revision=config["data"].get("revision"))
    examples = select_examples(manifest, config["evaluation"]["profile"], config["data"]["num_samples"])
    config["data"]["revision"] = manifest["dataset"]["revision"]
    parent = Path(output_dir or config.get("output_dir") or root / "LLaDA/GSM8K/decoder_baselines" / method)
    directory, started = create_run_directory({"output_dir": str(parent),
                                               "experiment_timezone": config.get("experiment_timezone", "America/Los_Angeles")})
    config["run"] = {"directory": str(directory), "started_at": started.isoformat(),
                     "example_ids": [row["example_id"] for row in examples],
                     "manifest_hash": digest(manifest), "manifest_source": str(manifest_path.resolve())}
    save_json(directory / "config.json", config)
    save_json(directory / "manifest.json", manifest)
    (directory / "traces").mkdir()
    (directory / "predictions.jsonl").touch()
    state = {"status": "running", "completed_predictions": 0, "model_load_count": 0,
             "model_loading_seconds": None, "warmup_nfes": 0, "warmup_seconds": 0.0}
    save_json(directory / "run_state.json", state)
    print(f"Evaluation artifacts: {directory}", flush=True)
    try:
        started_loading = perf_counter()
        wrapper = (model_factory or _load_model)(config)
        _sync(wrapper.device)
        state["model_load_count"] = 1
        state["model_loading_seconds"] = perf_counter() - started_loading
        _verify_model(wrapper)
        config["run"]["model"] = wrapper.identity
        save_json(directory / "config.json", config)
        if config["evaluation"]["warmup"]:
            started_warmup = perf_counter()
            _warmup(wrapper, build_prompt(examples[0]["question"], config["prompt"]["instructions"]),
                    config["generation"]["gen_length"])
            state["warmup_nfes"] = 1
            state["warmup_seconds"] = perf_counter() - started_warmup
            save_json(directory / "run_state.json", state)
        with (directory / "predictions.jsonl").open("a") as predictions:
            for example in examples:
                prompt = build_prompt(example["question"], config["prompt"]["instructions"])
                seed = (config["decoding"]["seed"] + example["position"]) % 2**63
                state.update(active_example=example["example_id"])
                save_json(directory / "run_state.json", state)
                torch.manual_seed(seed)
                _sync(wrapper.device)
                if torch.device(wrapper.device).type == "cuda":
                    torch.cuda.reset_peak_memory_stats(wrapper.device)
                started_prediction = perf_counter()
                prompt_ids = wrapper.encode_prompt(prompt)
                result = _decode(method, wrapper, prompt_ids, config, seed)
                _sync(wrapper.device)
                latency = perf_counter() - started_prediction
                text = wrapper.tokenizer.batch_decode(
                    result.token_ids[:, prompt_ids.shape[1]:], skip_special_tokens=True)[0]
                score = score_answer(text, example["reference_solution"])
                trace_file = f"traces/example_{example['position']:04d}.jsonl"
                with (directory / trace_file).open("w") as trace:
                    for event in result.trace:
                        trace.write(json.dumps(event, allow_nan=False) + "\n")
                row = {"example_id": example["example_id"], "manifest_position": example["position"],
                       "partition": example["partition"], "method": method,
                       "model_revision": wrapper.identity["revision"],
                       "dataset_revision": manifest["dataset"]["revision"],
                       "seed": seed, "prompt_hash": digest(prompt), "prompt": prompt,
                       "content_hash": example["content_hash"], "raw_output": text,
                       **score, "generation_nfes": result.generation_nfes,
                       "scoring_nfes": result.scoring_nfes, "total_nfes": result.total_nfes,
                       "latency_seconds": latency,
                       "peak_gpu_memory_bytes": (torch.cuda.max_memory_allocated(wrapper.device)
                                                  if torch.device(wrapper.device).type == "cuda" else None),
                       "model_device": str(wrapper.device), "prompt_token_count": prompt_ids.shape[1],
                       "generation_length": config["generation"]["gen_length"],
                       "trace_file": trace_file, **result.details}
                predictions.write(json.dumps(row, allow_nan=False) + "\n")
                predictions.flush()
                state["completed_predictions"] += 1
                save_json(directory / "run_state.json", state)
                generate_report(directory)
                print(f"{row['example_id']} | {method} | exact={row['exact_match']} | "
                      f"NFEs={row['total_nfes']} | {latency:.2f}s", flush=True)
        state.update(status="completed", active_example=None)
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}")
        save_json(directory / "run_state.json", state)
        generate_report(directory)
        raise
    save_json(directory / "run_state.json", state)
    summary = generate_report(directory)
    print(f"Saved report: {directory / 'report.md'}", flush=True)
    return directory, summary


def main(method):
    parser = argparse.ArgumentParser(description=f"Pinned LLaDA GSM8K evaluation: {method}.")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "configs/gsm8k/llada_decoder_baselines.yaml")
    parser.add_argument("--profile", choices=("smoke", "milestone"))
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-only", type=Path, metavar="RUN_DIR")
    args = parser.parse_args()
    if args.report_only:
        saved = json.loads((args.report_only / "config.json").read_text())
        if saved["method"] != method:
            parser.error("The saved run belongs to another decoder.")
        generate_report(args.report_only)
        print(f"Saved report: {args.report_only / 'report.md'}")
        return
    config = resolve_settings(method, load_config(args.config), profile=args.profile,
                              num_samples=args.num_samples)
    run_evaluation(method, config, manifest_path=args.manifest, output_dir=args.output_dir)
