"""Comparable single-method GSM8K accuracy and inference-cost evaluations."""

import argparse
import copy
from datetime import timezone
import json
from pathlib import Path
from statistics import mean, median
import subprocess
from time import perf_counter

import torch

from dlm_project.decoding import best_of_5, confidence, random_mdm, top_k_margin
from dlm_project.decoding.common import validate_generation
from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.config import load_config, validate_config
from dlm_project.experiments.manifests import digest
from dlm_project.experiments.paths import create_run_directory
from dlm_project.models.profiles import get_profile
from dlm_project.paths import PROJECT_ROOT
from dlm_project.tasks.gsm8k.benchmark_dataset import PROFILE_SIZES, TEST_SIZE, get_manifest, select_examples
from dlm_project.tasks.gsm8k.dataset import build_prompt
from dlm_project.tasks.gsm8k.evaluation import extract_answer, score_answer


DECODERS = {"random_mdm": random_mdm.decode, "top_k_margin": top_k_margin.decode,
            "confidence": confidence.decode, "best_of_5": best_of_5.decode}


def resolve_settings(method, source, *, profile=None, num_samples=None, seed=None,
                     temperature=None, model_path=None, local_files_only=None):
    """Resolve CLI overrides without changing the caller's configuration."""
    if method not in DECODERS:
        raise ValueError(f"Unknown decoder: {method}")
    config = copy.deepcopy(source)
    evaluation = config.setdefault("evaluation", {})
    evaluation["profile"] = profile if profile is not None else evaluation.get("profile", "smoke")
    if evaluation["profile"] not in PROFILE_SIZES:
        raise ValueError("evaluation.profile must be smoke, initial, or full.")
    data = config.setdefault("data", {})
    count = num_samples if num_samples is not None else data.get("num_samples")
    data["num_samples"] = PROFILE_SIZES[evaluation["profile"]] if count is None else count
    if type(data["num_samples"]) is not int or not 1 <= data["num_samples"] <= TEST_SIZE:
        raise ValueError(f"data.num_samples must be between 1 and {TEST_SIZE}.")
    for key, expected in (("dataset_id", "openai/gsm8k"), ("dataset_config", "main"), ("split", "test")):
        if data.get(key, expected) != expected:
            raise ValueError("Decoder benchmarks use only openai/gsm8k main test.")
        data[key] = expected
    prompt = config.get("prompt", {})
    if not isinstance(prompt.get("instructions"), str) or not prompt["instructions"].strip():
        raise ValueError("prompt.instructions must be nonempty text.")
    if prompt.get("custom_question") is not None:
        raise ValueError("Custom questions are not supported by the GSM8K accuracy benchmark.")
    model = config.setdefault("model", {})
    model.setdefault("family", "llada")
    if model["family"] != "llada":
        raise ValueError("Decoder benchmarks require model.family: llada.")
    for key, value in (("model_path", get_profile("llada")["repo_id"]), ("device", "cuda"),
                       ("dtype", "bfloat16"), ("local_files_only", False)):
        model.setdefault(key, value)
    if model_path is not None:
        model["model_path"] = str(model_path)
    if local_files_only is not None:
        model["local_files_only"] = local_files_only
    generation = config.setdefault("generation", {})
    for key, value in (("gen_length", 256), ("steps", 256), ("block_length", 128), ("temperature", 1.0)):
        generation.setdefault(key, value)
    # Position selection belongs to the entrypoint, not the legacy dispatcher.
    generation.pop("remasking", None)
    decoding_seed = seed if seed is not None else config.get("decoding", {}).get("seed", 1)
    config["decoding"] = {"method": method, "seed": decoding_seed}
    if temperature is not None:
        generation["temperature"] = temperature
    validate_generation(**generation, seed=decoding_seed)
    compatible = {**config, "generation": {**generation, "remasking": "low_confidence"},
                  "decoding": {"method": "baseline", "seed": decoding_seed}}
    validate_config(compatible)
    return config


def example_seed(base_seed, row_id):
    """Reserve five distinct seeds per source row; candidate zero is paired."""
    return (base_seed + 5 * row_id) % 2**63


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _load_model(config):
    from dlm_project.models.llada import LLaDA
    return LLaDA(**config["model"])


def _verify_model(wrapper):
    pinned = get_profile("llada")
    if any(wrapper.identity.get(key) != pinned[key] for key in ("repo_id", "revision")):
        raise ValueError("The benchmark requires the pinned original LLaDA checkpoint; "
                         "local snapshots must include .model_source.json.")


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _metrics(config, state, records):
    count = len(records)
    times = [row["latency_seconds"] for row in records]
    total_seconds = sum(times)
    calls = sum(row["total_model_calls"] for row in records)
    memory = [row["peak_gpu_memory_bytes"] for row in records if row["peak_gpu_memory_bytes"] is not None]
    return {"status": state["status"], "method": config["decoding"]["method"],
            "profile": config["evaluation"]["profile"],
            "requested_examples": config["data"]["num_samples"], "processed": count,
            "correct": sum(row["exact_match"] for row in records),
            "accuracy": sum(row["exact_match"] for row in records) / count if count else None,
            "parse_failures": sum(not row["parse_success"] for row in records),
            "parse_rate": sum(row["parse_success"] for row in records) / count if count else None,
            "generation_latency_seconds": total_seconds,
            "mean_latency_seconds": mean(times) if times else None,
            "median_latency_seconds": median(times) if times else None,
            "questions_per_second": count / total_seconds if total_seconds > 0 else None,
            "total_model_calls": calls, "mean_model_calls": calls / count if count else None,
            "peak_gpu_memory_bytes": max(memory) if memory else None,
            "model_load_count": state["model_load_count"],
            "model_loading_seconds": state.get("model_loading_seconds"),
            "model_revision": config.get("provenance", {}).get("model", {}).get("revision"),
            "dataset_revision": config["data"]["revision"], "error": state.get("error")}


def run_evaluation(method, source, *, manifest_path=None, output_dir=None, model_factory=None):
    """Load one model, preserve every completed prediction, and report actual cost."""
    config = resolve_settings(method, source)
    root = Path(config.get("experiment_root", "results")).expanduser()
    root = (root if root.is_absolute() else PROJECT_ROOT / root).resolve()
    manifest_path = Path(manifest_path).expanduser() if manifest_path else root / "manifests/gsm8k_test.json"
    manifest = get_manifest(manifest_path, cache_dir=config["data"].get("cache_dir"),
                            revision=config["data"].get("revision"))
    examples = select_examples(manifest, config["evaluation"]["profile"], config["data"]["num_samples"])
    config["data"]["revision"] = manifest["dataset"]["revision"]
    generation = config["generation"]
    name = (f"gen{generation['gen_length']}-steps{generation['steps']}-blk{generation['block_length']}-"
            f"temp{generation['temperature']:g}-seed{config['decoding']['seed']}")
    parent = Path(output_dir).expanduser() if output_dir else root / "LLaDA/GSM8K"
    parent = parent / method / config["evaluation"]["profile"] / name
    directory, started = create_run_directory({"output_dir": str(parent),
                                               "experiment_timezone": config.get("experiment_timezone", "America/Los_Angeles")})
    config["run"] = {"directory": str(directory), "started_at": started.isoformat(),
                     "started_at_utc": started.astimezone(timezone.utc).isoformat(),
                     "example_ids": [row["example_id"] for row in examples],
                     "manifest_hash": digest(manifest), "manifest_source": str(manifest_path.resolve())}
    config["provenance"] = {"git_commit": _git_commit(), "dataset": manifest["dataset"]}
    save_json(directory / "config.json", config)
    save_json(directory / "manifest.json", manifest)
    (directory / "traces").mkdir()
    (directory / "predictions.jsonl").touch()
    state = {"status": "running", "model_load_count": 0}
    records = []

    def save_progress():
        metrics = _metrics(config, state, records)
        save_json(directory / "metrics.json", metrics)
        return metrics

    save_progress()
    print(f"Evaluation artifacts: {directory}", flush=True)
    try:
        load_started = perf_counter()
        wrapper = (model_factory or _load_model)(config)
        _sync(wrapper.device)
        state.update(model_load_count=1, model_loading_seconds=perf_counter() - load_started)
        _verify_model(wrapper)
        config["provenance"]["model"] = wrapper.identity
        save_json(directory / "config.json", config)
        save_progress()
        with (directory / "predictions.jsonl").open("a", encoding="utf-8") as predictions:
            for example in examples:
                prompt = build_prompt(example["question"], config["prompt"]["instructions"])
                seed = example_seed(config["decoding"]["seed"], example["row_id"])
                _sync(wrapper.device)
                if torch.device(wrapper.device).type == "cuda":
                    torch.cuda.reset_peak_memory_stats(wrapper.device)
                prediction_started = perf_counter()
                prompt_ids = wrapper.encode_prompt(prompt)
                kwargs = {**generation, "seed": seed}
                if method == "best_of_5":
                    kwargs.update(tokenizer=wrapper.tokenizer, answer_extractor=extract_answer)
                result = DECODERS[method](wrapper.model, prompt_ids, wrapper.mask_id, **kwargs)
                if method == "best_of_5":
                    text = result.details["candidates"][result.details["selected_candidate_index"]]["raw_output"]
                else:
                    text = wrapper.tokenizer.batch_decode(result.token_ids[:, prompt_ids.shape[1]:],
                                                          skip_special_tokens=True)[0]
                _sync(wrapper.device)
                latency = perf_counter() - prediction_started
                peak_memory = (torch.cuda.max_memory_allocated(wrapper.device)
                               if torch.device(wrapper.device).type == "cuda" else None)
                # Reference solutions enter only post-generation scoring.
                score = score_answer(text, example["reference_solution"])
                trace_file = f"traces/test_{example['row_id']:04d}.jsonl"
                with (directory / trace_file).open("w", encoding="utf-8") as trace:
                    for event in result.trace:
                        trace.write(json.dumps(event, allow_nan=False) + "\n")
                row = {"example_id": example["example_id"], "row_id": example["row_id"],
                       "method": method, "seed": seed, "prompt": prompt, "prompt_hash": digest(prompt),
                       "content_hash": example["content_hash"], "reference_solution": example["reference_solution"],
                       "raw_output": text, **score, "model_revision": wrapper.identity["revision"],
                       "dataset_revision": manifest["dataset"]["revision"],
                       "generation_nfes": result.generation_nfes, "total_model_calls": result.total_nfes,
                       "latency_seconds": latency, "peak_gpu_memory_bytes": peak_memory,
                       "model_device": str(wrapper.device), "prompt_token_count": prompt_ids.shape[1],
                       "generation_length": generation["gen_length"], "trace_file": trace_file,
                       **result.details}
                predictions.write(json.dumps(row, allow_nan=False) + "\n")
                predictions.flush()
                records.append(row)
                metrics = save_progress()
                print(f"{row['example_id']} | {method} | accuracy={metrics['accuracy']:.3f} | "
                      f"forwards={result.total_nfes} | {latency:.2f}s", flush=True)
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}")
        save_progress()
        raise
    metrics = save_progress()
    print(json.dumps(metrics, indent=2), flush=True)
    return directory, metrics


def main(method):
    parser = argparse.ArgumentParser(description=f"LLaDA GSM8K accuracy with {method} decoding.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/gsm8k/decoding.yaml")
    parser.add_argument("--profile", choices=tuple(PROFILE_SIZES))
    parser.add_argument("--num-samples", type=int, help="Override the profile's question count")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--local-files-only", action="store_true", default=None)
    parser.add_argument("--manifest", type=Path, help="Shared pinned GSM8K test manifest")
    parser.add_argument("--output-dir", type=Path, help="Artifact parent; adds method/profile/settings/timestamp")
    args = parser.parse_args()
    try:
        config = resolve_settings(method, load_config(args.config), profile=args.profile,
                                  num_samples=args.num_samples, seed=args.seed, temperature=args.temperature,
                                  model_path=args.model_path, local_files_only=args.local_files_only)
    except (ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    run_evaluation(method, config, manifest_path=args.manifest, output_dir=args.output_dir)
