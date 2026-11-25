"""E0: LLaDA Confidence primary baseline and optional LLaDA Entropy baseline."""

import argparse
from collections import defaultdict
import copy
import gc
import json
from math import comb
import os
from pathlib import Path
import signal
import subprocess
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from dlm_project.decoding import confidence, entropy, mcts_decoding as medal
from dlm_project.decoding.ab_batch import pack_states, sync
from dlm_project.decoding.common import commit_from_logits, initialize_state, validate_generation
from dlm_project.decoding.decoder_result import CountingModel
from dlm_project.decoding.medal_demo import DecodingResult
from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.config import load_config
from dlm_project.experiments.manifests import digest
from dlm_project.experiments.paths import create_run_directory
from dlm_project.experiments.resources import (
    profiled_stage,
    record_batch,
    record_completed,
    record_prepared,
    resource_phase,
    synchronize,
)
from dlm_project.experiments.resume import begin_question, resume_signature, validate_resume
from dlm_project.experiments.sweep_report import wilson_interval, write_csv
from dlm_project.models.profiles import get_profile
from dlm_project.paths import PROJECT_ROOT
from dlm_project.tasks.gsm8k.experiment_protocol import (
    E2_EVALUATION_COUNT,
    E2_STAGE_COUNTS,
    EXTENDED_TEST_COUNT,
    STAGE_COUNTS,
    build_prompt as build_e0_prompt,
    example_seed,
    get_e2_manifest,
    get_extended_manifest,
    get_manifest,
    load_shared_protocol,
    resolve_common_reference,
    select_examples,
)
from dlm_project.tasks.gsm8k.e0_tokens import decode_output, resolve_termination_tokens
from dlm_project.tasks.gsm8k.evaluation import score_answer


PROTOCOL_VERSION = "e0_fixed_budget_v6"
DECODERS = {"confidence": confidence.decode, "entropy": entropy.decode}
SELECTION_RULES = {"confidence": "token_probability", "entropy": "lowest_entropy"}
MIGRATION = ("E0 focuses on search-free baselines (LLaDA Confidence, optional Entropy). "
             "MEDAL search has migrated to Experiment 1 (scripts/run_e1_e3.py).")


def resolve_settings(source, **overrides):
    common, config = resolve_common_reference(source)
    evaluation = config.setdefault("evaluation", {})
    for key, default in (("stage", "dev"), ("run_order", "condition")):
        evaluation[key] = overrides.get(key) if overrides.get(key) is not None else evaluation.get(key, default)
    if evaluation["stage"] not in STAGE_COUNTS:
        raise ValueError("E0 stage must be dev/evaluation/test.")
    if evaluation["run_order"] not in ("example", "condition"):
        raise ValueError("E0 run_order must be example or condition.")
    if ("search_budgets" in evaluation or overrides.get("search_budgets") is not None
            or overrides.get("init_tokens") is not None
            or overrides.get("search_mode", "none") not in ("none", "budget")
            or evaluation.get("search_mode", "none") not in ("none", "budget")):
        raise ValueError(MIGRATION)
    evaluation["search_mode"] = "none"
    if set(evaluation) - {"stage", "run_order", "search_mode", "methods"}:
        raise ValueError("Unknown E0 evaluation setting.")
    data = config["data"]
    for key in ("cache_dir", "dataset_revision"):
        if overrides.get(key) is not None:
            data["revision" if key == "dataset_revision" else key] = overrides[key]
    count = STAGE_COUNTS[evaluation["stage"]]
    requested_samples = overrides.get("num_samples") if overrides.get("num_samples") is not None else data.get("num_samples")
    if requested_samples is not None:
        if type(requested_samples) is not int or not 1 <= requested_samples <= count:
            raise ValueError(f"num_samples must be an integer between 1 and {count} for stage {evaluation['stage']}.")
        data["num_samples"] = requested_samples
    else:
        data["num_samples"] = count
    if (data["revision"] != common["data"]["revision"] or type(data["sample_seed"]) is not int
            or data["sample_seed"] != common["data"]["sample_seed"]
            or set(data) - {"revision", "sample_seed", "num_samples", "cache_dir"}):
        raise ValueError("Dataset settings differ from the frozen shared protocol.")
    if data.get("cache_dir") is not None:
        data["cache_dir"] = str(data["cache_dir"])
    if config["prompt"] != common["prompt"]:
        raise ValueError("Prompt differs from the frozen shared protocol.")
    if config["generation"] != common["generation"]:
        raise ValueError("Generation settings differ from the frozen shared protocol.")
    decoding = config["decoding"]
    for key in ("seed", "completion_method", "completion_temperature"):
        if decoding[key] != common["decoding"][key]:
            raise ValueError(f"decoding.{key} differs from the frozen shared protocol.")
    if set(decoding) - {"seed", "completion_method", "completion_temperature", "mcts"}:
        raise ValueError("Unknown E0 decoding setting.")
    validate_generation(**config["generation"], temperature=decoding["completion_temperature"], seed=decoding["seed"])
    model = config["model"]
    for key in ("model_path", "device", "dtype", "local_files_only"):
        if overrides.get(key) is not None:
            model[key] = overrides[key]
    model["model_path"] = str(model["model_path"])
    if (model["family"] != "llada" or model["dtype"] != common["model"]["dtype"]
            or set(model) - {"family", "model_path", "device", "dtype", "local_files_only"}):
        raise ValueError("E0 requires the pinned LLaDA checkpoint with bfloat16 weights.")

    raw_methods = overrides.get("methods") or evaluation.get("methods", ["confidence"])
    if isinstance(raw_methods, str):
        methods = [raw_methods]
    else:
        methods = list(raw_methods)
    if overrides.get("include_entropy") and "entropy" not in methods:
        methods.append("entropy")

    conditions = []
    for m in methods:
        if m not in DECODERS:
            raise ValueError(f"Unknown E0 baseline method: {m}. Must be 'confidence' or 'entropy'.")
        conditions.append({"id": f"{m}_t0", "method": m, "temperature": 0.0,
                           "search_budget": 0, "requested_depth": None, "completion_method": "llada"})
    config.update(protocol_version=PROTOCOL_VERSION, conditions=conditions,
                  common_protocol=common, common_protocol_hash=digest(common),
                  prompt_template_hash=digest(common["prompt"]["template"]))
    return config



def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _load_model(config):
    from dlm_project.models.llada import LLaDA
    return LLaDA(**config["model"])


def _compact_event(value):
    omitted = {"entropies", "entropies_before", "entropies_after", "state", "child_state",
               "selected_state", "candidate_evaluations"}
    if isinstance(value, dict):
        return {key: _compact_event(item) for key, item in value.items() if key not in omitted}
    if isinstance(value, list):
        return [_compact_event(item) for item in value]
    return value


def _code_identity():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                                             text=True, stderr=subprocess.DEVNULL))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    sources = {str(path.relative_to(PROJECT_ROOT)): digest(path.read_text())
               for path in sorted((PROJECT_ROOT / "src/dlm_project").rglob("*.py"))}
    return {"git_commit": commit, "git_dirty": dirty, "source_hash": digest(sources)}


def run_experiment(source, *, manifest_path=None, output_dir=None, run_manifest=None, model_factory=None):
    """One model load, one tokenization per question, and independent condition trees."""
    from dlm_project.tasks.gsm8k.e0_report import generate_report, summarize

    config = resolve_settings(source)
    root = Path(config.get("experiment_root", "results")).expanduser()
    root = root if root.is_absolute() else PROJECT_ROOT / root
    manifest = get_manifest(manifest_path, common=config["common_protocol"])
    examples = select_examples(manifest, config["evaluation"]["stage"])[:config["data"]["num_samples"]]
    config["data"]["revision"] = manifest["dataset"]["revision"]
    parent = Path(output_dir) if output_dir else root / "LLaDA/GSM8K/experiment_0"
    parent = parent / config["evaluation"]["stage"] / "fixed-matrix-v6"
    directory, _ = create_run_directory({"output_dir": str(parent),
        "experiment_timezone": config.get("experiment_timezone", "America/Los_Angeles")})
    config["run"] = {"example_ids": [row["example_id"] for row in examples],
                     "manifest_hash": digest(manifest)}
    config["provenance"] = _code_identity()
    records, setup = [], {"status": "running", "model_load_count": 0, "warmup_nfes": 0,
                          "tokenization_seconds": 0.0, "scoring_seconds": 0.0, "offline_analysis_seconds": 0.0,
                          "artifact_writing_seconds": 0.0, "active_total_nfes": 0}
    (directory / "traces").mkdir()
    (directory / "prompts").mkdir()
    save_json(directory / "config.json", config)
    save_json(directory / "manifest.json", manifest)
    save_json(directory / "selection.json", examples)
    (directory / "predictions.jsonl").touch()
    receipt = Path(run_manifest) if run_manifest else None
    if receipt:
        receipt.parent.mkdir(parents=True, exist_ok=True)

    def progress():
        started = perf_counter()
        save_json(directory / "setup.json", setup)
        summary = summarize(config, setup, records)
        save_json(directory / "summary.json", summary)
        if receipt:
            save_json(receipt, {"run_dir": str(directory), "status": setup["status"],
                               "protocol_version": PROTOCOL_VERSION,
                               "common_protocol_hash": config["common_protocol_hash"],
                               "stage": config["evaluation"]["stage"], "search_mode": "none",
                               "run_order": config["evaluation"]["run_order"],
                               "condition_ids": [condition["id"] for condition in config["conditions"]],
                               "model_family": "llada"})
        setup["artifact_writing_seconds"] += perf_counter() - started
        return summary

    progress()
    print(f"E0 artifacts: {directory}", flush=True)
    try:
        started = perf_counter()
        wrapper = (model_factory or _load_model)(config)
        _sync(wrapper.device)
        setup.update(model_load_count=1, model_loading_seconds=perf_counter() - started)
        if any(wrapper.identity.get(key) != get_profile()[key] for key in ("repo_id", "revision")):
            raise ValueError("E0 requires the repository-pinned LLaDA checkpoint.")
        setup["gpu_name"] = torch.cuda.get_device_name(wrapper.device) if torch.device(wrapper.device).type == "cuda" else None
        config["provenance"]["model"] = wrapper.identity
        config["termination"] = resolve_termination_tokens(wrapper.tokenizer, wrapper.model, wrapper.mask_id)
        save_json(directory / "config.json", config)
        started = perf_counter()
        warm_ids = wrapper.encode_prompt(build_e0_prompt(examples[0]["question"], config["prompt"]))
        state = torch.full((1, warm_ids.shape[1] + config["generation"]["gen_length"]),
                           wrapper.mask_id, dtype=torch.long, device=wrapper.device)
        state[:, :warm_ids.shape[1]] = warm_ids
        with torch.inference_mode():
            setup["warmup_nfes"] = 1
            prediction = wrapper.model(state)
        del prediction, state, warm_ids
        _sync(wrapper.device)
        setup.update(warmup_nfes=1, warmup_seconds=perf_counter() - started)
        progress()
        conditions = config["conditions"]
        if config["evaluation"]["run_order"] == "condition":
            schedule = ((example, condition) for condition in conditions for example in examples)
        else:
            schedule = ((example, condition) for example in examples for condition in conditions)
        prompt_cache, completed_counts = {}, {condition["id"]: 0 for condition in conditions}
        print("E0 execution order:", config["evaluation"]["run_order"],
              "→".join(condition["id"] for condition in conditions), flush=True)
        with (directory / "predictions.jsonl").open("a") as output:
            for example, condition in schedule:
                if example["example_id"] not in prompt_cache:
                    started = perf_counter()
                    prompt = build_e0_prompt(example["question"], config["prompt"])
                    prompt_ids = wrapper.encode_prompt(prompt)
                    _sync(wrapper.device)
                    setup["tokenization_seconds"] += perf_counter() - started
                    seed = example_seed(example["row_id"], config["decoding"]["seed"])
                    prompt_cache[example["example_id"]] = prompt, prompt_ids, seed
                    write_started = perf_counter()
                    save_json(directory / "prompts" / f"{example['example_id'].replace(':', '_')}.json",
                              {"text": prompt, "prompt_ids": prompt_ids[0].cpu().tolist(),
                               "mask_id": wrapper.mask_id, "gen_length": config["generation"]["gen_length"],
                               "prompt_ids_hash": digest(prompt_ids.cpu().tolist())})
                    setup["artifact_writing_seconds"] += perf_counter() - write_started
                prompt, prompt_ids, seed = prompt_cache[example["example_id"]]
                if completed_counts[condition["id"]] == 0:
                    print(f"Starting {condition['id']} on {len(examples)} questions", flush=True)
                counter = CountingModel(wrapper.model)
                events = []
                trace_file = f"traces/{example['example_id'].replace(':', '_')}_{condition['id']}.jsonl"
                setup.update(active_example_id=example["example_id"], active_condition=condition["id"],
                             active_total_nfes=0, active_trace_file=trace_file)
                progress()
                kwargs = {**config["generation"], "temperature": condition["temperature"], "seed": seed}
                _sync(wrapper.device)
                started = perf_counter()
                result, details, status = None, {}, "completed"
                try:
                    result = DECODERS[condition["method"]](counter, prompt_ids.clone(), wrapper.mask_id, **kwargs)
                    details = getattr(result, "details", {})
                    if hasattr(result, "trace"):
                        events.extend(_compact_event(event) for event in result.trace)
                finally:
                    _sync(wrapper.device)
                    elapsed = perf_counter() - started
                    setup["active_total_nfes"] = counter.calls
                    setup["active_decoding_seconds"] = elapsed
                    write_started = perf_counter()
                    with (directory / trace_file).open("w") as trace:
                        for event in events:
                            trace.write(json.dumps(event, allow_nan=False) + "\n")
                    setup["artifact_writing_seconds"] += perf_counter() - write_started
                search_nfes = details.get("search_model_calls", 0)
                completion_nfes = counter.calls - search_nfes
                if (search_nfes < 0 or completion_nfes < 0
                        or (condition["search_budget"] is not None and search_nfes > condition["search_budget"])
                        or (result is not None and result.total_nfes != counter.calls)):
                    raise RuntimeError("E0 observed model forwards do not reconcile.")
                depth = details.get("selected_depth", 0)
                if result is not None:
                    if completion_nfes != config["generation"]["gen_length"] - depth:
                        raise RuntimeError("Completion must fill one remaining mask per forward.")
                    if (not torch.equal(result.token_ids[:, :prompt_ids.shape[1]].to(prompt_ids.device), prompt_ids)
                            or (result.token_ids[:, prompt_ids.shape[1]:] == wrapper.mask_id).any()):
                        raise RuntimeError("Decoder changed the prompt or left output masks.")
                scoring_started = perf_counter()
                decoded = decode_output(wrapper.tokenizer,
                    [] if result is None else result.token_ids[0, prompt_ids.shape[1]:].tolist(),
                    config["termination"]["token_ids"])
                score = score_answer(decoded["scoring_text"], example["reference_solution"])
                setup["scoring_seconds"] += perf_counter() - scoring_started
                audit_file, audit_metrics = None, None
                if any(event["stage"] == "node_evaluated" for event in events):
                    from dlm_project.decoding.search_audit import from_events, metrics as allocation_metrics
                    analysis_started = perf_counter()
                    audit = from_events(events, prompt_ids=prompt_ids[0].cpu().tolist(),
                        mask_id=wrapper.mask_id, gen_length=config["generation"]["gen_length"],
                        requested_depth=condition["requested_depth"], requested_budget=condition["search_budget"],
                        search_nfes=search_nfes)
                    audit_metrics = allocation_metrics(audit)
                    setup["offline_analysis_seconds"] += perf_counter() - analysis_started
                    write_started = perf_counter()
                    audit_file = trace_file.replace(".jsonl", "_audit.json")
                    save_json(directory / audit_file, audit)
                    setup["artifact_writing_seconds"] += perf_counter() - write_started
                row = {**example, "protocol_version": PROTOCOL_VERSION, "condition": condition["id"],
                       "method": condition["method"], "temperature": condition["temperature"],
                       "requested_depth": condition["requested_depth"],
                       "common_protocol_hash": config["common_protocol_hash"],
                       "prompt_template_hash": config["prompt_template_hash"],
                       "completion_method": condition.get("completion_method"),
                       "status": status, "seed": seed, "prompt": prompt, "prompt_hash": digest(prompt),
                       "prompt_ids_hash": digest(prompt_ids.cpu().tolist()),
                       "model_revision": wrapper.identity["revision"], "dataset_revision": config["data"]["revision"],
                       "generation_length": config["generation"]["gen_length"], "requested_search_budget": condition["search_budget"],
                       "search_nfes": search_nfes, "completion_nfes": completion_nfes, "total_nfes": counter.calls,
                       "budget_utilization": search_nfes / condition["search_budget"] if condition["search_budget"] else None,
                       "decoding_seconds": elapsed, "search_seconds": details.get("search_latency_seconds", elapsed if status == "search_incomplete" else 0.0),
                       "completion_seconds": details.get("completion_latency_seconds", elapsed if not search_nfes else 0.0),
                       "selected_depth": depth, "max_depth": details.get("max_depth", 0),
                       "average_search_depth": details.get("average_search_depth", 0.0),
                       "candidate_count": details.get("candidate_count", 0),
                       "target_depth_candidate_count": details.get("target_depth_candidate_count", 0),
                       "expanded_parents": details.get("expanded_parents", 0),
                       "evaluated_children": details.get("evaluated_children", 0),
                       "realized_branching_factor": details.get("realized_branching_factor", 0.0),
                       "selected_score": details.get("selected_score"),
                       "shortlist_size": details.get("shortlist_size", 0),
                       "shortlist_underfilled": details.get("shortlist_underfilled", False),
                       "selection_source": details.get("selection_source", "no_search"),
                       "fallback_decode_used": status == "completed" and details.get("fallback_decode_used", False),
                       "stop_reason": details.get("stop_reason", "all_masks_filled"),
                       "trace_file": trace_file, "search_audit_file": audit_file,
                       "search_audit_metrics": audit_metrics, "details": details, **decoded, **score}
                write_started = perf_counter()
                output.write(json.dumps(row, allow_nan=False) + "\n")
                output.flush()
                records.append(row)
                setup["artifact_writing_seconds"] += perf_counter() - write_started
                setup.update(active_example_id=None, active_condition=None, active_total_nfes=0,
                             active_decoding_seconds=0.0)
                current_summary = progress()
                print(f"{row['example_id']} | {row['condition']} | {status} | correct={row['exact_match']} "
                      f"| search={search_nfes}, completion={completion_nfes}, total={counter.calls} NFEs "
                      f"| depth={depth} | {elapsed:.2f}s", flush=True)
                completed_counts[condition["id"]] += 1
                if completed_counts[condition["id"]] == len(examples):
                    metrics = next(item for item in current_summary["observed_conditions"]
                                   if item["id"] == condition["id"])
                    print(f"Finished {condition['id']} | n={metrics['n']} | "
                          f"accuracy={metrics['accuracy_percent']:.2f}% | "
                          f"parse={100 * metrics['parse_rate']:.2f}% | "
                          f"mean total NFEs={metrics['mean_total_nfes']:.2f} | "
                          f"mean decode={metrics['mean_decoding_seconds']:.2f}s", flush=True)
        setup["status"] = "completed"
    except BaseException as error:
        setup.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}")
        progress()
        try:
            generate_report(directory)
        except Exception as report_error:
            print(f"Partial report unavailable: {report_error}", flush=True)
        raise
    progress()
    started = perf_counter()
    summary = generate_report(directory)
    setup["report_seconds"] = perf_counter() - started
    progress()
    print(f"E0 report: {directory / 'report/report.md'}", flush=True)
    return directory, summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 0: Baseline Test — Search-Free LLaDA Baselines")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/gsm8k/experiment_0.yaml")
    parser.add_argument("--stage", choices=tuple(STAGE_COUNTS), default="dev")
    parser.add_argument("--num-samples", type=int, help="Optional prefix count for testing (up to stage maximum: 5 dev, 20 evaluation, 200 test).")
    parser.add_argument("--include-entropy", action="store_true", help="Include optional LLaDA Entropy baseline.")
    parser.add_argument("--methods", nargs="+", choices=("confidence", "entropy"), default=["confidence"],
                        help="Baselines to run (default: confidence).")
    parser.add_argument("--init-tokens", type=int, help="Legacy option: rejected by the fixed protocol.")
    parser.add_argument("--search-mode", choices=("budget", "candidates", "none"))
    parser.add_argument("--search-budgets", type=int, nargs="+", help="Legacy option: rejected by the fixed protocol.")
    parser.add_argument("--run-order", choices=("example", "condition"), default="condition",
                        help="Finish all conditions per example, or all examples per condition.")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true", default=None)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--manifest", type=Path, help="Exact copy of the checked-in shared sample manifest.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--report-only", type=Path, metavar="RUN_DIR")
    args = parser.parse_args()
    if args.report_only:
        from dlm_project.tasks.gsm8k.e0_report import generate_report
        generate_report(args.report_only)
        print(args.report_only / "report/report.md")
        return
    overrides = {key: value for key, value in vars(args).items() if value is not None}
    config = resolve_settings(load_config(args.config), **overrides)
    # Notebook Stop terminates the owned subprocess. Convert SIGTERM to the
    # same flushed partial-run path as Ctrl-C, including the current counter.
    def interrupted(signum, frame):
        raise KeyboardInterrupt("E0 subprocess terminated")
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        run_experiment(config, manifest_path=args.manifest, output_dir=args.output_dir,
                       run_manifest=args.run_manifest)
    finally:
        signal.signal(signal.SIGTERM, previous)


# ---------------------------------------------------------------------------
# Unified Stage-Runner API for E0 (Search-Free LLaDA Confidence & Entropy)
# Aligned with Experiment 1 & Experiment 2 test/evaluation protocols
# ---------------------------------------------------------------------------


def context(
    *,
    stage: str = "test",
    methods: Sequence[str] = ("confidence",),
    num_samples: Optional[int] = None,
    sample_slice: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Create and validate execution context for E0 across dev, evaluation, or test splits."""
    common = load_shared_protocol()
    norm_stage = "evaluation" if stage in ("eval", "evaluation") else stage
    stage_limits = {
        "dev": STAGE_COUNTS["dev"],
        "evaluation": E2_EVALUATION_COUNT,
        "test": EXTENDED_TEST_COUNT,
    }
    if norm_stage not in stage_limits:
        raise ValueError("Stage must be dev, eval, evaluation, or test.")
    limit = stage_limits[norm_stage]

    if num_samples is not None and (type(num_samples) is not int or not 1 <= num_samples <= limit):
        raise ValueError(f"num_samples for stage {stage!r} must be an integer from 1 to {limit}.")
    if sample_slice is not None:
        if (
            not isinstance(sample_slice, (tuple, list))
            or len(sample_slice) != 2
            or any(type(value) is not int for value in sample_slice)
            or not 0 <= sample_slice[0] < sample_slice[1] <= limit
        ):
            raise ValueError(f"sample_slice for stage {stage!r} must satisfy 0 <= start < end <= {limit}.")

    if norm_stage == "test":
        requested_end = sample_slice[1] if sample_slice is not None else num_samples
        if requested_end is not None and requested_end > STAGE_COUNTS["test"]:
            manifest = get_extended_manifest(common=common, test_count=EXTENDED_TEST_COUNT)
            expected_count = EXTENDED_TEST_COUNT
        else:
            manifest = get_manifest(common=common)
            expected_count = STAGE_COUNTS["test"]
    elif norm_stage == "evaluation":
        requested_end = sample_slice[1] if sample_slice is not None else num_samples
        if sample_slice is not None or (requested_end is not None and requested_end > STAGE_COUNTS["evaluation"]):
            manifest = get_e2_manifest(common=common)
            expected_count = E2_STAGE_COUNTS["evaluation"]
        else:
            manifest = get_manifest(common=common)
            expected_count = STAGE_COUNTS["evaluation"]
    else:
        manifest = get_manifest(common=common)
        expected_count = STAGE_COUNTS["dev"]

    examples = select_examples(manifest, norm_stage, expected_count=expected_count)
    if sample_slice is not None:
        start, end = sample_slice
        examples = examples[start:end]
    if num_samples is not None:
        if num_samples > len(examples):
            raise ValueError(f"num_samples={num_samples} exceeds the selected split of {len(examples)} questions.")
        examples = examples[:num_samples]
    if not examples:
        raise ValueError(f"No examples found for stage {stage} (slice={sample_slice}, num_samples={num_samples}).")

    if isinstance(methods, str):
        methods = (methods,)
    methods_list = list(methods)
    if not methods_list:
        raise ValueError("At least one E0 baseline method ('confidence' or 'entropy') must be specified.")
    for m in methods_list:
        if m not in SELECTION_RULES:
            raise ValueError(f"Unknown E0 baseline method {m!r}. Must be 'confidence' or 'entropy'.")

    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": stage,
        "norm_stage": norm_stage,
        "methods": methods_list,
        "budgets": [0],
        "max_depth": 0,
        "num_candidates": 1,
        "reward_mode": "none",
        "common": common,
        "manifest": manifest["dataset"],
        "manifest_hash": digest(manifest),
        "prompt_template_hash": digest(common["prompt"]["template"]),
        "examples": examples,
        "setup": {},
    }


def load_model(ctx: Optional[Dict[str, Any]] = None, *, model_path=None, device: str = "cuda", local_files_only: bool = False):
    """Load LLaDA wrapper for E0 experiments."""
    from dlm_project.models.llada import LLaDA

    dtype = ctx["common"]["model"]["dtype"] if ctx and "common" in ctx else "bfloat16"
    started = perf_counter()
    wrapper = LLaDA(device=device, model_path=model_path, dtype=dtype, local_files_only=local_files_only)
    synchronize(wrapper.model)
    wrapper.model_load_seconds = perf_counter() - started
    if ctx is not None:
        ctx.setdefault("setup", {})["model_load_seconds"] = wrapper.model_load_seconds
    return wrapper


def prepare(wrapper, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Tokenize and format prompts for all examples in the stage."""
    termination = resolve_termination_tokens(wrapper.tokenizer, wrapper.model, wrapper.mask_id)
    pad = getattr(wrapper.tokenizer, "pad_token_id", None) or termination["eos_token_id"]
    prepared = []
    for example in ctx["examples"]:
        prompt = build_e0_prompt(example["question"], ctx["common"]["prompt"])
        ids = wrapper.encode_prompt(prompt)
        prepared.append({
            "example": example,
            "text": prompt,
            "ids": ids,
            "prompt_ids": ids[0].cpu().tolist(),
            "prompt_hash": digest(prompt),
            "prompt_ids_hash": digest(ids.cpu().tolist()),
            "seed": example_seed(example["row_id"]),
            "prompt_template_hash": ctx["prompt_template_hash"],
        })
    ctx.update(termination=termination, pad_token_id=pad, model=wrapper.identity)
    return prepared


def _jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + "\n")


@torch.inference_mode()
@resource_phase("completion")
def decode_states_batch_e0(wrapper, prepared_group: Sequence[Dict[str, Any]], ctx: Dict[str, Any], method: str):
    """Execute search-free LLaDA decoding at T=0 across a batch of questions."""
    if method not in SELECTION_RULES:
        raise ValueError(f"Unknown E0 baseline method {method!r}. Must be 'confidence' or 'entropy'.")
    selection_rule = SELECTION_RULES[method]
    gen_length = int(ctx["common"]["generation"]["gen_length"])
    device = medal.model_device(wrapper.model)
    pad_token_id = ctx["pad_token_id"]
    mask_id = wrapper.mask_id

    entries = []
    for item in prepared_group:
        prompt_len = item["ids"].shape[1]
        state = initialize_state(item["ids"], mask_id, gen_length, device)
        entries.append({
            "item": item,
            "prompt_len": prompt_len,
            "state": state,
            "calls": 0,
            "trace": [],
            "comp_time": None,
        })

    if entries:
        sync(wrapper.model)
        t0 = perf_counter()
        invocation_id = 0

        while True:
            active = [
                i for i, entry in enumerate(entries)
                if (entry["state"][0, entry["prompt_len"]:] == mask_id).any()
            ]
            if not active:
                break
            invocation_id += 1
            if invocation_id % 50 == 0 or invocation_id == 1:
                masks_left = [
                    int((entries[i]["state"][0, entries[i]["prompt_len"]:] == mask_id).sum().item())
                    for i in active
                ]
                print(
                    f"  [{method.upper()} Decode Step {invocation_id}] Active: {len(active)}/{len(entries)} | "
                    f"Masks remaining: {masks_left}",
                    flush=True,
                )
            current_states = [entries[i]["state"] for i in active]
            ids, attention, _ = pack_states(current_states, pad_token_id, device)
            for i in active:
                entries[i]["calls"] += 1
                entries[i]["trace"].append({
                    "stage": "forward",
                    "phase": "completion",
                    "model_calls": entries[i]["calls"],
                    "invocation_id": invocation_id,
                    "batch_size": len(active),
                })
            prediction = wrapper.model(ids) if len(active) == 1 else wrapper.model(ids, attention_mask=attention)
            for row, i in enumerate(active):
                entry = entries[i]
                state = entry["state"]
                prompt_len = entry["prompt_len"]
                positions = torch.nonzero(state[0, prompt_len:] == mask_id, as_tuple=False).flatten() + prompt_len
                actions, signals = commit_from_logits(
                    state,
                    prediction.logits[row, :state.shape[1]],
                    prompt_len,
                    mask_id,
                    positions,
                    1,
                    selection_rule=selection_rule,
                    temperature=0.0,
                )
                entry["trace"].append(_compact_event({
                    "stage": "commit",
                    "block": 0,
                    "step": entry["calls"] - 1,
                    "model_calls": entry["calls"],
                    "actions": actions,
                    **signals,
                    "remaining_masks": int((state[0, prompt_len:] == mask_id).sum().item()),
                }))
                if not (state[0, prompt_len:] == mask_id).any():
                    sync(wrapper.model)
                    entry["comp_time"] = perf_counter() - t0
            del prediction
            del ids, attention, current_states

        sync(wrapper.model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        total_comp_time = perf_counter() - t0
    else:
        total_comp_time = 0.0

    results = []
    for entry in entries:
        comp_time = entry["comp_time"] if entry["comp_time"] is not None else total_comp_time
        item = entry["item"]
        prompt_len = entry["prompt_len"]
        token_ids = entry["state"].detach().cpu()
        completion = DecodingResult(
            token_ids=token_ids,
            model_calls=entry["calls"],
            wall_seconds=comp_time,
            method=method,
            trace=entry["trace"],
        )
        decoded = decode_output(
            wrapper.tokenizer,
            completion.token_ids[0, prompt_len:].tolist(),
            ctx["termination"]["token_ids"],
        )
        score = score_answer(decoded["scoring_text"], item["example"]["reference_solution"])
        results.append((item, completion, decoded, score, comp_time))
    return results


@resource_phase("artifacts")
def export_question_e0_artifacts(
    wrapper,
    completion: DecodingResult,
    decoded: Dict[str, Any],
    score: Dict[str, Any],
    comp_time: float,
    prepared_item: Dict[str, Any],
    ctx: Dict[str, Any],
    method: str,
    base_dir: Path,
) -> Dict[str, Any]:
    """Export per-question E0 artifacts and completion marker."""
    ex_id = prepared_item["example"]["example_id"]
    q_dir = base_dir / method / ex_id
    q_dir.mkdir(parents=True, exist_ok=True)
    begin_question(q_dir)

    events = [_compact_event(e) for e in completion.trace]
    _jsonl(q_dir / "trace.jsonl", events)

    prompt_record = {k: v for k, v in prepared_item.items() if k != "ids"}
    prompt_record.update(mask_id=wrapper.mask_id, gen_length=ctx["common"]["generation"]["gen_length"])
    save_json(q_dir / "prompt.json", prompt_record)
    save_json(q_dir / "config.json", ctx)

    comp_nfes = int(getattr(completion, "model_calls", ctx["common"]["generation"]["gen_length"]))
    completion_record = {
        **decoded,
        "score": score,
        "token_ids": completion.token_ids.tolist(),
        "completion_nfes": comp_nfes,
        "completion_wall_seconds": comp_time,
        "was_cached": False,
    }
    save_json(q_dir / "completion.json", completion_record)

    metrics_row = {
        "method": method,
        "condition": f"{method}_t0",
        "selection_rule": SELECTION_RULES[method],
        "search_budget": 0,
        "example_id": ex_id,
        "row_id": prepared_item["example"]["row_id"],
        "question": prepared_item["example"]["question"],
        "reference_solution": prepared_item["example"]["reference_solution"],
        "scoring_text": decoded["scoring_text"],
        "raw_output": decoded["raw_output"],
        "first_terminator_position": decoded["first_terminator_position"],
        "predicted_answer": score["predicted_answer"],
        "reference_answer": score["reference_answer"],
        "exact_match": bool(score["exact_match"]),
        "parse_success": bool(score["parse_success"]),
        "search_nfes": 0,
        "completion_nfes": comp_nfes,
        "total_nfes": comp_nfes,
        "nfe_utilization": 1.0,
        "simulations_completed": 0,
        "simulation_cap": None,
        "simulation_cap_bound": False,
        "reward_mode": "none",
        "search_wall_seconds": 0.0,
        "completion_wall_seconds": comp_time,
        "total_wall_seconds": comp_time,
        "max_depth_reached": 0,
        "selected_depth": 0,
        "selection_depth_gap": 0,
        "selected_q": 0.0,
        "fallback_used": False,
        "stop_reason": "all_masks_filled",
        "was_cached": False,
        "artifact_directory": str(q_dir),
    }
    save_json(q_dir / "metrics.json", metrics_row)
    save_json(q_dir / "complete.json", {"method": method, "example_id": ex_id, "budgets": [0]})

    match_sym = "✓" if score["exact_match"] else "✗"
    print(
        f"  [{match_sym} E0 {method.upper():10s}] Q: {ex_id:<10} | "
        f"Pred: {score['predicted_answer']!r:<10} | Ref: {score['reference_answer']!r:<10} | "
        f"Match: {str(score['exact_match']):<5} | NFEs: {comp_nfes} | Time: {comp_time:.1f}s",
        flush=True,
    )
    return metrics_row


def run_question_e0(wrapper, prepared_item: Dict[str, Any], ctx: Dict[str, Any], method: str, base_dir: Path) -> Dict[str, Any]:
    """Run a single question under an E0 search-free baseline method."""
    results = decode_states_batch_e0(wrapper, [prepared_item], ctx, method)
    _, completion, decoded, score, comp_time = results[0]
    return export_question_e0_artifacts(
        wrapper, completion, decoded, score, comp_time, prepared_item, ctx, method, base_dir
    )


def run_questions_batch_e0(
    wrapper,
    prepared_group: Sequence[Dict[str, Any]],
    ctx: Dict[str, Any],
    method: str,
    base_dir: Path,
) -> List[Dict[str, Any]]:
    """Run a batch of questions cooperatively under an E0 search-free baseline method."""
    if len(prepared_group) == 1:
        return [run_question_e0(wrapper, prepared_group[0], ctx, method, base_dir)]

    decoded_batch = decode_states_batch_e0(wrapper, prepared_group, ctx, method)
    batch_rows = []
    for item, completion, decoded, score, comp_time in decoded_batch:
        row = export_question_e0_artifacts(
            wrapper, completion, decoded, score, comp_time, item, ctx, method, base_dir
        )
        batch_rows.append(row)

    del decoded_batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return batch_rows


def is_question_complete(stage_dir: Path, method: str, ex_id: str, budgets: Sequence[int] = (0,)) -> bool:
    """Verify whether all per-question E0 artifacts and completion marker are valid on disk."""
    q_dir = Path(stage_dir) / method / ex_id
    try:
        marker = json.loads((q_dir / "complete.json").read_text())
        if marker != {"method": method, "example_id": ex_id, "budgets": sorted(budgets)}:
            return False
        trace = [json.loads(line) for line in (q_dir / "trace.jsonl").read_text().splitlines() if line.strip()]
        if not trace or not all(isinstance(row, dict) for row in trace):
            return False
        metrics = json.loads((q_dir / "metrics.json").read_text())
        completion = json.loads((q_dir / "completion.json").read_text())
        if (
            metrics["method"] != method
            or metrics["example_id"] != ex_id
            or metrics["search_budget"] != 0
            or not isinstance(metrics["exact_match"], bool)
            or metrics["predicted_answer"] != completion["score"]["predicted_answer"]
            or metrics["exact_match"] != completion["score"]["exact_match"]
            or not completion["token_ids"]
        ):
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def load_completed_question_row(stage_dir: Path, method: str, ex_id: str) -> Dict[str, Any]:
    """Load cached E0 per-question metrics row from disk."""
    met_file = Path(stage_dir) / method / ex_id / "metrics.json"
    return json.loads(met_file.read_text())


def compute_e0_aggregate_rows(
    all_rows: Sequence[Dict[str, Any]],
    stage: str,
    methods: Sequence[str],
) -> List[Dict[str, Any]]:
    """Compute aggregate accuracy, Wilson 95% CIs, parse rate, NFEs, and latencies for E0."""
    agg_rows = []
    for method in methods:
        subset = [r for r in all_rows if r["method"] == method]
        n = len(subset)
        if not n:
            continue
        correct_cnt = sum(bool(r["exact_match"]) for r in subset)
        parsed_cnt = sum(bool(r["parse_success"]) for r in subset)
        ci_low, ci_high = wilson_interval(correct_cnt, n)

        agg_rows.append({
            "stage": stage,
            "method": method,
            "condition": f"{method}_t0",
            "search_budget": 0,
            "sample_count": n,
            "correct_count": correct_cnt,
            "accuracy": correct_cnt / n,
            "accuracy_percent": 100.0 * correct_cnt / n,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "parse_rate": parsed_cnt / n,
            "mean_search_nfe": sum(r["search_nfes"] for r in subset) / n,
            "mean_completion_nfe": sum(r["completion_nfes"] for r in subset) / n,
            "mean_total_nfe": sum(r["total_nfes"] for r in subset) / n,
            "mean_nfe_utilization": 1.0,
            "mean_simulations": 0.0,
            "sim_cap_bind_rate": 0.0,
            "mean_search_wall_seconds": 0.0,
            "mean_completion_wall_seconds": sum(r["completion_wall_seconds"] for r in subset) / n,
            "mean_total_wall_seconds": sum(r["total_wall_seconds"] for r in subset) / n,
            "mean_selected_depth": 0.0,
            "fallback_rate": 0.0,
        })
    return agg_rows


def mcnemar_exact_pvalue(b_wins: int, c_losses: int) -> float:
    """Two-sided exact binomial McNemar p-value on discordant pairs."""
    n = b_wins + c_losses
    if n == 0:
        return 1.0
    k = min(b_wins, c_losses)
    tail = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def compute_paired_comparison(
    q_rows: Sequence[Dict[str, Any]],
    baseline_method: str = "confidence",
    comparison_method: str = "entropy",
) -> List[Dict[str, Any]]:
    """Compute paired comparison and McNemar stats between two E0 baseline methods."""
    by_key = {(r["example_id"], r["method"]): r for r in q_rows}
    ex_ids = sorted({r["example_id"] for r in q_rows})
    pairs = [
        (by_key[(ex_id, baseline_method)], by_key[(ex_id, comparison_method)])
        for ex_id in ex_ids
        if (ex_id, baseline_method) in by_key and (ex_id, comparison_method) in by_key
    ]
    if not pairs:
        return []
    n = len(pairs)
    both_correct = sum(1 for base, comp in pairs if base["exact_match"] and comp["exact_match"])
    comp_only = sum(1 for base, comp in pairs if (not base["exact_match"]) and comp["exact_match"])
    base_only = sum(1 for base, comp in pairs if base["exact_match"] and (not comp["exact_match"]))
    both_wrong = sum(1 for base, comp in pairs if (not base["exact_match"]) and (not comp["exact_match"]))

    base_acc = (both_correct + base_only) / n
    comp_acc = (both_correct + comp_only) / n
    delta_acc = comp_acc - base_acc
    p_val = mcnemar_exact_pvalue(comp_only, base_only)

    return [{
        "search_budget": 0,
        "paired_questions": n,
        "baseline_method": baseline_method,
        "comparison_method": comparison_method,
        f"{baseline_method}_accuracy": base_acc,
        f"{comparison_method}_accuracy": comp_acc,
        "accuracy_delta": delta_acc,
        "accuracy_delta_percent": 100.0 * delta_acc,
        "both_correct": both_correct,
        f"{comparison_method}_only_correct": comp_only,
        f"{baseline_method}_only_correct": base_only,
        "both_wrong": both_wrong,
        "mcnemar_exact_p": p_val,
    }]


@resource_phase("artifacts")
def save_e0_progress(
    output_dir: Path,
    summaries_dir: Path,
    stage: str,
    all_rows: List[Dict[str, Any]],
    methods: Sequence[str],
    manifest_data: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Immediately write and flush per-question and aggregate CSVs and manifest, syncing OS buffers."""
    summaries_dir.mkdir(parents=True, exist_ok=True)
    agg_rows = []
    if all_rows:
        per_q_path = summaries_dir / f"e0_{stage}_per_question.csv"
        write_csv(per_q_path, all_rows)
        write_csv(summaries_dir / "e0_per_question.csv", all_rows)

        agg_rows = compute_e0_aggregate_rows(all_rows, stage, methods)
        agg_path = summaries_dir / f"e0_{stage}_aggregate.csv"
        write_csv(agg_path, agg_rows)
        write_csv(summaries_dir / "e0_aggregate.csv", agg_rows)

    if manifest_data is not None:
        save_json(output_dir / "run_manifest.json", manifest_data)
        save_json(summaries_dir / f"run_manifest_{stage}.json", manifest_data)

    try:
        os.sync()
    except (AttributeError, OSError):
        pass

    return agg_rows


@profiled_stage
def run_e0_stage(
    wrapper,
    ctx: Dict[str, Any],
    output_dir: Path,
    methods: Optional[Sequence[str]] = None,
    resume: bool = True,
    batch_size: int = 1,
    profile_resources: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Execute complete E0 stage for search-free baseline methods with incremental saving, resumption, and cooperative batching."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    resolved_methods = tuple(methods) if methods is not None else tuple(ctx["methods"])
    for m in resolved_methods:
        if m not in SELECTION_RULES:
            raise ValueError(f"Unknown E0 baseline method {m!r}. Must be 'confidence' or 'entropy'.")

    output_dir = Path(output_dir)
    stage_dir = output_dir / ctx["stage"]
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare(wrapper, ctx)
    record_prepared(prepared)
    signature = resume_signature(ctx, prepared, resolved_methods, (0,), 0, 1, batch_size)
    validate_resume(output_dir, ctx["stage"], signature, resume)

    manifest_data = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": ctx["stage"],
        "methods": list(resolved_methods),
        "budgets": [0],
        "max_depth": 0,
        "num_candidates": 1,
        "reward_mode": "none",
        "batch_size": batch_size,
        "sample_count": len(prepared),
        "examples": [p["example"]["example_id"] for p in prepared],
        "setup": ctx.get("setup", {}),
        "model": ctx.get("model", {}),
        "status": "in_progress",
        "resume_signature": signature,
    }
    save_json(output_dir / "run_manifest.json", manifest_data)
    save_json(summaries_dir / f"run_manifest_{ctx['stage']}.json", manifest_data)

    total_q = len(prepared)
    rows_by_ex_method: Dict[Tuple[str, str], Dict[str, Any]] = {}
    ordered_ex_ids = [p["example"]["example_id"] for p in prepared]
    all_rows: List[Dict[str, Any]] = []
    agg_rows: List[Dict[str, Any]] = []

    for start_idx in range(0, total_q, batch_size):
        group_items = prepared[start_idx : start_idx + batch_size]
        end_idx = start_idx + len(group_items)
        group_ex_ids = [item["example"]["example_id"] for item in group_items]

        for method in resolved_methods:
            if len(group_items) == 1:
                print(
                    f"\n[Progress {end_idx}/{total_q}] Question: {group_ex_ids[0]} | Baseline: {method.upper()}",
                    flush=True,
                )
            else:
                print(
                    f"\n[Progress {start_idx + 1}-{end_idx}/{total_q} | batch_size={len(group_items)}] "
                    f"Questions: {', '.join(group_ex_ids)} | Baseline: {method.upper()}",
                    flush=True,
                )

            cached_by_ex: Dict[str, List[Dict[str, Any]]] = {}
            pending_items = []
            for item in group_items:
                ex_id = item["example"]["example_id"]
                if resume and is_question_complete(stage_dir, method, ex_id):
                    cached_row = load_completed_question_row(stage_dir, method, ex_id)
                    match_sym = "✓" if cached_row["exact_match"] else "✗"
                    print(
                        f"  [{match_sym} Cached E0 {method.upper():10s}] Q: {ex_id:<10} | "
                        f"Pred: {cached_row['predicted_answer']!r:<10} | "
                        f"Ref: {cached_row['reference_answer']!r:<10} | "
                        f"Match: {str(cached_row['exact_match']):<5} | (loaded from disk)",
                        flush=True,
                    )
                    cached_by_ex[ex_id] = [cached_row]
                else:
                    pending_items.append(item)

            record_batch(pending_items, cached_by_ex, method)
            computed_by_ex: Dict[str, Dict[str, Any]] = {}
            if pending_items:
                if batch_size == 1 or len(pending_items) == 1:
                    item = pending_items[0]
                    row = run_question_e0(wrapper, item, ctx, method, stage_dir)
                    computed_by_ex[item["example"]["example_id"]] = row
                else:
                    rows = run_questions_batch_e0(wrapper, pending_items, ctx, method, stage_dir)
                    for r in rows:
                        computed_by_ex[r["example_id"]] = r

            record_completed(list(computed_by_ex.values()))
            for item in group_items:
                ex_id = item["example"]["example_id"]
                rows_by_ex_method[(ex_id, method)] = (
                    cached_by_ex[ex_id][0] if ex_id in cached_by_ex else computed_by_ex[ex_id]
                )

            all_rows = [
                rows_by_ex_method[(ex_id, m)]
                for ex_id in ordered_ex_ids
                for m in resolved_methods
                if (ex_id, m) in rows_by_ex_method
            ]
            manifest_data["completed_examples_count"] = end_idx
            agg_rows = save_e0_progress(
                output_dir,
                summaries_dir,
                ctx["stage"],
                all_rows,
                resolved_methods,
                manifest_data=manifest_data,
            )
            print(f"  💾 Saved progress to CSV ({len(all_rows)} rows recorded). Disk synced.", flush=True)

    manifest_data["status"] = "completed"
    manifest_data["completion_timestamp"] = perf_counter()
    agg_rows = save_e0_progress(
        output_dir,
        summaries_dir,
        ctx["stage"],
        all_rows,
        resolved_methods,
        manifest_data=manifest_data,
    )
    print(f"\n✅ All {total_q} questions completed successfully for E0 methods {list(resolved_methods)}.", flush=True)
    return all_rows, agg_rows


