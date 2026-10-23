"""E0: one confidence baseline and the frozen four-condition MEDAL matrix."""

import argparse
import json
from pathlib import Path
import signal
import subprocess
from time import perf_counter

import torch

from dlm_project.decoding import confidence, mcts_confidence
from dlm_project.decoding.common import validate_generation
from dlm_project.decoding.decoder_result import CountingModel
from dlm_project.decoding.mcts_decoding import MCTSConfig
from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.config import load_config
from dlm_project.experiments.manifests import digest
from dlm_project.experiments.paths import create_run_directory
from dlm_project.models.profiles import get_profile
from dlm_project.paths import PROJECT_ROOT
from dlm_project.tasks.gsm8k.experiment_protocol import (
    STAGE_COUNTS, get_manifest, select_examples, build_prompt as build_e0_prompt,
    example_seed, resolve_common_reference,
)
from dlm_project.tasks.gsm8k.e0_tokens import decode_output, resolve_termination_tokens
from dlm_project.tasks.gsm8k.evaluation import score_answer


PROTOCOL_VERSION = "e0_fixed_budget_v5"
DECODERS = {"confidence": confidence.decode, "medal": mcts_confidence.decode}
E0_MCTS = {"margin_gamma": 10.0, "exploration_c": 0.01, "epsilon": 1e-8,
           "stop_mode": "budget", "max_simulations": None}
MIGRATION = ("E0 v5 uses the fixed A/B/C/D depth-budget matrix. Remove single-depth, "
             "search_budgets and candidate-quota settings; see docs/experiments.md.")


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
            or overrides.get("search_mode", "budget") != "budget"
            or evaluation.get("search_mode", "budget") != "budget"):
        raise ValueError(MIGRATION)
    evaluation["search_mode"] = "budget"
    if set(evaluation) - {"stage", "run_order", "search_mode"}:
        raise ValueError("Unknown E0 evaluation setting.")
    data = config["data"]
    for key in ("cache_dir", "dataset_revision", "num_samples"):
        if overrides.get(key) is not None:
            data["revision" if key == "dataset_revision" else key] = overrides[key]
    count = STAGE_COUNTS[evaluation["stage"]]
    if data.get("num_samples") is not None and (type(data["num_samples"]) is not int or data["num_samples"] != count):
        raise ValueError(f"Frozen {evaluation['stage']} stage requires exactly {count} samples.")
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
    options = {**E0_MCTS, **decoding.get("mcts", {})}
    if "init_tokens" in options or "max_model_calls" in options:
        raise ValueError(MIGRATION)
    expected = {**E0_MCTS, **common["decoding"]["mcts"]}
    if options != expected:
        raise ValueError("E0 MCTS settings differ from the fixed protocol. " + MIGRATION)
    decoding["mcts"] = options
    model = config["model"]
    for key in ("model_path", "device", "dtype", "local_files_only"):
        if overrides.get(key) is not None:
            model[key] = overrides[key]
    model["model_path"] = str(model["model_path"])
    if (model["family"] != "llada" or model["dtype"] != common["model"]["dtype"]
            or set(model) - {"family", "model_path", "device", "dtype", "local_files_only"}):
        raise ValueError("E0 requires the pinned LLaDA checkpoint with bfloat16 weights.")
    conditions = [{"id": "confidence_t0", "method": "confidence", "temperature": 0.0,
                   "search_budget": 0, "requested_depth": None, "completion_method": "llada"}]
    for item in common["search"]["conditions"]:
        depth, budget = item["depth"], item["budget"]
        MCTSConfig(**options, init_tokens=depth, max_model_calls=budget).validate_length(256)
        conditions.append({"id": f"medal_lc{depth}_b{budget}", "configuration": item["label"], "method": "medal",
                           "temperature": 0.0, "search_budget": budget, "requested_depth": depth,
                           "completion_method": "llada"})
    if "conditions" in source and source["conditions"] != conditions:
        raise ValueError(MIGRATION)
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
    examples = select_examples(manifest, config["evaluation"]["stage"])
    config["data"]["revision"] = manifest["dataset"]["revision"]
    m = config["decoding"]["mcts"]
    parent = Path(output_dir) if output_dir else root / "LLaDA/GSM8K/experiment_0"
    parent = parent / config["evaluation"]["stage"] / "fixed-matrix-v5"
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
                               "stage": config["evaluation"]["stage"], "search_mode": m["stop_mode"],
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
                def emit(event):
                    events.append(_compact_event(event))
                    if event["stage"] == "forward" and counter.calls % 128 == 0:
                        print(f"{example['example_id']} | {condition['id']} | "
                              f"{counter.calls} actual forwards | {event.get('phase', 'search')}", flush=True)
                if condition["method"] == "medal":
                    kwargs.update(mcts={**m, "max_model_calls": condition["search_budget"],
                                        "init_tokens": condition["requested_depth"]},
                                  tokenizer=wrapper.tokenizer, on_event=emit,
                                  completion_method=condition["completion_method"])
                if condition["method"] == "medal":
                    kwargs["search_excluded_token_ids"] = config["termination"]["token_ids"]
                _sync(wrapper.device)
                started = perf_counter()
                result, details, status = None, {}, "completed"
                try:
                    result = DECODERS[condition["method"]](counter, prompt_ids.clone(), wrapper.mask_id, **kwargs)
                    details = result.details
                    if condition["method"] == "confidence":
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
    parser = argparse.ArgumentParser(description="Experiment 0: Baseline Test — Fixed-Budget MEDAL")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/gsm8k/experiment_0.yaml")
    parser.add_argument("--stage", choices=tuple(STAGE_COUNTS))
    parser.add_argument("--num-samples", type=int, help="Must match the frozen stage count: 5 dev, 20 evaluation, 200 test.")
    parser.add_argument("--init-tokens", type=int, help="Legacy option: rejected by the fixed-matrix protocol.")
    parser.add_argument("--search-mode", choices=("budget", "candidates"))
    parser.add_argument("--search-budgets", type=int, nargs="+", help="Legacy option: rejected by the fixed-matrix protocol.")
    parser.add_argument("--run-order", choices=("example", "condition"),
                        help="Finish all conditions per example, or all examples per condition.")
    parser.add_argument("--model-path", type=Path)
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
