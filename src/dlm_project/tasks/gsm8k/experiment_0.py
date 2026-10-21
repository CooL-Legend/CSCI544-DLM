"""Experiment 0: paired greedy, no-search MEDAL, and fixed-budget MEDAL runs."""

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import signal
import subprocess
from time import perf_counter

import torch

from dlm_project.decoding import confidence, entropy, medal_no_search, mcts_confidence
from dlm_project.decoding.common import validate_generation
from dlm_project.decoding.decoder_result import CountingModel
from dlm_project.decoding.mcts_decoding import MCTSConfig, SearchIncompleteError
from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.config import load_config
from dlm_project.experiments.manifests import digest
from dlm_project.experiments.paths import create_run_directory
from dlm_project.models.profiles import get_profile
from dlm_project.paths import PROJECT_ROOT
from dlm_project.tasks.gsm8k.e0_dataset import STAGE_COUNTS, get_manifest, select_examples
from dlm_project.tasks.gsm8k.e0_prompt import build_e0_prompt
from dlm_project.tasks.gsm8k.e0_tokens import decode_output, resolve_termination_tokens
from dlm_project.tasks.gsm8k.evaluation import score_answer


PROTOCOL_VERSION = "e0_fixed_budget_v4"
DECODERS = {"confidence": confidence.decode, "entropy": entropy.decode,
            "medal_no_search": medal_no_search.decode, "medal": mcts_confidence.decode}


def resolve_settings(source, **overrides):
    config = copy.deepcopy(source)
    evaluation = config.setdefault("evaluation", {})
    for key, default in (("stage", "dev"), ("search_mode", "budget"), ("run_order", "example"),
                         ("search_budgets", [512, 1024, 2048])):
        evaluation[key] = overrides.get(key) if overrides.get(key) is not None else evaluation.get(key, default)
    if evaluation["stage"] not in STAGE_COUNTS or evaluation["search_mode"] not in ("budget", "candidates"):
        raise ValueError("E0 requires stage dev/evaluation/test and search_mode budget/candidates.")
    if evaluation["run_order"] not in ("example", "condition"):
        raise ValueError("E0 run_order must be example or condition.")
    budgets = evaluation["search_budgets"]
    if (not isinstance(budgets, (list, tuple)) or not budgets
            or any(type(b) is not int or b < 1 for b in budgets) or len(set(budgets)) != len(budgets)):
        raise ValueError("Search budgets must be distinct positive integer forward caps.")
    evaluation["search_budgets"] = list(budgets)
    data = config.setdefault("data", {})
    count = overrides.get("num_samples", data.get("num_samples"))
    data["num_samples"] = STAGE_COUNTS[evaluation["stage"]] if count is None else count
    if type(data["num_samples"]) is not int or data["num_samples"] < 1:
        raise ValueError("num_samples must be a positive integer.")
    data.setdefault("sample_seed", 1)
    if type(data["sample_seed"]) is not int or data["sample_seed"] < 0:
        raise ValueError("sample_seed must be a nonnegative integer.")
    if overrides.get("cache_dir") is not None:
        data["cache_dir"] = str(overrides["cache_dir"])
    if overrides.get("dataset_revision") is not None:
        data["revision"] = overrides["dataset_revision"]
    prompt = config.setdefault("prompt", {})
    if (not isinstance(prompt.get("template"), str)
            or prompt["template"].count("{GSM8K_question}") != 1
            or "instructions" in prompt
            or prompt.get("medal_decomposition", False) is not False
            or prompt.get("custom_question") is not None):
        raise ValueError("E0 requires prompt.template with one {GSM8K_question} placeholder; "
                         "automatic medal_decomposition, legacy instructions and custom questions are disabled.")
    prompt["medal_decomposition"] = False
    generation = config.setdefault("generation", {})
    length = generation.setdefault("gen_length", 256)
    generation.setdefault("steps", length)
    generation.setdefault("block_length", length)
    if set(generation) != {"gen_length", "steps", "block_length"}:
        raise ValueError("E0 generation contains only gen_length, steps, block_length; use decoding.completion_temperature for MEDAL.")
    if generation["steps"] != length or generation["block_length"] != length:
        raise ValueError("E0 requires steps = block_length = gen_length.")
    decoding = config.setdefault("decoding", {})
    decoding.setdefault("seed", 1)
    decoding.setdefault("completion_method", "llada")
    temperature = decoding.setdefault("completion_temperature", 0.0)
    if decoding["completion_method"] not in ("llada", "medal"):
        raise ValueError("decoding.completion_method must be llada or medal.")
    validate_generation(**generation, temperature=temperature, seed=decoding["seed"])
    if decoding["completion_method"] == "medal" and temperature <= 0:
        raise ValueError("The adjusted-score MEDAL sampler requires positive completion_temperature; use llada at T=0.")
    options = dict(decoding.get("mcts", {}))
    options.setdefault("init_tokens", 5)
    if overrides.get("init_tokens") is not None:
        options["init_tokens"] = overrides["init_tokens"]
    if options.get("max_simulations") is not None:
        raise ValueError("E0 caps forwards, not simulation counts.")
    options.update(stop_mode=evaluation["search_mode"], max_model_calls=max(budgets), max_simulations=None)
    mcts = MCTSConfig(**options)
    mcts.validate_length(length)
    decoding["mcts"] = asdict(mcts)
    model = config.setdefault("model", {})
    if model.setdefault("family", "llada") != "llada":
        raise ValueError("E0 requires the pinned original LLaDA model.")
    for key, default in (("model_path", get_profile()["repo_id"]), ("device", "cuda"),
                         ("dtype", "bfloat16"), ("local_files_only", False)):
        model.setdefault(key, default)
        if overrides.get(key) is not None:
            model[key] = str(overrides[key]) if key == "model_path" else overrides[key]
    config["protocol_version"] = PROTOCOL_VERSION
    config["conditions"] = [
        {"id": "confidence_t0", "method": "confidence", "temperature": 0.0, "search_budget": None},
        {"id": "entropy_t0", "method": "entropy", "temperature": 0.0, "search_budget": None},
        {"id": f"medal_no_search_t{temperature:g}", "method": "medal_no_search", "temperature": temperature, "search_budget": None,
         "completion_method": decoding["completion_method"]},
        *[{"id": f"medal_{evaluation['search_mode']}_b{b}", "method": "medal",
           "temperature": temperature, "search_budget": b, "completion_method": decoding["completion_method"]} for b in budgets],
    ]
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
    manifest_path = Path(manifest_path) if manifest_path else root / "manifests/gsm8k_e0.json"
    manifest = get_manifest(manifest_path, revision=config["data"].get("revision"),
                            sample_seed=config["data"]["sample_seed"], cache_dir=config["data"].get("cache_dir"))
    examples = select_examples(manifest, config["evaluation"]["stage"], config["data"]["num_samples"])
    config["data"]["revision"] = manifest["dataset"]["revision"]
    m = config["decoding"]["mcts"]
    parent = Path(output_dir) if output_dir else root / "LLaDA/GSM8K/experiment_0"
    parent = parent / config["evaluation"]["stage"] / f"{m['stop_mode']}-lc{m['init_tokens']}"
    directory, _ = create_run_directory({"output_dir": str(parent),
        "experiment_timezone": config.get("experiment_timezone", "America/Los_Angeles")})
    config["run"] = {"example_ids": [row["example_id"] for row in examples],
                     "manifest_hash": digest(manifest)}
    config["provenance"] = _code_identity()
    records, setup = [], {"status": "running", "model_load_count": 0, "warmup_nfes": 0,
                          "tokenization_seconds": 0.0, "scoring_seconds": 0.0,
                          "artifact_writing_seconds": 0.0, "active_total_nfes": 0}
    (directory / "traces").mkdir()
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
                    seed = (config["decoding"]["seed"] + 5 * example["row_id"]) % 2**63
                    prompt_cache[example["example_id"]] = prompt, prompt_ids, seed
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
                if condition["method"] in ("medal", "medal_no_search"):
                    kwargs.update(mcts={**m, "max_model_calls": condition["search_budget"] or m["max_model_calls"]},
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
                    if condition["method"] in ("confidence", "entropy"):
                        events.extend(_compact_event(event) for event in result.trace)
                except SearchIncompleteError as error:
                    if m["stop_mode"] != "candidates":
                        raise
                    details, status = error.stats, "search_incomplete"
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
                row = {**example, "protocol_version": PROTOCOL_VERSION, "condition": condition["id"],
                       "method": condition["method"], "temperature": condition["temperature"],
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
                       "shortlist_size": details.get("shortlist_size", 0),
                       "shortlist_underfilled": details.get("shortlist_underfilled", False),
                       "selection_source": details.get("selection_source", "no_search"),
                       "fallback_decode_used": status == "completed" and details.get("fallback_decode_used", False),
                       "stop_reason": details.get("stop_reason", "all_masks_filled"),
                       "trace_file": trace_file, "details": details, **decoded, **score}
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
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--init-tokens", type=int)
    parser.add_argument("--search-mode", choices=("budget", "candidates"))
    parser.add_argument("--search-budgets", type=int, nargs="+")
    parser.add_argument("--run-order", choices=("example", "condition"),
                        help="Finish all conditions per example, or all examples per condition.")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--local-files-only", action="store_true", default=None)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--manifest", type=Path)
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
