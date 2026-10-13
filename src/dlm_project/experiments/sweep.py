"""Sequential, paired decoder sweeps with one pinned LLaDA load per process.

Task-specific prompting and scoring remain in task modules. Search stays in
mcts_decoding.py. Report-only execution never constructs a model or loads data.
"""

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import torch

from dlm_project.decoding.mcts_decoding import resolve_decoding, IMPLEMENTATION_VERSION
from dlm_project.decoding.medal_prompting import get_gsm8k_demonstrations, gsm8k_prompt, sudoku_prompt
from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.config import load_config, validate_config
from dlm_project.experiments.manifests import POOL_SIZES, digest, get_manifest, select_examples
from dlm_project.experiments.paths import create_run_directory
from dlm_project.models.profiles import get_profile
from dlm_project.paths import PROJECT_ROOT


def resolve_settings(task, config, *, profile=None, num_samples=None, budgets=None):
    """Resolve CLI overrides without mutating the caller's YAML dictionary."""
    if task not in POOL_SIZES:
        raise ValueError(f"Unknown sweep task: {task}")
    config = copy.deepcopy(config)
    sweep = config.setdefault("sweep", {})
    sweep["profile"] = profile or sweep.get("profile", "smoke")
    if sweep["profile"] not in ("smoke", "milestone"):
        raise ValueError("sweep.profile must be smoke or milestone.")
    sweep.setdefault("sample_seed", 1)
    sweep.setdefault("warmup", True)
    if type(sweep["sample_seed"]) is not int or not 0 <= sweep["sample_seed"] < 2**63:
        raise ValueError("sweep.sample_seed must be an integer in [0, 2**63).")
    if type(sweep["warmup"]) is not bool:
        raise ValueError("sweep.warmup must be boolean.")
    values = budgets if budgets is not None else sweep.get("budgets", [128, 256, 512])
    if (not isinstance(values, list) or not values
            or any(type(value) is not int or value < 1 for value in values)
            or len(set(values)) != len(values)):
        raise ValueError("sweep.budgets must contain distinct positive integer NFE caps.")
    sweep["budgets"] = sorted(values)
    default_count = 3 if sweep["profile"] == "smoke" else (50 if task == "gsm8k" else 30)
    data = config.setdefault("data", {})
    count = num_samples if num_samples is not None else data.get("num_samples")
    data["num_samples"] = default_count if count is None else count
    partition = "development" if sweep["profile"] == "smoke" else "evaluation"
    if type(data["num_samples"]) is not int or not 1 <= data["num_samples"] <= POOL_SIZES[task][partition]:
        raise ValueError(f"Select 1–{POOL_SIZES[task][partition]} {partition} examples.")
    config["task"] = task
    config["evaluation_variant"] = "Budget-limited MEDAL reconstruction; inline decomposition; mean stepwise IG"
    config["report_schema_version"] = 2
    config["implementation_version"] = IMPLEMENTATION_VERSION
    config["trace_policy"] = "all search-cycle and completion events; no vocabulary logits"
    if config["model"].get("family", "llada") != "llada":
        raise ValueError("Both sweep scripts require original model.family: llada.")
    config["model"].setdefault("family", "llada")
    config["model"].setdefault("model_path", get_profile()["repo_id"])
    if task == "gsm8k":
        instructions = config.get("prompt", {}).get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("GSM8K requires nonempty prompt.instructions.")
    elif data.get("source", "bgpo_4x4") != "bgpo_4x4":
        raise ValueError("The Sudoku sweep requires data.source: bgpo_4x4.")
    else:
        data["source"] = "bgpo_4x4"
    decoding = config.setdefault("decoding", {})
    decoding["method"] = "mcts_entropy"
    mcts = decoding.setdefault("mcts", {})
    if mcts.get("stop_mode", "budget") != "budget":
        raise ValueError("Sweeps require mcts.stop_mode: budget.")
    mcts["stop_mode"] = "budget"
    mcts["max_model_calls"] = sweep["budgets"][0]
    config["decoding"] = resolve_decoding(decoding, config["generation"])
    validate_config(config)
    config["conditions"] = make_conditions(config)
    return config


def make_conditions(config):
    conditions = []
    for method, budget in [("baseline", None), ("confidence", None),
                            *[("mcts_entropy", b) for b in config["sweep"]["budgets"]]]:
        decoding = copy.deepcopy(config["decoding"])
        decoding["method"] = method
        if budget is not None:
            decoding["mcts"]["max_model_calls"] = budget
        conditions.append({"id": f"mcts_{budget}" if budget is not None else method,
                           "method": method, "search_budget": budget, "decoding": decoding,
                           "generation": {**config["generation"], "temperature": 1.0 if budget is not None else 0.0},
                           "prompt_template": ("gsm8k_train_two_shot" if config["task"] == "gsm8k" else
                                               "sudoku_4_synthetic_two_shot") if budget is not None else None})
    return conditions


def user_prompt(task, example, config):
    if task == "gsm8k":
        from dlm_project.tasks.gsm8k.dataset import build_prompt
        return build_prompt(example["question"], config["prompt"]["instructions"])
    return example["prompt"]


def score_output(task, output, example):
    if task == "gsm8k":
        from dlm_project.tasks.gsm8k.evaluation import score_answer
        score = score_answer(output, example["reference_solution"])
        return {**score, "exact": score["exact_match"]}
    from dlm_project.tasks.sudoku.evaluation import score_prediction, is_valid_sudoku, preserves_clues
    score = score_prediction(output, example["solution"], 4)
    grid = [list(score["prediction"][row:row + 4]) for row in range(0, 16, 4)] if score["parse_success"] else None
    return {**score, "parsed_grid": grid, "exact_match": score["exact"],
            "cell_accuracy": score["correct_cells"] / 16,
            "clues_preserved": preserves_clues(example["puzzle"], score["prediction"], 4),
            "valid_sudoku": is_valid_sudoku(score["prediction"], 4)}


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _load_model(config):
    from dlm_project.models.llada import LLaDA
    return LLaDA(**config["model"])


def _verify_identity(model):
    profile = get_profile("llada")
    if any(model.identity.get(key) != profile[key] for key in ("repo_id", "revision")):
        raise ValueError("Sweeps require the pinned original LLaDA snapshot, including local model metadata.")


@torch.inference_mode()
def _warmup(model, prompt, length):
    """One setup forward, excluded from per-example NFEs but recorded separately."""
    ids = model.encode_prompt(prompt)
    state = torch.full((1, ids.shape[1] + length), model.mask_id, dtype=torch.long, device=model.device)
    state[:, :ids.shape[1]] = ids
    output = model.model(state)
    del output
    _sync(model.device)


def _environment():
    versions = {}
    for package in ("torch", "transformers", "datasets", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"python": platform.python_version(), "packages": versions, "git_commit": commit}


def _record(task, example, condition, prompt, prediction, seed, trace_file, identity, evaluations,
            base_prompt=None, decomposition=None):
    stats = prediction["stats"]
    search, completion = stats["search_model_calls"], stats["completion_model_calls"]
    if stats["total_model_calls"] != search + completion:
        raise ValueError("Decoder total NFE accounting does not reconcile.")
    is_search = condition["method"] == "mcts_entropy"
    if is_search:
        phase_total = sum(stats.get(f"{phase}_model_calls", 0) for phase in
                          ("tree_analysis", "simulation", "candidate_evaluation"))
        if search != phase_total or search > condition["search_budget"]:
            raise ValueError("Search phase NFEs must reconcile and respect the cap.")
    parents = len(evaluations["parents"])
    record = {"example_id": example["example_id"], "manifest_position": example["position"],
              "partition": example["partition"], "condition": condition["id"],
              "method": condition["method"], "model_revision": identity["revision"], "seed": seed,
              "requested_search_budget": condition["search_budget"],
              "prompt": prompt, "prompt_hash": digest(prompt), "content_hash": example["content_hash"],
              "raw_output": prediction["text"], "trace_file": trace_file,
              "search_nfes": search, "completion_nfes": completion, "total_nfes": search + completion,
              "latency_seconds": stats["latency_seconds"],
              "search_latency_seconds": stats["search_latency_seconds"],
              "completion_latency_seconds": stats["completion_latency_seconds"],
              "peak_gpu_memory_bytes": stats["peak_gpu_memory_bytes"],
              "model_device": stats["model_device"],
              "evaluated_children": evaluations["children"] if is_search else None,
              "expanded_parents": parents if is_search else None,
              "realized_branching_factor": (evaluations["children"] / parents if parents else 0.0)
              if is_search else None}
    record.update(implementation_version=stats.get("implementation_version"),
                  base_prompt=base_prompt or prompt, base_prompt_hash=digest(base_prompt or prompt),
                  effective_prompt_hash=digest(prompt), decomposition=decomposition,
                  prompt_template=decomposition["template"] if decomposition else None)
    for phase in ("tree_analysis", "simulation", "candidate_evaluation"):
        record[f"{phase}_nfes"] = stats.get(f"{phase}_model_calls", 0)
        record[f"{phase}_latency_seconds"] = stats.get(f"{phase}_latency_seconds", 0)
    for key in ("cycles", "stop_mode", "requested_depth", "max_depth", "selected_depth", "candidate_count",
                "stop_reason", "fallback_used", "fallback_reason", "selected_node_id"):
        record[key] = stats.get(key) if is_search else None
    return {**record, **score_output(task, prediction["text"], example)}


def run_sweep(task, config, *, manifest_path=None, output_dir=None, model_factory=None):
    """Run one task, persisting each result before attempting the next condition."""
    from dlm_project.experiments.sweep_report import generate_report

    config = resolve_settings(task, config)
    root = Path(config.get("experiment_root", "results")).expanduser()
    root = root if root.is_absolute() else PROJECT_ROOT / root
    manifest_path = Path(manifest_path) if manifest_path else root / "manifests" / f"{task}-seed{config['sweep']['sample_seed']}.json"
    manifest = get_manifest(manifest_path, task, sample_seed=config["sweep"]["sample_seed"],
                            cache_dir=config["data"].get("cache_dir"), revision=config["data"].get("revision"))
    examples = select_examples(manifest, config["sweep"]["profile"], config["data"]["num_samples"])
    config["data"]["revision"] = manifest["dataset"]["revision"]
    dataset_name = "GSM8K" if task == "gsm8k" else "Sudoku4x4"
    output_dir = output_dir or config.get("output_dir")
    parent = Path(output_dir).expanduser() if output_dir else root / "LLaDA" / dataset_name / "sweeps" / "medal-v1"
    directory, started = create_run_directory({"output_dir": str(parent),
                                               "experiment_timezone": config.get("experiment_timezone", "America/Los_Angeles")})
    config["run"] = {"directory": str(directory), "started_at": started.isoformat(),
                     "example_ids": [row["example_id"] for row in examples],
                     "manifest_hash": digest(manifest), "manifest_source": str(manifest_path.resolve()),
                     "environment": _environment()}
    demonstrations = None
    if task == "gsm8k":
        demonstrations = get_gsm8k_demonstrations(manifest_path.with_name("gsm8k_decomposition_train.json"),
                          revision=manifest["dataset"]["revision"], cache_dir=config["data"].get("cache_dir"))
        save_json(directory / "decomposition_manifest.json", demonstrations)
    save_json(directory / "config.json", config)
    save_json(directory / "manifest.json", manifest)
    state = {"status": "running", "completed_predictions": 0,
             "model_load_count": 0, "model_loading_seconds": None,
             "warmup_model_calls": 0, "warmup_seconds": 0.0}
    save_json(directory / "run_state.json", state)
    (directory / "traces").mkdir()
    (directory / "predictions.jsonl").touch()
    print(f"Sweep artifacts: {directory}", flush=True)
    try:
        started_loading = perf_counter()
        model = (model_factory or _load_model)(config)
        state["model_load_count"] = 1
        _sync(model.device)
        state["model_loading_seconds"] = perf_counter() - started_loading
        _verify_identity(model)
        config["run"]["model"] = model.identity
        config["run"]["gpu"] = (torch.cuda.get_device_name(model.device)
                                  if torch.device(model.device).type == "cuda" else None)
        save_json(directory / "config.json", config)
        if config["sweep"]["warmup"]:
            started_warmup = perf_counter()
            state["warmup_model_calls"] = 1
            _warmup(model, user_prompt(task, examples[0], config), config["generation"]["gen_length"])
            state["warmup_seconds"] = perf_counter() - started_warmup
        save_json(directory / "run_state.json", state)
        with (directory / "predictions.jsonl").open("a") as predictions:
            for example in examples:
                prompt = user_prompt(task, example, config)
                seed = (config["decoding"]["seed"] + example["position"]) % 2**63
                for condition in config["conditions"]:
                    state.update(active_example=example["example_id"], active_condition=condition["id"], active_prediction_model_calls=0)
                    save_json(directory / "run_state.json", state)
                    base_prompt = prompt = user_prompt(task, example, config)
                    decomposition = None
                    if condition["method"] == "mcts_entropy":
                        prompt, decomposition = (gsm8k_prompt(base_prompt, example["question"], demonstrations)
                            if task == "gsm8k" else sudoku_prompt(base_prompt, 4, example["puzzle"]))
                    decoding = {**condition["decoding"], "seed": seed}
                    trace_file = f"traces/example_{example['position']:04d}_{condition['id']}.jsonl"
                    evaluations = {"children": 0, "parents": set()}
                    with (directory / trace_file).open("w") as trace:
                        def emit(event):
                            trace.write(json.dumps(event, allow_nan=False) + "\n")
                            trace.flush()
                            state["active_prediction_model_calls"] = event.get("generation_nfes",
                                event.get("search_model_calls", event.get("stats", {}).get("total_model_calls", 0)))
                            if event["stage"] == "evaluation":
                                evaluations["children"] += 1
                                evaluations["parents"].add(event["parent_id"])
                        prediction = model.predict_with_stats(prompt, **condition["generation"],
                                                              decoding=decoding, on_event=emit)
                    record = _record(task, example, condition, prompt, prediction, seed,
                                     trace_file, model.identity, evaluations, base_prompt, decomposition)
                    predictions.write(json.dumps(record, allow_nan=False) + "\n")
                    predictions.flush()
                    state["active_prediction_model_calls"] = 0
                    state["completed_predictions"] += 1
                    save_json(directory / "run_state.json", state)
                    generate_report(directory, plots=False)
                    print(f"{example['example_id']} | {condition['id']} | exact={record['exact']} | "
                          f"NFEs={record['total_nfes']} | {record['latency_seconds']:.2f}s", flush=True)
        state.update(status="reporting", active_example=None, active_condition=None)
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}")
        save_json(directory / "run_state.json", state)
        try:
            generate_report(directory)
        except Exception as report_error:
            print(f"Could not finish partial report: {report_error}", flush=True)
        raise
    save_json(directory / "run_state.json", state)
    try:
        summary = generate_report(directory)
    except Exception as error:
        state.update(status="report_failed", error=f"{type(error).__name__}: {error}")
        save_json(directory / "run_state.json", state)
        generate_report(directory, plots=False)
        raise
    print(f"Saved report: {directory / 'report.md'}", flush=True)
    return directory, summary


def main(task):
    parser = argparse.ArgumentParser(description=f"Paired {task} budget sweep using pinned original LLaDA.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / f"configs/{task}/llada_sweep.yaml")
    parser.add_argument("--profile", choices=("smoke", "milestone"))
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--manifest", type=Path, help="Create once or reuse a frozen, self-contained example manifest")
    parser.add_argument("--output-dir", type=Path, help="Parent directory; each invocation creates a new timestamp")
    parser.add_argument("--report-only", type=Path, metavar="RUN_DIR", help="Regenerate saved tables/plots without loading model or dataset")
    args = parser.parse_args()
    if args.report_only:
        from dlm_project.experiments.sweep_report import generate_report
        if json.loads((args.report_only / "config.json").read_text())["task"] != task:
            parser.error("The saved run belongs to the other task.")
        generate_report(args.report_only)
        print(f"Saved report: {args.report_only / 'report.md'}")
        return
    config = resolve_settings(task, load_config(args.config), profile=args.profile,
                              num_samples=args.num_samples, budgets=args.budgets)
    run_sweep(task, config, manifest_path=args.manifest, output_dir=args.output_dir)
