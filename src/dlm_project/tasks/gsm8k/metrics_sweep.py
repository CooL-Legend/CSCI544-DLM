"""One model load for four baselines and fixed-simulation MCTS comparisons."""

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import torch

from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.config import load_config
from dlm_project.experiments.paths import create_run_directory
from dlm_project.experiments.primary_metrics import generate_primary_report
from dlm_project.paths import PROJECT_ROOT
from dlm_project.tasks.gsm8k import benchmark
from dlm_project.tasks.gsm8k.benchmark_dataset import get_manifest, select_examples
from dlm_project.tasks.gsm8k.dataset import build_prompt

BASELINES = ("random_mdm", "top_k_margin", "confidence", "best_of_5")


def run_metrics_sweep(source, *, budgets=(4, 8, 16, 32, 64), manifest_path=None,
                      output_dir=None, model_factory=None):
    """Run paired conditions sequentially and retain partial artifacts on failure.

    All conditions use MEDAL two-shot prompting and a single global output block.
    The reconstruction's simulation budget counts completed child rollouts;
    candidate evaluation and final completion are included in total NFEs.
    """
    if not budgets or any(type(budget) is not int or budget < 1 for budget in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("Simulation budgets must be distinct positive integers.")
    config = copy.deepcopy(source)
    config["generation"].update(steps=config["generation"]["gen_length"],
                                block_length=config["generation"]["gen_length"])
    config["prompt"]["medal_decomposition"] = True
    config = benchmark.resolve_settings("confidence", config, expected_family="llada")
    if config["generation"]["temperature"] <= 0:
        raise ValueError("The shared MEDAL comparison needs positive temperature.")
    root = Path(config.get("experiment_root", "results")).expanduser()
    root = root if root.is_absolute() else PROJECT_ROOT / root
    manifest_path = Path(manifest_path) if manifest_path else root / "manifests/gsm8k_test.json"
    manifest = get_manifest(manifest_path, cache_dir=config["data"].get("cache_dir"), revision=config["data"].get("revision"))
    examples = select_examples(manifest, config["evaluation"]["profile"], config["data"]["num_samples"])
    config["data"]["revision"] = manifest["dataset"]["revision"]
    directory, _ = create_run_directory({"output_dir": str(output_dir or root / "LLaDA/GSM8K/metrics_sweeps"),
                                         "experiment_timezone": config.get("experiment_timezone", "America/Los_Angeles")})
    save_json(directory / "config.json", {**config, "simulation_budgets": list(budgets)})
    save_json(directory / "manifest.json", manifest)
    setup = {"status": "running", "model_load_count": 0, "warmup_nfes": 0, "warmup_seconds": 0.0}
    runs = {}
    save_json(directory / "runs.json", runs)
    save_json(directory / "setup.json", setup)
    print(f"Metrics comparison artifacts: {directory}", flush=True)
    # Keep the user's MCTS hyperparameters, except the two explicitly varied caps.
    settings = copy.deepcopy(source.get("decoding", {}).get("mcts", {}))
    conditions = [(method, method, None) for method in BASELINES]
    conditions += [(f"mcts_{budget}_simulations", "mcts_confidence", budget) for budget in budgets]
    try:
        started = perf_counter()
        wrapper = (model_factory or benchmark._load_model)(config)
        benchmark._sync(wrapper.device)
        benchmark._verify_model(wrapper, "llada")
        setup.update(model_load_count=1, model_loading_seconds=perf_counter() - started,
                     model=wrapper.identity,
                     gpu_name=torch.cuda.get_device_name(wrapper.device) if torch.device(wrapper.device).type == "cuda" else None)
        started = perf_counter()
        prompt = build_prompt(examples[0]["question"], config["prompt"]["instructions"])
        ids = wrapper.encode_prompt(prompt)
        with torch.inference_mode():
            state = torch.full((1, ids.shape[1] + config["generation"]["gen_length"]), wrapper.mask_id,
                               dtype=torch.long, device=wrapper.device)
            state[:, :ids.shape[1]] = ids
            prediction = wrapper.model(state)
            del prediction, state, ids
        benchmark._sync(wrapper.device)
        setup.update(warmup_nfes=1, warmup_seconds=perf_counter() - started)
        save_json(directory / "setup.json", setup)
        for label, method, budget in conditions:
            condition = copy.deepcopy(config)
            if budget is not None:
                condition["decoding"]["mcts"] = {**settings, "max_simulations": budget,
                                                   "max_model_calls": None, "stop_mode": "budget"}
            def register(path, label=label):
                runs[label] = str(path)
                save_json(directory / "runs.json", runs)
            benchmark.run_evaluation(method, condition, manifest_path=manifest_path,
                                     output_dir=directory / "runs", model_instance=wrapper,
                                     on_run_created=register)
        setup["status"] = "completed"
    except BaseException as error:
        setup.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}")
        save_json(directory / "setup.json", setup)
        if runs:
            # Missing conditions remain visible in the report's requested list.
            try:
                generate_primary_report(runs, directory / "report",
                                        expected_conditions=[label for label, _, _ in conditions])
            except Exception as report_error:
                print(f"Partial report unavailable: {report_error}", flush=True)
        raise
    save_json(directory / "setup.json", setup)
    summary = generate_primary_report(runs, directory / "report",
                                      expected_conditions=[label for label, _, _ in conditions])
    print(f"Saved metrics report: {directory / 'report/report.md'}", flush=True)
    return directory, summary


def main():
    parser = argparse.ArgumentParser(description="One-load LLaDA GSM8K comparison with simulation budgets.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/gsm8k/decoding.yaml")
    parser.add_argument("--profile", choices=("smoke", "initial", "full"))
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--report-only", type=Path, metavar="COMPARISON_DIR")
    args = parser.parse_args()
    if args.report_only:
        runs = json.loads((args.report_only / "runs.json").read_text())
        saved = json.loads((args.report_only / "config.json").read_text())
        expected = [*BASELINES, *(f"mcts_{budget}_simulations" for budget in saved["simulation_budgets"])]
        generate_primary_report(runs, args.report_only / "report", expected_conditions=expected)
        print(f"Saved metrics report: {args.report_only / 'report/report.md'}")
        return
    source = load_config(args.config)
    config = benchmark.resolve_settings("confidence", source, profile=args.profile,
                                        num_samples=args.num_samples, expected_family="llada")
    # resolve_settings selects a decoder; retain the user's separate MCTS options.
    config["decoding"]["mcts"] = source.get("decoding", {}).get("mcts", {})
    directory, summary = run_metrics_sweep(config, budgets=args.budgets,
                                           manifest_path=args.manifest, output_dir=args.output_dir)
    if args.run_manifest:
        args.run_manifest.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.run_manifest, {"run_dir": str(directory), "status": "completed" if summary["complete"] else "incomplete",
                                      "model_family": "llada", "task": "gsm8k_metrics"})


if __name__ == "__main__":
    main()
