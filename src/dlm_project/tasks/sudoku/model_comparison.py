"""Compare original LLaDA and BGPO in separate sequential inference processes."""

import argparse
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

import yaml

from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.config import load_config, validate_config
from dlm_project.experiments.paths import create_run_directory, experiment_name
from dlm_project.decoding.mcts_decoding import resolve_decoding
from dlm_project.models.profiles import get_profile
from dlm_project.paths import PROJECT_ROOT as ROOT


RATE_KEYS = ("exact_accuracy", "cell_accuracy", "completion_rate",
             "clue_preservation_rate", "sudoku_validity_rate")


def comparison_configs(config):
    """Resolve independent single-model configs before starting either run."""
    if config["data"].get("source") != "bgpo_4x4":
        raise ValueError("The paired comparison requires data.source: bgpo_4x4.")
    if resolve_decoding(config.get("decoding"), config["generation"])["method"] != "baseline":
        raise ValueError("The paired comparison requires baseline decoding.")
    configs = {}
    for name, family, model_config in (
        ("baseline", "llada", config["comparison"]["baseline_model"]),
        ("bgpo", "llada_bgpo", config["model"]),
    ):
        child = copy.deepcopy(config)
        child.pop("comparison", None)
        child["model"] = copy.deepcopy(model_config)
        if child["model"].get("family") != family:
            raise ValueError(f"{name} must use model.family: {family}.")
        validate_config(child)
        configs[name] = child
    return configs


def _read_run(directory, family):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    records = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()
               if line.strip()]
    profile = get_profile(family)
    identity = config["provenance"]["model"]
    if (identity.get("family") != family
            or any(identity.get(k) != profile[k] for k in ("repo_id", "revision"))):
        raise ValueError(f"The {family} run does not identify the expected pinned checkpoint.")
    count = config["data"]["num_samples"]
    if (metrics["status"] != "completed" or not records or len(records) != count
            or metrics["processed"] != count or metrics["plots_saved"] != count):
        raise ValueError(f"Incomplete model run: {directory}")
    return config, metrics, records


def summarize_comparison(baseline_dir, bgpo_dir, output_dir):
    """Validate paired inputs before producing any combined scores or plots."""
    from dlm_project.tasks.sudoku.dataset import BGPO_PROVENANCE
    from dlm_project.tasks.sudoku.visualization import save_paired_prediction_plot

    baseline_config, baseline_metrics, baseline = _read_run(baseline_dir, "llada")
    bgpo_config, bgpo_metrics, bgpo = _read_run(bgpo_dir, "llada_bgpo")
    for config in (baseline_config, bgpo_config):
        if (config["data"].get("source") != "bgpo_4x4"
                or config["provenance"].get("dataset") != BGPO_PROVENANCE):
            raise ValueError("Comparison requires the same pinned BGPO 4x4 test set.")
    if (baseline_config["generation"] != bgpo_config["generation"]
            or baseline_config["decoding"] != bgpo_config["decoding"]
            or baseline_config["decoding"]["method"] != "baseline"):
        raise ValueError("Comparison generation/decoding settings do not match.")
    if len(baseline) != len(bgpo):
        raise ValueError("Comparison sample counts do not match.")
    seen = set()
    for index, (left, right) in enumerate(zip(baseline, bgpo)):
        keys = ("index", "source_index", "puzzle", "solution", "prompt", "board_size", "decoding_method")
        if any(left[k] != right[k] for k in keys):
            raise ValueError(f"Paired puzzle IDs, prompts, or inputs differ at index {index}.")
        if left["index"] != index or left["board_size"] != 4 or left["source_index"] in seen:
            raise ValueError(f"Invalid or duplicate comparison puzzle at index {index}.")
        seen.add(left["source_index"])
        expected_seed = (baseline_config["decoding"]["seed"] + index) % 2**63
        for record in (left, right):
            if record["decoding_method"] != "baseline":
                raise ValueError("Comparison predictions must use baseline decoding.")
            if record.get("decoding_stats") and record["decoding_stats"].get("seed") != expected_seed:
                raise ValueError(f"Comparison seed differs at index {index}.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, metrics in (("baseline", baseline_metrics), ("bgpo", bgpo_metrics)):
        latency = metrics["latency_seconds"]
        rows.append({
            "model": name, "processed": len(baseline),
            **{key: metrics[key] for key in RATE_KEYS},
            "mean_latency_seconds": latency / len(baseline) if latency is not None else None,
            "peak_gpu_memory_bytes": metrics["peak_gpu_memory_bytes"],
        })
    with (output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "paired_predictions.jsonl").open("w") as handle:
        for left, right in zip(baseline, bgpo):
            plot_file = f"plots/puzzle_{left['index']:03d}.png"
            pair = {"index": left["index"], "source_index": left["source_index"],
                    "puzzle": left["puzzle"], "solution": left["solution"], "prompt": left["prompt"],
                    "baseline": left, "bgpo": right, "plot_file": plot_file}
            handle.write(json.dumps(pair) + "\n")
            handle.flush()
            save_paired_prediction_plot(left, right, output_dir / plot_file)
    summary = {
        "status": "completed", "processed": len(baseline), "plots_saved": len(baseline),
        "dataset": BGPO_PROVENANCE,
        "model_runs": {"baseline": str(Path(baseline_dir).resolve()), "bgpo": str(Path(bgpo_dir).resolve())},
        "models": rows,
        "bgpo_minus_baseline": {key: bgpo_metrics[key] - baseline_metrics[key] for key in RATE_KEYS},
        "generation": baseline_config["generation"], "decoding": baseline_config["decoding"],
    }
    save_json(output_dir / "comparison.json", summary)
    return summary


def run_comparison(config, run_manifest=None):
    children = comparison_configs(config)
    from dlm_project.tasks.sudoku.dataset import load_sudoku_dataset

    # Validate the selected dataset before paying for either model load.
    dataset = load_sudoku_dataset(config["data"])
    root = Path(config.get("experiment_root", "results")).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    parent = root / "Comparisons" / "Sudoku4x4" / experiment_name(config["generation"], config.get("decoding"))
    directory_config = {"output_dir": str(parent),
                        "experiment_timezone": config.get("experiment_timezone", "America/Los_Angeles")}
    directory, started = create_run_directory(directory_config)
    saved_config = copy.deepcopy(config)
    saved_config["provenance"] = {"dataset": dict(dataset.provenance)}
    saved_config["run"] = {"directory": str(directory), "started_at": started.isoformat()}
    save_json(directory / "config.json", saved_config)
    state = {"status": "running", "requested": len(dataset), "model_runs": {}}

    def save_state():
        save_json(directory / "comparison.json", state)
        if run_manifest is not None:
            receipt = Path(run_manifest)
            receipt.parent.mkdir(parents=True, exist_ok=True)
            save_json(receipt, {"run_dir": str(directory), "status": state["status"]})

    save_state()
    print(f"Saving comparison to {directory}", flush=True)
    try:
        for name, child in children.items():
            config_path = directory / f"{name}-config.yaml"
            receipt = directory / f"{name}-run.json"
            config_path.write_text(yaml.safe_dump(child, sort_keys=False))
            print(f"Running {name} in a fresh process", flush=True)
            subprocess.run([
                sys.executable, "-u", str(ROOT / "scripts/run_sudoku.py"),
                "--config", str(config_path), "--run-manifest", str(receipt),
            ], check=True, cwd=ROOT)
            result = json.loads(receipt.read_text())
            if result["status"] != "completed":
                raise RuntimeError(f"The {name} model run did not complete.")
            state["model_runs"][name] = result["run_dir"]
            save_state()
        state = summarize_comparison(state["model_runs"]["baseline"], state["model_runs"]["bgpo"], directory)
        save_state()
    except BaseException as error:
        state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        state["error"] = str(error)
        save_state()
        raise
    print(json.dumps(state, indent=2), flush=True)
    return directory, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sudoku/llada_bgpo_4x4.yaml")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--run-manifest", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.num_samples is not None:
        config["data"]["num_samples"] = args.num_samples
    comparison_configs(config)
    if args.dataset_only:
        from dlm_project.tasks.sudoku.dataset import load_sudoku_dataset, item_prompt
        dataset = load_sudoku_dataset(config["data"])
        print(f"Loaded {len(dataset)} paired 4x4 test puzzles. First prompt:\n{item_prompt(dataset[0])}")
        return
    run_comparison(config, args.run_manifest)


if __name__ == "__main__":
    main()
