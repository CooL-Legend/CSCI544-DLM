"""Compare 128 and 256 generation tokens on the same first three puzzles."""

import argparse
import copy
import csv
from datetime import datetime, timezone
from pathlib import Path

from run_sudoku import ROOT, load_config, run_experiment, save_json, validate_config

GENERATION_LENGTHS = (128, 256)
NUM_PUZZLES = 3
METRICS = (
    "completion_rate", "clue_preservation_rate", "sudoku_validity_rate",
    "exact_accuracy", "cell_accuracy",
)


def comparison_config(config):
    config = copy.deepcopy(config)
    config["data"]["num_samples"] = NUM_PUZZLES
    config["generation"].update(steps=128, block_length=32)
    return config


def run_comparison(config, dataset, model, output_dir):
    """Reuse one loaded model and one puzzle list; change only gen_length.

    Each run saves its usual artifacts. The comparison manifest and CSV update
    after each completed run, so an interruption does not hide earlier results.
    """
    if len(dataset) < NUM_PUZZLES:
        raise ValueError("The comparison requires at least three puzzles.")
    config = comparison_config(config)
    configs = []
    for length in GENERATION_LENGTHS:
        run_config = copy.deepcopy(config)
        run_config["generation"]["gen_length"] = length
        validate_config(run_config)
        configs.append(run_config)
    puzzles = [dict(dataset[i]) for i in range(NUM_PUZZLES)]
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    save_json(output_dir / "puzzles.json", puzzles)
    report = {"status": "running", "num_samples": NUM_PUZZLES, "runs": []}
    save_json(output_dir / "comparison.json", report)
    columns = ["gen_length", "steps", "block_length", "processed", *METRICS, "run_dir"]
    for run_config in configs:
        length = run_config["generation"]["gen_length"]
        run_config.pop("experiment_root", None)
        run_config["output_dir"] = str(output_dir / f"length-{length}")
        run_dir, summary = run_experiment(run_config, puzzles, model)
        report["runs"].append({
            "gen_length": length, "steps": 128, "block_length": 32,
            "processed": summary["processed"],
            **{name: summary[name] for name in METRICS},
            "run_dir": str(run_dir),
        })
        save_json(output_dir / "comparison.json", report)
        with (output_dir / "comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(report["runs"])
    report["status"] = "completed"
    save_json(output_dir / "comparison.json", report)
    print(f"Comparison saved to {output_dir}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sudoku.yaml")
    parser.add_argument("--output-dir", type=Path, help="A new directory for this comparison")
    args = parser.parse_args()
    config = comparison_config(load_config(args.config))
    # Validate both settings before downloading data or loading weights.
    for length in GENERATION_LENGTHS:
        config["generation"]["gen_length"] = length
        validate_config(config)
    output_dir = args.output_dir or ROOT / "results/comparisons" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    from sudoku_dataset import SudokuDataset
    from Llada import LLaDA

    dataset = SudokuDataset(n=NUM_PUZZLES, cache_dir=config["data"].get("cache_dir"))
    model = LLaDA(**config["model"])
    run_comparison(config, dataset, model, output_dir)


if __name__ == "__main__":
    main()
