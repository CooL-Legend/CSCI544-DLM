"""Evaluate LLaDA on Sudoku and save each prediction as it completes."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The configuration must be a YAML mapping.")
    return config


def validate_config(config):
    n = config["data"]["num_samples"]
    if type(n) is not int or not 1 <= n <= 1_000_000:
        raise ValueError("data.num_samples must be between 1 and 1,000,000.")
    generation = config["generation"]
    for key in ("steps", "gen_length", "block_length"):
        if type(generation[key]) is not int or generation[key] < 1:
            raise ValueError(f"generation.{key} must be a positive integer.")
    steps, length, block = (generation[k] for k in ("steps", "gen_length", "block_length"))
    if length % block or steps > length or steps % (length // block):
        raise ValueError("gen_length must divide into blocks, and steps must be <= gen_length and divisible by the number of blocks.")
    if generation["temperature"] < 0:
        raise ValueError("temperature must be nonnegative.")
    if generation["remasking"] not in {"low_confidence", "random"}:
        raise ValueError("remasking must be low_confidence or random.")
    if config["model"]["dtype"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("dtype must be bfloat16, float16, or float32.")


def score_prediction(output, target):
    # Preserve the original experiment's parser: take the first 81 nonzero digits.
    digits = "".join(re.findall(r"[1-9]", output))
    prediction = digits[:81] if len(digits) >= 81 else ""
    return {
        "prediction": prediction,
        "exact": prediction == target,
        "correct_cells": sum(p == t for p, t in zip(prediction, target)),
    }


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_experiment(config, dataset, model):
    output_dir = Path(config["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    run_dir = output_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "config.json", config)
    summary = {
        "status": "running", "requested": len(dataset), "processed": 0,
        "exact_accuracy": 0.0, "cell_accuracy": 0.0,
    }
    save_json(run_dir / "metrics.json", summary)
    print(f"Saving results to {run_dir}", flush=True)
    exact = correct_cells = 0
    from sudoku_dataset import format_prompt

    with (run_dir / "predictions.jsonl").open("w") as handle:
        for index in range(len(dataset)):
            item = dataset[index]
            output = model.predict(format_prompt(item["puzzle"]), **config["generation"])
            score = score_prediction(output, item["solution"])
            record = {"index": index, **item, "raw_output": output, **score}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            exact += score["exact"]
            correct_cells += score["correct_cells"]
            processed = index + 1
            summary.update(
                processed=processed,
                exact_accuracy=exact / processed,
                cell_accuracy=correct_cells / (81 * processed),
            )
            save_json(run_dir / "metrics.json", summary)
            print(f"{index:03d} | exact={score['exact']} | output={score['prediction']}", flush=True)

    summary["status"] = "completed"
    save_json(run_dir / "metrics.json", summary)
    print("Exact accuracy:", summary["exact_accuracy"], flush=True)
    print("Cell accuracy:", summary["cell_accuracy"], flush=True)
    return run_dir, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sudoku.yaml")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--model-path", help="Hugging Face ID or local model directory")
    parser.add_argument("--output-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dataset-only", action="store_true", help="Check the dataset without loading model weights")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.num_samples is not None:
        config["data"]["num_samples"] = args.num_samples
    if args.model_path is not None:
        config["model"]["model_path"] = args.model_path
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.local_files_only:
        config["model"]["local_files_only"] = True
    validate_config(config)

    from sudoku_dataset import SudokuDataset, format_prompt
    dataset = SudokuDataset(n=config["data"]["num_samples"], cache_dir=config["data"].get("cache_dir"))
    if args.dataset_only:
        print(f"Loaded {len(dataset)} puzzle(s). First prompt:\n{format_prompt(dataset[0]['puzzle'])}")
        return
    from Llada import LLaDA
    model = LLaDA(**config["model"])
    run_experiment(config, dataset, model)


if __name__ == "__main__":
    main()
