"""Evaluate LLaDA on Sudoku and save each prediction as it completes."""

import argparse
from datetime import timezone
import json
from pathlib import Path
import copy
import subprocess

from dlm_project.experiments.config import load_config, validate_config
from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.paths import DEFAULT_TIMEZONE, create_run_directory, experiment_name
from dlm_project.models.profiles import get_profile
from dlm_project.decoding.mcts_decoding import resolve_decoding
from dlm_project.tasks.sudoku.evaluation import is_valid_sudoku, preserves_clues, score_prediction
from dlm_project.paths import PROJECT_ROOT as ROOT


def run_experiment(config, dataset, model):
    from dlm_project.tasks.sudoku.visualization import save_prediction_plot

    config = copy.deepcopy(config)
    config["decoding"] = resolve_decoding(config.get("decoding"), config["generation"])
    config["data"]["num_samples"] = len(dataset)
    family = config["model"].setdefault("family", "llada")
    profile = get_profile(family)
    config["model"].setdefault("model_path", profile["repo_id"])
    run_dir, started = create_run_directory(config)
    config["output_dir"] = str(run_dir.parent)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                             text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    identity = getattr(model, "identity", {
        "family": family, "model_path": config["model"]["model_path"],
        "repo_id": profile["repo_id"], "revision": profile["revision"],
        "mask_id": profile["mask_id"],
    })
    config["provenance"] = {**config.get("provenance", {}), "git_commit": git_commit,
                            "model": identity}
    config["run"] = {
        "directory": str(run_dir),
        "experiment_name": experiment_name(config["generation"], config["decoding"]),
        "timezone": config.get("experiment_timezone", DEFAULT_TIMEZONE),
        "started_at": started.isoformat(), "started_at_utc": started.astimezone(timezone.utc).isoformat(),
    }
    save_json(run_dir / "config.json", config)
    summary = {
        "status": "running", "requested": len(dataset), "processed": 0,
        "exact_accuracy": 0.0, "cell_accuracy": 0.0,
        "completion_rate": 0.0, "clue_preservation_rate": 0.0,
        "sudoku_validity_rate": 0.0,
        "plots_saved": 0,
        "decoding": config["decoding"],
        "search_model_calls": None, "completion_model_calls": None,
        "latency_seconds": None, "peak_gpu_memory_bytes": None,
    }
    save_json(run_dir / "metrics.json", summary)
    print(f"Saving results to {run_dir}", flush=True)
    exact = correct_cells = complete = clues_preserved = valid_sudoku = 0
    from dlm_project.tasks.sudoku.dataset import format_prompt

    with (run_dir / "predictions.jsonl").open("w") as handle:
        for index in range(len(dataset)):
            item = dataset[index]
            trace_file, stats = None, None
            if hasattr(model, "predict_with_stats"):
                trace_file = f"traces/puzzle_{index:03d}.jsonl"
                (run_dir / "traces").mkdir(exist_ok=True)
                decoding = {**config["decoding"], "seed": (config["decoding"]["seed"] + index) % 2**63}
                with (run_dir / trace_file).open("w") as trace_handle:
                    def save_event(event):
                        trace_handle.write(json.dumps(event, allow_nan=False) + "\n")
                        trace_handle.flush()

                    prediction = model.predict_with_stats(
                        format_prompt(item["puzzle"]), **config["generation"],
                        decoding=decoding, on_event=save_event,
                    )
                output, stats = prediction["text"], prediction["stats"]
            else:
                if config["decoding"]["method"] != "baseline":
                    raise TypeError("This decoding method requires model.predict_with_stats().")
                output = model.predict(format_prompt(item["puzzle"]), **config["generation"])
            score = score_prediction(output, item["solution"])
            score["clues_preserved"] = preserves_clues(item["puzzle"], score["prediction"])
            score["valid_sudoku"] = is_valid_sudoku(score["prediction"])
            plot_file = f"plots/puzzle_{index:03d}.png"
            record = {"index": index, **item, "raw_output": output, **score, "plot_file": plot_file,
                      "decoding_method": config["decoding"]["method"],
                      "decoding_stats": stats, "trace_file": trace_file}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            exact += score["exact"]
            correct_cells += score["correct_cells"]
            complete += score["parse_success"]
            clues_preserved += score["clues_preserved"]
            valid_sudoku += score["valid_sudoku"]
            processed = index + 1
            summary.update(
                processed=processed,
                exact_accuracy=exact / processed,
                cell_accuracy=correct_cells / (81 * processed),
                completion_rate=complete / processed,
                clue_preservation_rate=clues_preserved / processed,
                sudoku_validity_rate=valid_sudoku / processed,
            )
            if stats is not None:
                for key in ("search_model_calls", "completion_model_calls", "latency_seconds"):
                    summary[key] = (summary[key] or 0) + stats[key]
                if stats["peak_gpu_memory_bytes"] is not None:
                    summary["peak_gpu_memory_bytes"] = max(
                        summary["peak_gpu_memory_bytes"] or 0, stats["peak_gpu_memory_bytes"])
            save_json(run_dir / "metrics.json", summary)
            save_prediction_plot(record, run_dir / plot_file)
            summary["plots_saved"] += 1
            save_json(run_dir / "metrics.json", summary)
            print(
                f"{index:03d} | complete={score['parse_success']} | "
                f"clues={score['clues_preserved']} | valid={score['valid_sudoku']} | "
                f"exact={score['exact']} | output={score['prediction']}", flush=True,
            )

    summary["status"] = "completed"
    save_json(run_dir / "metrics.json", summary)
    print("Exact accuracy:", summary["exact_accuracy"], flush=True)
    print("Cell accuracy:", summary["cell_accuracy"], flush=True)
    return run_dir, summary


def main(default_config=None, expected_family=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config or ROOT / "configs/sudoku/llada.yaml")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--model-path", help="Hugging Face ID or local model directory")
    parser.add_argument("--output-dir", help="Explicit experiment parent (overrides model/dataset grouping)")
    parser.add_argument("--run-manifest", type=Path, help="Write the exact created run directory here")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dataset-only", action="store_true", help="Check the dataset without loading model weights")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.num_samples is not None:
        config["data"]["num_samples"] = args.num_samples
    if args.model_path is not None:
        config["model"]["model_path"] = args.model_path
    if args.output_dir is not None:
        config.pop("experiment_root", None)
        config["output_dir"] = args.output_dir
    if args.local_files_only:
        config["model"]["local_files_only"] = True
    validate_config(config)
    if expected_family and config["model"].get("family", "llada") != expected_family:
        parser.error(f"This entry point requires model.family: {expected_family}.")

    from dlm_project.tasks.sudoku.dataset import SudokuDataset, format_prompt
    dataset = SudokuDataset(n=config["data"]["num_samples"], cache_dir=config["data"].get("cache_dir"))
    if args.dataset_only:
        print(f"Loaded {len(dataset)} puzzle(s). First prompt:\n{format_prompt(dataset[0]['puzzle'])}")
        return
    from dlm_project.models.llada import LLaDA
    model = LLaDA(**config["model"])
    run_dir, summary = run_experiment(config, dataset, model)
    if args.run_manifest:
        args.run_manifest.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.run_manifest, {"run_dir": str(run_dir), "status": summary["status"]})


if __name__ == "__main__":
    main()
