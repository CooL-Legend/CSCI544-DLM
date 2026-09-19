import csv
import json
from pathlib import Path

from datasets import load_dataset
import pytest

import sudoku_dataset
from run_sudoku import load_config, run_experiment, score_prediction, validate_config

ROOT = Path(__file__).resolve().parents[1]
PUZZLE = "009083400005179802823005700107000605200560179604900003971030504000400917000001300"
SOLUTION = "719283456465179832823645791197328645238564179654917283971832564382456917546791328"


def test_csv_preserves_81_digit_strings_and_leading_zeros(tmp_path, monkeypatch):
    path = tmp_path / "puzzles.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["puzzle", "solution"])
        writer.writerows([(PUZZLE, SOLUTION), ("0" * 81, SOLUTION)])

    def local_loader(repo_id, **kwargs):
        return load_dataset("csv", data_files=str(path), **kwargs)

    monkeypatch.setattr(sudoku_dataset, "load_dataset", local_loader)
    dataset = sudoku_dataset.SudokuDataset(n=2, cache_dir=str(tmp_path / "cache"))
    assert dataset[0]["puzzle"] == PUZZLE
    assert dataset[0]["solution"] == SOLUTION
    assert dataset[1]["puzzle"] == "0" * 81
    assert dataset[1]["num_blanks"] == 81
    assert ". . 9 . 8 3 4 . ." in sudoku_dataset.format_prompt(PUZZLE)


def test_scoring_retains_original_behavior():
    assert score_prediction(SOLUTION, SOLUTION)["exact"] is True
    assert score_prediction("123", SOLUTION)["correct_cells"] == 0
    assert score_prediction(" ".join(SOLUTION) + "123", SOLUTION)["prediction"] == SOLUTION


@pytest.mark.parametrize("key,value", [("steps", 3), ("steps", 129), ("gen_length", 127), ("block_length", 0)])
def test_invalid_sampling_settings_fail_early(key, value):
    config = load_config(ROOT / "configs/sudoku.yaml")
    config["generation"][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_partial_run_keeps_completed_results(tmp_path):
    config = load_config(ROOT / "configs/sudoku.yaml")
    config["output_dir"] = str(tmp_path)
    dataset = [{"puzzle": PUZZLE, "solution": SOLUTION, "num_blanks": PUZZLE.count("0")}] * 2

    class InterruptedModel:
        calls = 0

        def predict(self, prompt, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("Simulated runtime interruption")
            return SOLUTION

    with pytest.raises(RuntimeError, match="interruption"):
        run_experiment(config, dataset, InterruptedModel())
    run_dir = next(tmp_path.iterdir())
    records = (run_dir / "predictions.jsonl").read_text().splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["raw_output"] == SOLUTION
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["processed"] == 1
    assert metrics["status"] == "running"
    assert metrics["exact_accuracy"] == 1.0


def test_completed_run_counts_invalid_outputs(tmp_path):
    config = load_config(ROOT / "configs/sudoku.yaml")
    config["output_dir"] = str(tmp_path)
    dataset = [{"puzzle": PUZZLE, "solution": SOLUTION}] * 2

    class Model:
        outputs = iter([SOLUTION, "No solution"])

        def predict(self, prompt, **kwargs):
            return next(self.outputs)

    _, summary = run_experiment(config, dataset, Model())
    assert summary["status"] == "completed"
    assert summary["exact_accuracy"] == summary["cell_accuracy"] == 0.5
