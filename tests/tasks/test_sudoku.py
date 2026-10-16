import csv
import copy
import json
from pathlib import Path

from datasets import load_dataset
import pytest

from dlm_project.tasks.sudoku import dataset as sudoku_dataset
from dlm_project.tasks.sudoku.comparison import run_comparison
from dlm_project.experiments.config import load_config, validate_config
from dlm_project.tasks.sudoku.evaluation import is_valid_sudoku, preserves_clues, score_prediction
from dlm_project.tasks.sudoku.runner import run_experiment
from dlm_project.tasks.sudoku.visualization import plot_prediction, prediction_display

from dlm_project.paths import PROJECT_ROOT as ROOT
PUZZLE = "009083400005179802823005700107000605200560179604900003971030504000400917000001300"
SOLUTION = "719283456465179832823645791197328645238564179654917283971832564382456917546791328"
INCOMPLETE_OUTPUT = """5 7 9 8 3 4 6 1
3 5 1 7 9 8 4 2
8 2 3 4 5 7 9 6
1 4 7 9 2 6 3 5
2 8 5 6 4 1 7 9
6 1 4 9 7 8 2 3
9 7 1 3 6 5 8 4
4 6 2 4 8 9 1 7"""


def prediction_record(output):
    return {"puzzle": PUZZLE, "solution": SOLUTION, "raw_output": output}


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


def test_offline_dataset_reuses_stored_config_and_schema(tmp_path, monkeypatch):
    class CachedDataset:
        column_names = ["puzzle", "solution"]

        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

        def select(self, indices):
            return CachedDataset([self.rows[index] for index in indices])

    monkeypatch.setattr(sudoku_dataset.datasets_config, "HF_DATASETS_OFFLINE", True)
    monkeypatch.setattr(sudoku_dataset.HFDataset, "from_file", staticmethod(
        lambda path: CachedDataset([{"puzzle": PUZZLE, "solution": SOLUTION}])
    ))
    monkeypatch.setattr(sudoku_dataset, "load_dataset", lambda *args, **kwargs: pytest.fail(
        "Offline loading must not resolve the Hub dataset."
    ))
    cache = tmp_path / "prepared-cache"
    normalized = sudoku_dataset.camelcase_to_snakecase("Sudoku-1M-Puzzles")
    prepared = cache / f"certen___{normalized}" / "default-cache" / "0.0.0" / "hash"
    prepared.mkdir(parents=True)
    (prepared / "dataset_info.json").write_text("{}")
    (prepared / "data-train.arrow").touch()

    dataset = sudoku_dataset.SudokuDataset(n=1, cache_dir=cache)

    assert dataset[0]["puzzle"] == PUZZLE


def test_scoring_requires_a_complete_answer_and_does_not_truncate():
    assert score_prediction(SOLUTION, SOLUTION)["exact"] is True
    assert score_prediction("123", SOLUTION)["correct_cells"] == 0
    assert score_prediction(" ".join(SOLUTION), SOLUTION)["prediction"] == SOLUTION
    assert score_prediction("Answer: " + SOLUTION, SOLUTION)["prediction"] == SOLUTION
    assert score_prediction(" ".join(SOLUTION) + "123", SOLUTION)["prediction"] == ""
    assert score_prediction(SOLUTION + "123", SOLUTION)["prediction"] == ""


def test_supplied_incomplete_response_has_zero_accuracy_and_keeps_rows():
    score = score_prediction(INCOMPLETE_OUTPUT, SOLUTION)
    assert score["prediction"] == ""
    assert score["parse_success"] is score["exact"] is False
    assert score["correct_cells"] == 0
    assert score["generated_digit_count"] == 64
    assert not preserves_clues(PUZZLE, score["prediction"])
    assert not is_valid_sudoku(score["prediction"])
    display = prediction_display(prediction_record(INCOMPLETE_OUTPUT))
    assert display["title"] == "Raw response\nIncomplete: 64/81 digits"
    assert display["parsed"] is False
    assert display["rows"] == [line.split() for line in INCOMPLETE_OUTPUT.splitlines()]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plot_prediction(prediction_record(INCOMPLETE_OUTPUT))
    try:
        labels = {text.get_position(): text.get_text() for text in figure.axes[2].texts}
        assert len(labels) == 64
        assert labels[(0.5, 0.5)] == "5"
        assert labels[(7.5, 0.5)] == "1"
        assert labels[(0.5, 1.5)] == "3"  # First digit of row two stays in row two.
        assert all(x < 8 and y < 8 for x, y in labels)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("output,clues,valid,issue", [
    (SOLUTION, True, True, None),
    ("1" + SOLUTION[1:], True, False, "Sudoku rules"),
    (SOLUTION.translate(str.maketrans("12", "21")), False, True, "clue conflict"),
    (SOLUTION[:2] + "1" + SOLUTION[3:], False, False, "clue conflict, Sudoku rules"),
])
def test_complete_grids_are_validated_without_repair(output, clues, valid, issue):
    score = score_prediction(output, SOLUTION)
    assert score["prediction"] == output
    assert score["parse_success"] is True
    assert preserves_clues(PUZZLE, output) is clues
    assert is_valid_sudoku(output) is valid
    display = prediction_display(prediction_record(output))
    assert "".join("".join(row) for row in display["rows"]) == output
    if issue:
        assert f"Invalid: {issue}" in display["title"]
        assert display["color"] == "#b91c1c"
    else:
        assert "Valid; clues preserved" in display["title"]


@pytest.mark.parametrize("output", ["", "No solution", "x" * 81, "0" * 81])
def test_malformed_outputs_do_not_pass_validation(output):
    assert not preserves_clues(PUZZLE, output)
    assert not is_valid_sudoku(output)


@pytest.mark.parametrize("output", ["1234567891\n123", "123\n" * 10, "Answer: 123"])
def test_ambiguous_raw_layout_is_text_instead_of_a_reshaped_grid(output):
    assert prediction_display(prediction_record(output))["rows"] is None


@pytest.mark.parametrize("key,value", [("steps", 3), ("steps", 129), ("gen_length", 127), ("block_length", 0)])
def test_invalid_sampling_settings_fail_early(key, value):
    config = load_config(ROOT / "configs/sudoku/llada.yaml")
    config["generation"][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_partial_run_keeps_completed_results(tmp_path):
    config = load_config(ROOT / "configs/sudoku/llada.yaml")
    config["experiment_root"] = str(tmp_path)
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
    run_dir = next(tmp_path.rglob("metrics.json")).parent
    records = (run_dir / "predictions.jsonl").read_text().splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["raw_output"] == SOLUTION
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["processed"] == 1
    assert metrics["status"] == "failed"
    assert metrics["exact_accuracy"] == 1.0
    assert metrics["plots_saved"] == 1
    assert (run_dir / "plots/puzzle_000.png").is_file()
    assert metrics["completion_rate"] == metrics["clue_preservation_rate"] == metrics["sudoku_validity_rate"] == 1.0


def test_completed_run_counts_invalid_outputs(tmp_path):
    config = load_config(ROOT / "configs/sudoku/llada.yaml")
    config["experiment_root"] = str(tmp_path)
    dataset = [{"puzzle": PUZZLE, "solution": SOLUTION}] * 2

    class Model:
        outputs = iter([SOLUTION, "No solution"])

        def predict(self, prompt, **kwargs):
            return next(self.outputs)

    _, summary = run_experiment(config, dataset, Model())
    assert summary["status"] == "completed"
    assert summary["exact_accuracy"] == summary["cell_accuracy"] == 0.5
    assert summary["completion_rate"] == summary["clue_preservation_rate"] == summary["sudoku_validity_rate"] == 0.5


def test_aggregate_metrics_distinguish_completion_clues_and_validity(tmp_path):
    config = load_config(ROOT / "configs/sudoku/llada.yaml")
    config["experiment_root"] = str(tmp_path)
    dataset = [{"puzzle": PUZZLE, "solution": SOLUTION}] * 4
    outputs = [SOLUTION, "1" + SOLUTION[1:], SOLUTION.translate(str.maketrans("12", "21")), INCOMPLETE_OUTPUT]

    class Model:
        responses = iter(outputs)

        def predict(self, prompt, **kwargs):
            return next(self.responses)

    run_dir, metrics = run_experiment(config, dataset, Model())
    assert metrics["processed"] == 4
    assert metrics["completion_rate"] == 0.75
    assert metrics["clue_preservation_rate"] == 0.5
    assert metrics["sudoku_validity_rate"] == 0.5
    assert metrics["exact_accuracy"] == 0.25
    assert metrics["cell_accuracy"] == pytest.approx((81 + 80 + 63) / (81 * 4))
    assert json.loads((run_dir / "metrics.json").read_text()) == metrics
    records = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    assert [record["raw_output"] for record in records] == outputs
    assert [record["prediction"] for record in records] == outputs[:3] + [""]
    assert [record["valid_sudoku"] for record in records] == [True, False, True, False]
    assert [record["clues_preserved"] for record in records] == [True, True, False, False]
    assert metrics["plots_saved"] == 4
    for record in records:
        assert (run_dir / record["plot_file"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    saved = json.loads((run_dir / "config.json").read_text())
    assert saved["run"]["timezone"] == "America/Los_Angeles"
    assert saved["provenance"]["git_commit"]


def test_comparison_reuses_first_three_puzzles_and_only_changes_length(tmp_path):
    config = load_config(ROOT / "configs/sudoku/llada.yaml")
    config["data"]["num_samples"] = 10
    config["generation"].update(steps=64, block_length=16, temperature=0.3, remasking="random")
    original_config = copy.deepcopy(config)
    dataset = [{"puzzle": puzzle, "solution": SOLUTION} for puzzle in (PUZZLE, SOLUTION, "0" * 81, PUZZLE)]

    class Model:
        calls = []

        def predict(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            return INCOMPLETE_OUTPUT if kwargs["gen_length"] == 128 else SOLUTION

    model = Model()
    output_dir = tmp_path / "comparison"
    report = run_comparison(config, dataset, model, output_dir)
    assert config == original_config
    assert len(model.calls) == 6
    assert [prompt for prompt, _ in model.calls[:3]] == [sudoku_dataset.format_prompt(item["puzzle"]) for item in dataset[:3]]
    assert [prompt for prompt, _ in model.calls[3:]] == [prompt for prompt, _ in model.calls[:3]]
    for index, (_, settings) in enumerate(model.calls):
        assert settings == {**original_config["generation"], "steps": 128, "block_length": 32, "gen_length": 128 if index < 3 else 256}
    assert report["status"] == "completed"
    assert [run["completion_rate"] for run in report["runs"]] == [0.0, 1.0]
    assert [run["exact_accuracy"] for run in report["runs"]] == [0.0, 1.0]
    assert json.loads((output_dir / "comparison.json").read_text()) == report
    assert json.loads((output_dir / "puzzles.json").read_text()) == dataset[:3]
    with (output_dir / "comparison.csv").open() as handle:
        assert [int(row["gen_length"]) for row in csv.DictReader(handle)] == [128, 256]
    for run in report["runs"]:
        run_dir = Path(run["run_dir"])
        saved_config = json.loads((run_dir / "config.json").read_text())
        assert saved_config["data"]["num_samples"] == 3
        assert saved_config["generation"]["gen_length"] == run["gen_length"]
        records = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
        assert [record["puzzle"] for record in records] == [item["puzzle"] for item in dataset[:3]]
