"""Offline checks of paired 4x4 data, scoring, plots, and subprocess coordination."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from datasets import Dataset, load_dataset
import pytest
import yaml

from dlm_project.experiments.config import load_config, validate_config
from dlm_project.models.cache import stage_model
from dlm_project.models.profiles import get_profile
from dlm_project.paths import PROJECT_ROOT as ROOT
from dlm_project.tasks.sudoku import dataset as data_module
from dlm_project.tasks.sudoku import model_comparison as comparison
from dlm_project.tasks.sudoku.evaluation import is_valid_sudoku, preserves_clues, score_prediction
from dlm_project.tasks.sudoku.runner import run_experiment
from dlm_project.tasks.sudoku.visualization import plot_paired_prediction, prediction_display
from tests.workflows.test_notebook_and_cache import make_snapshot


PUZZLE = "0102200002100320"
SOLUTION = "3142243142131324"
PROMPT = f"Original 4x4 instructions.\nSolve: {PUZZLE}\n<answer>16 digits</answer>\n"


def config_for(tmp_path):
    config = load_config(ROOT / "configs/sudoku/llada_bgpo_4x4.yaml")
    config["experiment_root"] = str(tmp_path)
    config["data"]["num_samples"] = 2
    return config


class Puzzles(list):
    provenance = data_module.BGPO_PROVENANCE


def puzzles():
    return Puzzles([{"puzzle": PUZZLE, "solution": SOLUTION, "prompt": PROMPT,
                     "board_size": 4, "source_index": i, "num_blanks": PUZZLE.count("0")}
                    for i in range(2)])


class FakeModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    def predict_with_stats(self, prompt, *, decoding, on_event, **kwargs):
        self.prompts.append(prompt)
        stats = {"seed": decoding["seed"], "search_model_calls": 0,
                 "completion_model_calls": kwargs["steps"], "latency_seconds": 2.0,
                 "peak_gpu_memory_bytes": None, "model_device": "cpu"}
        text = next(self.outputs)
        on_event({"text": text, "stats": stats})
        return {"text": text, "stats": stats}


def tagged(solution=SOLUTION):
    return f"<reasoning>Consider row 1, then row 2.</reasoning>\n<answer>\n{solution}\n</answer>"


def make_runs(tmp_path):
    children = comparison.comparison_configs(config_for(tmp_path))
    left = FakeModel([tagged(), "<answer>123</answer>"])
    right = FakeModel([tagged(), tagged()])
    baseline_dir, baseline = run_experiment(children["baseline"], puzzles(), left)
    bgpo_dir, bgpo = run_experiment(children["bgpo"], puzzles(), right)
    assert left.prompts == right.prompts == [PROMPT, PROMPT]
    assert SOLUTION not in PROMPT
    return baseline_dir, bgpo_dir, baseline, bgpo


def test_parquet_adapter_preserves_strings_original_prompt_and_mapping(tmp_path, monkeypatch):
    rows = [{"prompt": [{"role": "user", "content": PROMPT}],
             "extra_info": {"index": i, "puzzle": PUZZLE},
             "reward_model": {"ground_truth": SOLUTION}} for i in range(2)]
    file = tmp_path / "test.parquet"
    Dataset.from_list(rows).to_parquet(file)

    def local_loader(kind, **kwargs):
        assert kind == "parquet"
        assert kwargs["data_files"] == {"test": data_module.BGPO_DATA_URL}
        assert data_module.BGPO_DATA_REVISION in data_module.BGPO_DATA_URL
        kwargs["data_files"] = {"test": str(file)}
        return load_dataset(kind, **kwargs)

    monkeypatch.setattr(data_module, "load_dataset", local_loader)
    dataset = data_module.BGPOSudokuDataset(2, str(tmp_path / "cache"))
    assert dataset[0] == puzzles()[0]
    assert [row["source_index"] for row in dataset] == [0, 1]
    assert data_module.item_prompt(dataset[0]) == PROMPT
    assert dataset.provenance == data_module.BGPO_PROVENANCE


@pytest.mark.parametrize("n", [0, -1, 501, True, 1.5])
def test_sample_limits_fail_before_loading(n, monkeypatch, tmp_path):
    monkeypatch.setattr(data_module, "load_dataset", lambda *a, **k: pytest.fail("Must not load data"))
    with pytest.raises(ValueError, match="500"):
        data_module.BGPOSudokuDataset(n)
    config = config_for(tmp_path)
    config["data"]["num_samples"] = n
    with pytest.raises(ValueError, match="500"):
        validate_config(config)


@pytest.mark.parametrize("text", [
    SOLUTION, "", "<answer>123</answer>", tagged(SOLUTION + "1"), tagged("5" + SOLUTION[1:]),
    tagged() + tagged(), "<answer>" + SOLUTION, "</answer>" + SOLUTION + "<answer>",
    f"<reasoning>{SOLUTION}</reasoning><answer>123</answer>",
    f"<answer>Solution: {SOLUTION}</answer>",
])
def test_only_one_complete_tagged_answer_is_scored(text):
    result = score_prediction(text, SOLUTION, 4)
    assert result["parse_success"] is False
    assert result["prediction"] == "" and result["correct_cells"] == 0


def test_4x4_scoring_checks_rows_columns_boxes_and_clues():
    assert score_prediction(tagged(" \n".join(SOLUTION)), SOLUTION, 4)["exact"]
    assert is_valid_sudoku(SOLUTION, 4)
    assert preserves_clues(PUZZLE, SOLUTION, 4)
    assert not is_valid_sudoku("1" * 16, 4)
    # Latin square satisfies rows/columns but violates the 2x2 boxes.
    assert not is_valid_sudoku("1234234134124123", 4)
    wrong_clues = SOLUTION.translate(str.maketrans("12", "21"))
    assert is_valid_sudoku(wrong_clues, 4)
    assert not preserves_clues(PUZZLE, wrong_clues, 4)


def test_same_architecture_checkpoints_cannot_replace_each_other(tmp_path):
    source, destination = tmp_path / "bgpo", tmp_path / "original"
    for directory, family in ((source, "llada_bgpo"), (destination, "llada")):
        make_snapshot(directory, family)
        profile = get_profile(family)
        (directory / ".model_source.json").write_text(json.dumps({k: profile[k] for k in ("repo_id", "revision")}))
    original = (destination / "model-1.safetensors").read_bytes()
    with pytest.raises(ValueError, match="different model/revision"):
        stage_model(source, destination, "llada_bgpo")
    assert (destination / "model-1.safetensors").read_bytes() == original


def test_paired_metrics_use_16_cells_and_save_both_outputs(tmp_path):
    left, right, baseline, bgpo = make_runs(tmp_path)
    assert baseline["cell_accuracy"] == baseline["exact_accuracy"] == 0.5
    assert bgpo["cell_accuracy"] == bgpo["exact_accuracy"] == 1.0
    assert "LLaDA/Sudoku4x4" in str(left)
    assert "LLaDA-BGPO/Sudoku4x4" in str(right)
    output = tmp_path / "comparison"
    summary = comparison.summarize_comparison(left, right, output)
    assert summary["status"] == "completed"
    assert summary["bgpo_minus_baseline"]["exact_accuracy"] == 0.5
    assert summary["models"][0]["mean_latency_seconds"] == 2
    pairs = [json.loads(line) for line in (output / "paired_predictions.jsonl").read_text().splitlines()]
    assert pairs[1]["baseline"]["raw_output"] == "<answer>123</answer>"
    assert pairs[1]["bgpo"]["raw_output"] == tagged()
    for pair in pairs:
        assert (output / pair["plot_file"]).read_bytes().startswith(b"\x89PNG")
    assert (output / "comparison.csv").is_file()


@pytest.mark.parametrize("mismatch", ["prompt", "source_index", "generation", "seed", "identity", "incomplete"])
def test_comparison_rejects_misaligned_runs_before_reporting(tmp_path, mismatch):
    left, right, _, _ = make_runs(tmp_path)
    if mismatch in ("prompt", "source_index", "seed"):
        file = right / "predictions.jsonl"
        records = [json.loads(line) for line in file.read_text().splitlines()]
        if mismatch == "seed":
            records[0]["decoding_stats"]["seed"] = 999
        else:
            records[0][mismatch] = "different"
        file.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    elif mismatch == "incomplete":
        file = right / "metrics.json"
        data = json.loads(file.read_text())
        data["status"] = "running"
        file.write_text(json.dumps(data))
    else:
        file = right / "config.json"
        data = json.loads(file.read_text())
        if mismatch == "generation":
            data["generation"]["steps"] = 128
        else:
            data["provenance"]["model"]["revision"] = "incorrect"
        file.write_text(json.dumps(data))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        comparison.summarize_comparison(left, right, output)
    assert not (output / "comparison.csv").exists()


@pytest.mark.parametrize("fail_second", [False, True])
def test_coordinator_runs_sequential_children_and_retains_completed_artifacts(tmp_path, monkeypatch, fail_second):
    monkeypatch.setattr(data_module, "load_sudoku_dataset", lambda config: puzzles())
    commands = []
    real_run = subprocess.run

    def fake_run(command, **kwargs):
        # run_experiment's Git provenance read also uses subprocess.run.
        if command[0] == "git":
            return real_run(command, **kwargs)
        commands.append(command)
        assert kwargs["check"] is True
        if len(commands) == 2 and fail_second:
            raise subprocess.CalledProcessError(1, command)
        child = load_config(command[command.index("--config") + 1])
        run_dir, metrics = run_experiment(child, puzzles(), FakeModel([tagged(), tagged()]))
        receipt = Path(command[command.index("--run-manifest") + 1])
        receipt.write_text(json.dumps({"run_dir": str(run_dir), "status": metrics["status"]}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(comparison.subprocess, "run", fake_run)
    receipt = tmp_path / "receipt.json"
    config = config_for(tmp_path)
    if fail_second:
        with pytest.raises(subprocess.CalledProcessError):
            comparison.run_comparison(config, receipt)
    else:
        comparison.run_comparison(config, receipt)
    assert len(commands) == 2
    assert commands[0][0] == commands[1][0] == sys.executable
    assert commands[0][3] == commands[1][3] == "--config"
    manifest = json.loads(receipt.read_text())
    state = json.loads((Path(manifest["run_dir"]) / "comparison.json").read_text())
    assert state["status"] == manifest["status"] == ("failed" if fail_second else "completed")
    assert (Path(state["model_runs"]["baseline"]) / "predictions.jsonl").is_file()


def test_four_panel_plot_and_partial_answer_rows():
    import matplotlib.pyplot as plt
    complete = {**puzzles()[0], "raw_output": tagged()}
    incomplete = {**puzzles()[0], "raw_output": "<answer>12\n34</answer>"}
    display = prediction_display(incomplete)
    assert display["rows"] == [["1", "2"], ["3", "4"]]
    assert "4/16 digits" in display["title"]
    figure = plot_paired_prediction(complete, incomplete)
    try:
        assert len(figure.axes) == 4
        assert figure.axes[0].get_xlim() == (0, 4)
        assert len(figure.axes[2].texts) == 16
        assert len(figure.axes[3].texts) == 4
    finally:
        plt.close(figure)


def test_notebook_configures_both_models_and_shared_defaults(tmp_path):
    notebook = json.loads((ROOT / "notebooks/sudoku/LLaDA_BGPO_Sudoku_Colab.ipynb").read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    scope = {
        "Path": Path, "REPO_DIR": ROOT, "MODEL_PATHS": {"baseline": tmp_path / "base", "bgpo": tmp_path / "bgpo"},
        "os": SimpleNamespace(environ={"HF_DATASETS_CACHE": str(tmp_path / "cache")}),
        "DRIVE_ROOT": tmp_path, "EXPERIMENT_TIMEZONE": "America/Los_Angeles",
        "PROFILES": {"baseline": get_profile("llada"), "bgpo": get_profile("llada_bgpo")},
    }
    exec(cells["base-config"], scope)
    scope["RUN_CONFIG"] = tmp_path / "run.yaml"
    commands = []
    from dlm_project.experiments import paths
    scope.update(sys=sys, experiment_paths=paths, subprocess=SimpleNamespace(run=lambda args, **kw: commands.append(args)))
    exec(cells["config-updater"], scope)
    config = load_config(scope["RUN_CONFIG"])
    assert config["data"]["num_samples"] == 3
    assert config["generation"]["block_length"] == 32
    assert config["decoding"] == {"method": "baseline", "seed": 1}
    assert config["model"]["model_path"] == str(tmp_path / "bgpo")
    assert config["comparison"]["baseline_model"]["model_path"] == str(tmp_path / "base")
    assert "--dataset-only" in commands[0]
