"""Offline sweep acceptance: real decoder/wrapper, controlled model and data."""

import copy
import csv
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from dlm_project.experiments import manifests, sweep, sweep_report, sweep_plots
from dlm_project.experiments.config import load_config
from dlm_project.models.llada import LLaDA
from dlm_project.models.profiles import get_profile
from dlm_project.paths import PROJECT_ROOT as ROOT
from tests.decoding.helpers import FakeModel, FakeTokenizer, MASK, PROMPT

REVISION = "a" * 40
SOLUTION = "1234341221434321"


@pytest.fixture
def sources(monkeypatch):
    """Actual dataset interfaces with distinctive references, no network calls."""
    import datasets
    import huggingface_hub
    from dlm_project.tasks.sudoku import dataset as sudoku
    calls = []
    class Rows(list):
        _fingerprint = "controlled-dataset"
    def load_dataset(repo, subset, *, split, revision, cache_dir):
        calls.append((repo, subset, split, revision))
        if split == "train[:2]":
            return Rows({"question": f"Independent training demo {i}", "answer": "1+1=2\n#### 2"} for i in range(2))
        return Rows({"question": f"{split} problem {index}: how many apples?",
                     "answer": f"REFERENCE_ONLY_{index}\n#### {index}"} for index in range(220))
    def dataset_info(repo):
        calls.append(("resolve_revision", repo))
        return SimpleNamespace(sha=REVISION)
    def bgpo(n, cache_dir):
        calls.append(("bgpo", n))
        rows = []
        for index in range(500):
            puzzle = "".join(value if index & (1 << cell) else "0" for cell, value in enumerate(SOLUTION))
            rows.append({"source_index": index, "puzzle": puzzle, "solution": SOLUTION,
                         "prompt": f"Fill this 4x4 board: {puzzle}. Use <answer> tags."})
        return rows
    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(dataset_info=dataset_info))
    monkeypatch.setattr(sudoku, "BGPOSudokuDataset", bgpo)
    return calls


def config_for(task, tmp_path, count=2):
    config = load_config(ROOT / f"configs/{task}/llada_sweep.yaml")
    config["experiment_root"] = str(tmp_path)
    config["data"]["num_samples"] = count
    config["model"].update(device="cpu", dtype="float32")
    config["generation"].update(gen_length=4, steps=4, block_length=4)
    config["decoding"]["mcts"].update(init_tokens=2, top_k_tokens=2, top_k_actions=2)
    config["sweep"]["budgets"] = [2, 5, 8]
    return config


def controlled_wrapper(output="#### 1"):
    instance = LLaDA.__new__(LLaDA)
    instance.device, instance.mask_id, instance.family = "cpu", MASK, "llada"
    instance.model, instance.tokenizer = FakeModel(), FakeTokenizer()
    instance.tokenizer.batch_decode = lambda *args, **kwargs: [output]
    instance.identity = {**get_profile("llada"), "controlled_test_model": True}
    return instance


def read_rows(directory):
    return [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("task", ["gsm8k", "sudoku"])
def test_manifest_freezes_revision_ids_order_content_and_disjoint_pools(tmp_path, sources, monkeypatch, task):
    path = tmp_path / "manifest.json"
    first = manifests.get_manifest(path, task)
    repeated = manifests.create_manifest(task)
    assert first == repeated
    assert first != manifests.create_manifest(task, sample_seed=2)
    assert [row["position"] for row in first["examples"]] == list(range(len(first["examples"])))
    dev = manifests.select_examples(first, "smoke", 3)
    evaluation = manifests.select_examples(first, "milestone", 30)
    assert not {row["example_id"] for row in dev} & {row["example_id"] for row in evaluation}
    if task == "gsm8k":
        assert first["dataset"]["revision"] == REVISION
        assert ("openai/gsm8k", "main", "train", REVISION) in sources
        assert ("openai/gsm8k", "main", "test", REVISION) in sources
    before = path.read_text()
    monkeypatch.setattr(manifests, "create_manifest", lambda *a, **kw: pytest.fail("Must reuse saved manifest offline"))
    assert manifests.get_manifest(path, task) == first
    assert path.read_text() == before
    with pytest.raises(ValueError, match="revision"):
        manifests.get_manifest(path, task, revision="b" * 40)
    changed = copy.deepcopy(first)
    changed["examples"][0]["content_hash"] = "changed"
    with pytest.raises(ValueError, match="hash"):
        manifests.validate_manifest(changed, task)
    changed = copy.deepcopy(first)
    changed["examples"][1] = {**changed["examples"][0], "position": 1}
    with pytest.raises(ValueError, match="Duplicate"):
        manifests.validate_manifest(changed, task)


@pytest.mark.parametrize("task,milestone", [("gsm8k", 50), ("sudoku", 30)])
def test_profiles_defaults_and_cli_overrides(task, milestone):
    config = load_config(ROOT / f"configs/{task}/llada_sweep.yaml")
    original = copy.deepcopy(config)
    result = sweep.resolve_settings(task, config)
    assert result["data"]["num_samples"] == 3
    assert [c["id"] for c in result["conditions"]] == ["baseline", "confidence", "mcts_128", "mcts_256", "mcts_512"]
    assert all(c["decoding"]["mcts"]["stop_mode"] == "budget" for c in result["conditions"])
    assert sweep.resolve_settings(task, config, profile="milestone")["data"]["num_samples"] == milestone
    assert sweep.resolve_settings(task, config, profile="milestone", num_samples=1, budgets=[17])["conditions"][-1]["search_budget"] == 17
    assert original == config
    for invalid in ([], [0], [2, 2], [True]):
        with pytest.raises(ValueError):
            sweep.resolve_settings(task, config, budgets=invalid)
    config["decoding"]["mcts"]["stop_mode"] = "candidates"
    with pytest.raises(ValueError, match="budget"):
        sweep.resolve_settings(task, config)


@pytest.mark.parametrize("task", ["gsm8k", "sudoku"])
def test_one_model_five_conditions_fresh_states_forward_accounting_and_incremental_saves(tmp_path, sources, monkeypatch, task):
    monkeypatch.setattr(sweep_plots, "save_plots", lambda *a: None)
    instance = controlled_wrapper()
    factory_calls, predictions, offsets, persisted = [], [], [], []
    original = instance.predict_with_stats
    def predict(prompt, **kwargs):
        # Every previous generation has been written before the next starts.
        files = list(tmp_path.rglob("predictions.jsonl"))
        persisted.append(len(files[0].read_text().splitlines()))
        predictions.append((prompt, copy.deepcopy(kwargs["decoding"])))
        offsets.append(len(instance.model.seen))
        return original(prompt, **kwargs)
    instance.predict_with_stats = predict
    def factory(config):
        factory_calls.append(config)
        return instance
    directory, summary = sweep.run_sweep(task, config_for(task, tmp_path), model_factory=factory)
    rows = read_rows(directory)
    assert len(factory_calls) == 1
    assert len(rows) == 10 and summary["common_n"] == 2 and summary["complete"]
    assert persisted == list(range(10))
    assert summary["setup"]["warmup_model_calls"] == 1
    assert summary["setup"]["model_load_count"] == 1
    assert sum(row["total_nfes"] for row in rows) + 1 == len(instance.model.seen)
    for start in (0, 5):
        group = rows[start:start+5]
        assert len({(row["example_id"], row["base_prompt_hash"], row["seed"]) for row in group}) == 1
        assert [row["search_nfes"] for row in group] == [0, 0, 2, 5, 7]
        assert [row["completion_nfes"] for row in group] == [4, 4, 3, 2, 2]
        assert [row["fallback_used"] for row in group] == [None, None, True, True, False]
        assert group[-1]["stop_reason"] == "frontier_exhausted"
        assert group[-1]["evaluated_children"] == 6
        assert group[-1]["expanded_parents"] == 3
        assert group[-1]["realized_branching_factor"] == 2
        assert all(row['candidate_evaluation_nfes']==0 for row in group)
    for index, ((prompt, decoding), offset) in enumerate(zip(predictions, offsets)):
        assert "REFERENCE_ONLY" not in prompt and SOLUTION not in prompt
        assert set(decoding) == {"method", "seed", "mcts"}
        first = instance.model.seen[offset]
        assert torch.equal(first[:, :2], PROMPT)
        assert (first[:, 2:] == MASK).all()  # No earlier condition's commitments.
        assert decoding["seed"] == 1 + rows[index]["manifest_position"]
    for row in rows:
        events = [json.loads(line) for line in (directory / row["trace_file"]).read_text().splitlines()]
        assert "logits" not in (directory / row["trace_file"]).read_text()
        assert events[-1]["stats"]["total_model_calls"] == row["total_nfes"]
    assert list(csv.DictReader((directory / "results.csv").open()))[0]["condition"] == "baseline"
    assert all(row["run_status"] == "completed" for row in csv.DictReader((directory / "summary.csv").open()))
    assert "paired" in (directory / "report.md").read_text()


@pytest.mark.parametrize("task", ["gsm8k", "sudoku"])
def test_failed_generation_retains_completed_records_and_uses_common_denominator(tmp_path, sources, monkeypatch, task):
    monkeypatch.setattr(sweep_plots, "save_plots", lambda *a: None)
    instance = controlled_wrapper()
    original = instance.predict_with_stats
    count = 0
    def predict(*a, **kw):
        nonlocal count
        count += 1
        if count == 8:
            raise RuntimeError("controlled failure")
        return original(*a, **kw)
    instance.predict_with_stats = predict
    with pytest.raises(RuntimeError, match="controlled failure"):
        sweep.run_sweep(task, config_for(task, tmp_path), model_factory=lambda _: instance)
    directory = next(tmp_path.rglob("run_state.json")).parent
    rows = read_rows(directory)
    summary = json.loads((directory / "summary.json").read_text())
    assert len(rows) == 7
    assert summary["status"] == "failed" and not summary["complete"]
    assert summary["common_n"] == 1
    assert all(item["n"] == 1 for item in summary["conditions"])
    assert [item["available_n"] for item in summary["conditions"]] == [2, 2, 1, 1, 1]
    assert "Incomplete run" in (directory / "report.md").read_text()
    data = sweep_plots.comparison_data(rows, summary)
    assert len(data["correctness"]) == 1
    assert data["conditions"] == summary["conditions"]


@pytest.mark.parametrize("task,outputs", [
    ("gsm8k", ["Reasoning\n#### 12", "#### 13", "12", ""]),
    ("sudoku", [f"<answer>{SOLUTION}</answer>", "<answer>" + "1" * 16 + "</answer>", "<answer>12</answer>", ""]),
])
def test_scorers_correct_incorrect_malformed_empty(task, outputs):
    example = {"reference_solution": "reference\n#### 12", "solution": SOLUTION, "puzzle": SOLUTION[:4] + "0" * 12}
    scores = [sweep.score_output(task, text, example) for text in outputs]
    assert [item["exact"] for item in scores] == [True, False, False, False]
    assert [item["parse_success"] for item in scores] == [True, True, False, False]
    if task == "sudoku":
        assert scores[0]["parsed_grid"] == [list(SOLUTION[row:row+4]) for row in range(0, 16, 4)]
        assert scores[2]["parsed_grid"] is None
        assert scores[0]["cell_accuracy"] == 1
        assert scores[1]["cell_accuracy"] == .25
        assert scores[2]["cell_accuracy"] == 0
        assert [item["clues_preserved"] for item in scores] == [True, False, False, False]
        assert [item["valid_sudoku"] for item in scores] == [True, False, False, False]


def test_wilson_extremes_latency_interpolation_and_empty_stats():
    for correct in (0, 5):
        lo, hi = sweep_report.wilson_interval(correct, 5)
        assert 0 <= lo < hi <= 1
        assert lo == pytest.approx(0 if correct == 0 else .5655175352)
        assert hi == pytest.approx(.4344824648 if correct == 0 else 1)
    assert sweep_report.wilson_interval(0, 0) == (None, None)
    assert sweep_report.percentile([1, 2, 3], .95) == pytest.approx(2.9)
    empty = sweep_report.aggregate([], {"id": "baseline", "method": "baseline", "search_budget": None})
    assert empty["n"] == 0 and empty["accuracy"] is None and empty["mean_total_nfes"] is None


@pytest.mark.parametrize("task", ["gsm8k", "sudoku"])
def test_full_figures_trace_data_and_report_only_without_loads(tmp_path, sources, monkeypatch, task):
    instance = controlled_wrapper(output="<answer>1 2\n3 4</answer>" if task == "sudoku" else "#### 1")
    config = config_for(task, tmp_path, count=1)
    # Exercise full reporting, including partial-output figures and trace plots.
    directory, summary = sweep.run_sweep(task, config, model_factory=lambda _: instance)
    data = json.loads((directory / "plot_data.json").read_text())
    assert data["conditions"] == summary["conditions"]
    assert data["correctness"] == [[int(row["exact"]) for row in read_rows(directory)]]
    assert len(list((directory / "plots").glob("*.png"))) == 5
    assert len(list((directory / "plots").glob("*.pdf"))) == 5
    assert len(list((directory / "inspection").glob("*.png"))) == 3
    rows = read_rows(directory)
    for row in rows[2:]:
        trace = [json.loads(line) for line in (directory / row["trace_file"]).read_text().splitlines()]
        inspected = sweep_plots.trace_data(trace, row["selected_node_id"])
        assert len(inspected["selected_path"]) == row["selected_depth"] + 1
        assert len(inspected["evaluations"]) == row["evaluated_children"]
        assert [item["search_model_calls"] for item in inspected["evaluations"]] == [
            e["search_model_calls"] for e in trace if e["stage"] == "evaluation"]
        assert sum(e["stage"] == "forward" and e.get("phase") != "completion" for e in trace) == row["search_nfes"]
    if task == "sudoku":
        assert len(list((directory / "examples").glob("*.png"))) == 1
    monkeypatch.setattr(sweep, "_load_model", lambda *a: pytest.fail("Report loaded weights"))
    monkeypatch.setattr(sweep, "get_manifest", lambda *a, **kw: pytest.fail("Report loaded dataset"))
    monkeypatch.setattr(sweep_plots, "save_plots", lambda *a: None)  # Already verified above.
    monkeypatch.setattr(sys, "argv", ["script", "--report-only", str(directory)])
    sweep.main(task)
    assert json.loads((directory / "summary.json").read_text()) == summary
    # Also exercise the installed CLI in a fresh offline process outside the
    # checkout. Loading CUDA weights here would fail on the CPU test runtime.
    completed = subprocess.run([sys.executable, "-I", str(ROOT / f"scripts/run_{task}_sweep.py"),
                                "--report-only", str(directory)], cwd=tmp_path,
                               env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert "Saved report" in completed.stdout


def test_model_load_failure_produces_empty_report_and_never_fabricates_predictions(tmp_path, sources, monkeypatch):
    monkeypatch.setattr(sweep_plots, "save_plots", lambda *a: None)
    def fail(_):
        raise RuntimeError("no device")
    with pytest.raises(RuntimeError, match="no device"):
        sweep.run_sweep("gsm8k", config_for("gsm8k", tmp_path), model_factory=fail)
    directory = next(tmp_path.rglob("run_state.json")).parent
    summary = json.loads((directory / "summary.json").read_text())
    assert read_rows(directory) == []
    assert summary["common_n"] == summary["completed_predictions"] == 0
    assert summary["status"] == "failed"
    assert all(item["accuracy"] is None for item in summary["conditions"])
    with pytest.raises(ValueError, match="pinned"):
        sweep._verify_identity(SimpleNamespace(identity={"repo_id": "another-model", "revision": REVISION}))


def test_report_recovery_and_rejection_of_inconsistent_rows(tmp_path, sources, monkeypatch):
    monkeypatch.setattr(sweep_plots, "save_plots", lambda *a: None)
    directory, _ = sweep.run_sweep("gsm8k", config_for("gsm8k", tmp_path, count=1),
                                   model_factory=lambda _: controlled_wrapper())
    predictions = directory / "predictions.jsonl"
    original = predictions.read_text()
    predictions.write_text(original + '{"example_id":')  # Abrupt stop during write.
    recovered = sweep_report.generate_report(directory)
    assert recovered["common_n"] == 1 and not recovered["complete"]
    assert recovered["warnings"] and "Recovery notice" in (directory / "report.md").read_text()
    assert predictions.read_text() == original + '{"example_id":'
    rows = [json.loads(line) for line in original.splitlines()]
    for mutate in (lambda values: values.append(values[0]),
                   lambda values: values[0].update(seed=999),
                   lambda values: values[0].update(total_nfes=999)):
        changed = copy.deepcopy(rows)
        mutate(changed)
        predictions.write_text("".join(json.dumps(row) + "\n" for row in changed))
        with pytest.raises(ValueError):
            sweep_report.generate_report(directory, plots=False)
    predictions.write_text(original)


def test_plot_failure_retains_generations_and_report_only_finishes_artifacts(tmp_path, sources, monkeypatch):
    def fail(*a):
        raise OSError("controlled plot failure")
    monkeypatch.setattr(sweep_plots, "save_plots", fail)
    with pytest.raises(OSError, match="plot failure"):
        sweep.run_sweep("gsm8k", config_for("gsm8k", tmp_path, count=1),
                        model_factory=lambda _: controlled_wrapper())
    directory = next(tmp_path.rglob("run_state.json")).parent
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["status"] == "report_failed" and not summary["complete"]
    assert summary["common_n"] == 1 and len(read_rows(directory)) == 5
    monkeypatch.setattr(sweep_plots, "save_plots", lambda *a: None)
    summary = sweep_report.generate_report(directory)
    assert summary["complete"] and summary["error"] is None


def test_yaml_output_parent_and_requested_sudoku_revision(tmp_path, sources, monkeypatch):
    with pytest.raises(ValueError, match="revision"):
        manifests.get_manifest(tmp_path / "new-manifest.json", "sudoku", revision="b" * 40)
    assert ("bgpo", 500) not in sources  # Reject before downloading the dataset.
    monkeypatch.setattr(sweep_plots, "save_plots", lambda *a: None)
    config = config_for("sudoku", tmp_path, count=1)
    config["output_dir"] = str(tmp_path / "custom-output")
    directory, _ = sweep.run_sweep("sudoku", config, model_factory=lambda _: controlled_wrapper())
    assert directory.parent == tmp_path / "custom-output"


@pytest.mark.parametrize("correct,count", [(False, 3), (True, 3), (False, 0)])
def test_comparison_plot_values_for_all_wrong_all_correct_and_empty_samples(tmp_path, correct, count):
    conditions = [{"id": name, "method": method, "search_budget": budget} for name, method, budget in
                  [("baseline", "baseline", None), ("confidence", "confidence", None),
                   ("mcts_128", "mcts_entropy", 128), ("mcts_256", "mcts_entropy", 256),
                   ("mcts_512", "mcts_entropy", 512)]]
    rows = [{"condition": c["id"], "example_id": f"test:{i}", "exact": correct, "parse_success": correct,
             "search_nfes": c["search_budget"] or 0,
             "completion_nfes": 235 if c["search_budget"] else 256,
             "total_nfes": (c["search_budget"] or 0) + (235 if c["search_budget"] else 256),
             "latency_seconds": 1 + i, "peak_gpu_memory_bytes": None,
             "selected_depth": 20 if c["search_budget"] else None,
             "evaluated_children": c["search_budget"] - 1 if c["search_budget"] else None,
             "realized_branching_factor": 5 if c["search_budget"] else None,
             "fallback_used": False if c["search_budget"] else None}
            for c in conditions for i in range(count)]
    summary = {"common_example_ids": [f"test:{i}" for i in range(count)],
               "conditions": [sweep_report.aggregate([row for row in rows if row["condition"] == c["id"]], c)
                              for c in conditions]}
    data = sweep_plots.comparison_data(rows, summary)
    assert data["correctness"] == [[int(correct)] * 5 for _ in range(count)]
    if count:
        assert [item["mean_total_nfes"] for item in data["conditions"]] == [256, 256, 363, 491, 747]
        assert all(item["accuracy"] == float(correct) for item in data["conditions"])
    # Rendering boundary accuracies must not generate negative error bars.
    for name, figure in (("tradeoff", sweep_plots._tradeoff(data, "mean_total_nfes", "NFEs")),
                         ("heatmap", sweep_plots._heatmap(data)),
                         ("diagnostics", sweep_plots._diagnostics(data))):
        sweep_plots._save(figure, tmp_path / name)
        assert (tmp_path / (name + ".png")).is_file()
