"""Paired reports, one model load, and notebook controls without GPU weights."""

import copy
import json
from pathlib import Path
import sys

import pytest

from dlm_project.experiments.primary_metrics import generate_primary_report, summarize_primary
from dlm_project.experiments.manifests import digest
from dlm_project.paths import PROJECT_ROOT as ROOT
from dlm_project.tasks.gsm8k import benchmark, metrics_sweep
from tests.tasks.test_gsm8k_benchmark import config_for, manifest, wrapper


@pytest.fixture
def controlled_data(monkeypatch):
    frozen = manifest()
    monkeypatch.setattr(metrics_sweep, "get_manifest", lambda *args, **kwargs: frozen)
    monkeypatch.setattr(benchmark, "get_manifest", lambda *args, **kwargs: frozen)
    examples = []
    for index in (0, 1):
        content = {"question": f"Training {index}: one plus one?", "reference_solution": "1 + 1 = 2\n#### 2"}
        examples.append({"row_id": index, **content, "content_hash": digest(content)})
    monkeypatch.setattr(benchmark, "get_gsm8k_demonstrations", lambda *args, **kwargs: {
        "version": "medal_inline_v1", "dataset": {"repo_id": "openai/gsm8k", "subset": "main",
            "split": "train", "revision": "a" * 40}, "examples": examples})


def test_one_load_all_conditions_accounting_and_offline_reports(tmp_path, monkeypatch, controlled_data):
    source = config_for(tmp_path)
    original = copy.deepcopy(source)
    instance, prompts = wrapper()
    loads = []
    def factory(config):
        loads.append(config)
        return instance
    directory, summary = metrics_sweep.run_metrics_sweep(source, budgets=[1, 2], model_factory=factory)
    assert source == original and len(loads) == 1
    setup = json.loads((directory / "setup.json").read_text())
    assert setup["status"] == "completed" and setup["model_load_count"] == setup["warmup_nfes"] == 1
    assert summary["complete"] and summary["common_n"] == 2 and len(summary["conditions"]) == 6
    assert summary["warnings"] == []
    rows = []
    for item in summary["conditions"]:
        run = Path(item["run_dir"])
        metrics = json.loads((run / "metrics.json").read_text())
        assert metrics["model_load_count"] == 0 and metrics["accuracy_ci95_high"] == 1
        rows.extend(json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines())
    assert len(instance.model.seen) == sum(row["total_nfes"] for row in rows) + 1
    for example_id in ("test:0", "test:1"):
        selected = [row for row in rows if row["example_id"] == example_id]
        assert len({row["effective_prompt_hash"] for row in selected}) == 1
        assert len({row["seed"] for row in selected}) == 1
        assert all(row["batch_size"] == 1 and row["state_evaluations"] == row["total_nfes"] for row in selected)
    assert all("REFERENCE_ONLY" not in prompt for prompt in prompts)
    best = next(item for item in summary["conditions"] if item["method"] == "best_of_5")
    assert best["mean_decoding_steps"] == 20 and best["mean_selected_output_steps"] == 4
    for budget in (1, 2):
        item = next(item for item in summary["conditions"] if item["condition"] == f"mcts_{budget}_simulations")
        assert item["mean_mcts_simulations"] == budget
        assert item["mean_total_nfes"] > budget
    plot_data = json.loads((directory / "report/plot_data.json").read_text())
    assert plot_data == summary
    for stem in ("accuracy_vs_simulations", "accuracy_vs_nfes", "accuracy_vs_latency"):
        for extension in ("png", "pdf"):
            assert (directory / f"report/plots/{stem}.{extension}").stat().st_size > 100
    monkeypatch.setattr(benchmark, "_load_model", lambda *args: pytest.fail("Report loaded weights"))
    monkeypatch.setattr(metrics_sweep, "get_manifest", lambda *args, **kwargs: pytest.fail("Report loaded data"))
    monkeypatch.setattr(sys, "argv", ["script", "--report-only", str(directory)])
    metrics_sweep.main()
    assert json.loads((directory / "report/summary.json").read_text()) == summary


def test_failed_condition_keeps_partial_rows_and_no_comparison_denominator(tmp_path, monkeypatch, controlled_data):
    instance, _ = wrapper()
    def fail(*args, **kwargs):
        raise RuntimeError("controlled decoding failure")
    monkeypatch.setitem(benchmark.DECODERS, "top_k_margin", fail)
    with pytest.raises(RuntimeError, match="controlled decoding failure"):
        metrics_sweep.run_metrics_sweep(config_for(tmp_path), budgets=[1], model_factory=lambda _: instance)
    setup_file = next(tmp_path.rglob("setup.json"))
    assert json.loads(setup_file.read_text())["status"] == "failed"
    summary = json.loads((setup_file.parent / "report/summary.json").read_text())
    assert not summary["complete"] and summary["common_n"] == 0
    assert "mcts_1_simulations" in summary["missing_conditions"]
    assert summary["conditions"][0]["available_n"] == 2
    assert all(item["accuracy_percent"] is None for item in summary["conditions"])


@pytest.mark.parametrize("correct", [False, True])
def test_accuracy_intervals_at_boundaries_and_empty_samples(correct):
    rows = [{"exact_match": correct, "total_model_calls": 8, "latency_seconds": seconds,
             "decoding_steps": 4, "mcts_simulations": 0, "search_invocations": 0}
            for seconds in (1., 3.)]
    summary = summarize_primary(rows)
    assert summary["accuracy_percent"] == 100 * int(correct)
    assert 0 <= summary["accuracy_ci95_low"] <= int(correct) <= summary["accuracy_ci95_high"] <= 1
    assert summary["mean_total_nfes"] == 8 and summary["mean_latency_seconds"] == 2
    assert summarize_primary([])["accuracy_percent"] is None


def test_notebook_new_cells_are_cleared_and_controls_do_not_load_weights():
    raw = json.loads((ROOT / "notebooks/gsm8k/LLaDA_GSM8K_Colab.ipynb").read_text())
    ids = ("primary-metrics-intro", "primary-metrics-config", "run-primary-metrics",
           "primary-metrics-report", "primary-metrics-signals")
    cells = {cell.get("id"): cell for cell in raw["cells"]}
    for cell_id in ids:
        cell = cells[cell_id]
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            source = "".join(cell["source"])
            compile(source, cell_id, "exec")
            assert "from_pretrained(" not in source
    controls = "".join(cells["primary-metrics-config"]["source"])
    assert "METRICS_NUM_SAMPLES = 2" in controls and "[4, 8, 16, 32, 64]" in controls
    assert "run_gsm8k_metrics.py" in "".join(cells["run-primary-metrics"]["source"])
    assert "generate_primary_report" in "".join(cells["primary-metrics-report"]["source"])
