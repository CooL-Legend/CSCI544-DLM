"""Unit and notebook tests for E0 (Confidence & Entropy) on the frozen 200/500 GSM8K test split."""

import json
import nbformat
import pytest
import torch

from dlm_project.paths import PROJECT_ROOT
from dlm_project.tasks.gsm8k import experiment_0 as e0, experiment_1 as e1
from dlm_project.tasks.gsm8k.experiment_protocol import build_prompt, example_seed
from tests.tasks.test_experiment_1 import mock_wrapper


@pytest.mark.parametrize("count", [None, 2, 200, 201, 500])
def test_e0_test_pool_matches_e1_content_order_and_prompts(count):
    reference = e1.context(stage="test", num_samples=count)
    actual = e0.context(stage="test", methods=("confidence", "entropy"), num_samples=count)
    assert actual["stage"] == actual["norm_stage"] == "test"
    assert actual["examples"] == reference["examples"]
    assert actual["manifest_hash"] == reference["manifest_hash"]
    assert actual["manifest"] == reference["manifest"]
    assert actual["common"] == reference["common"]
    assert actual["prompt_template_hash"] == reference["prompt_template_hash"]
    assert len(actual["examples"]) == (count if count is not None else 200)
    assert len({row["example_id"] for row in actual["examples"]}) == len(actual["examples"])
    for row, old in zip(actual["examples"], reference["examples"]):
        assert build_prompt(row["question"], actual["common"]["prompt"]) == build_prompt(
            old["question"], reference["common"]["prompt"]
        )
        assert example_seed(row["row_id"]) == example_seed(old["row_id"])


def test_e0_test_slices_primary_200_and_extended_300_equal_full_500():
    ref_500 = e1.context(stage="test", num_samples=500)
    primary_200 = e0.context(stage="test", methods=("confidence",), sample_slice=(0, 200))
    extended_300 = e0.context(stage="test", methods=("confidence",), sample_slice=(200, 500))
    full_500 = e0.context(stage="test", methods=("confidence",), sample_slice=(0, 500))

    assert len(primary_200["examples"]) == 200
    assert len(extended_300["examples"]) == 300
    assert primary_200["examples"] + extended_300["examples"] == full_500["examples"]
    assert full_500["examples"] == ref_500["examples"]


@pytest.mark.parametrize("options", [
    {"stage": "test", "num_samples": 0},
    {"stage": "test", "num_samples": -1},
    {"stage": "test", "num_samples": 501},
    {"stage": "test", "num_samples": True},
    {"stage": "test", "sample_slice": (-1, 2)},
    {"stage": "test", "sample_slice": (0, 501)},
    {"stage": "test", "sample_slice": (10, 10)},
    {"stage": "test", "methods": ("unknown_method",)},
    {"stage": "unknown"},
])
def test_e0_context_rejects_invalid_requests(options):
    with pytest.raises(ValueError):
        e0.context(**options)


@pytest.mark.parametrize("method", ["confidence", "entropy"])
def test_e0_batched_parity_and_resumption(tmp_path, monkeypatch, method):
    monkeypatch.setattr(e0.os, "sync", lambda: None, raising=False)
    w_serial = mock_wrapper()
    w_batch = mock_wrapper()

    ctx_serial = e0.context(stage="test", methods=(method,), num_samples=2)
    ctx_batch = e0.context(stage="test", methods=(method,), num_samples=2)
    ctx_serial["common"]["generation"] = {"gen_length": 8, "steps": 8, "block_length": 8}
    ctx_batch["common"]["generation"] = {"gen_length": 8, "steps": 8, "block_length": 8}

    rows_serial, agg_serial = e0.run_e0_stage(
        w_serial, ctx_serial, tmp_path / "serial", methods=(method,), batch_size=1
    )
    rows_batch, agg_batch = e0.run_e0_stage(
        w_batch, ctx_batch, tmp_path / "batch", methods=(method,), batch_size=2
    )

    assert len(rows_serial) == len(rows_batch) == 2
    assert len(agg_serial) == len(agg_batch) == 1
    for rs, rb in zip(rows_serial, rows_batch):
        assert rs["example_id"] == rb["example_id"]
        assert rs["method"] == rb["method"] == method
        assert rs["condition"] == rb["condition"] == f"{method}_t0"
        assert rs["scoring_text"] == rb["scoring_text"]
        assert rs["raw_output"] == rb["raw_output"]
        assert rs["predicted_answer"] == rb["predicted_answer"]
        assert rs["exact_match"] == rb["exact_match"]
        assert rs["search_nfes"] == rb["search_nfes"] == 0
        assert rs["completion_nfes"] == rb["completion_nfes"] == 8
        assert rs["total_nfes"] == rb["total_nfes"] == 8
        assert e0.is_question_complete(tmp_path / "batch" / "test", method, rb["example_id"])

    # Verify zero-forward-pass resumption
    calls_before = len(w_batch.model.calls)
    resumed_rows, _ = e0.run_e0_stage(
        w_batch, ctx_batch, tmp_path / "batch", methods=(method,), batch_size=2, resume=True
    )
    assert len(w_batch.model.calls) == calls_before
    assert [r["scoring_text"] for r in resumed_rows] == [r["scoring_text"] for r in rows_batch]


def test_e0_paired_comparison_computation():
    rows = [
        {"example_id": "test:1", "method": "confidence", "exact_match": True},
        {"example_id": "test:1", "method": "entropy", "exact_match": True},
        {"example_id": "test:2", "method": "confidence", "exact_match": False},
        {"example_id": "test:2", "method": "entropy", "exact_match": True},
    ]
    paired = e0.compute_paired_comparison(rows, baseline_method="confidence", comparison_method="entropy")
    assert len(paired) == 1
    assert paired[0]["paired_questions"] == 2
    assert paired[0]["confidence_accuracy"] == 0.5
    assert paired[0]["entropy_accuracy"] == 1.0
    assert paired[0]["entropy_only_correct"] == 1
    assert paired[0]["confidence_only_correct"] == 0


@pytest.mark.parametrize("filename,expected_mode", [
    ("E0_Experiment_Confidence_Test.ipynb", "confidence"),
    ("E0_Experiment_Entropy_Test.ipynb", "entropy"),
])
def test_generated_e0_test_notebooks_are_valid_and_compile(filename, expected_mode):
    nb_path = PROJECT_ROOT / "notebooks/gsm8k" / filename
    assert nb_path.is_file(), f"Missing generated notebook: {nb_path}"
    notebook = nbformat.read(nb_path, as_version=4)
    nbformat.validate(notebook)

    full_source = []
    for idx, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{filename}:cell_{idx}", "exec")
            full_source.append(cell.source)

    joined = "\n".join(full_source)
    assert "STAGE = 'test'" in joined
    assert f"METHODS = ({expected_mode!r},)" in joined
    assert "sample_slice=(0, 200)" in joined
    assert "sample_slice=(200, 500)" in joined
    assert "run_e0_stage" in joined
