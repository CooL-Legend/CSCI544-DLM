"""Four independent scripts share examples while storing honest accuracy/cost."""

import copy
import csv
import json
from pathlib import Path
import sys

import pytest
import torch

from dlm_project.models.llada import LLaDA
from dlm_project.models.profiles import get_profile
from dlm_project.experiments.config import load_config
from dlm_project.experiments.manifests import digest
from dlm_project.paths import PROJECT_ROOT as ROOT
from dlm_project.tasks.gsm8k import decoder_evaluation as evaluation
from tests.decoding.helpers import FakeModel, FakeTokenizer, MASK, PROMPT


def example(position):
    question = f"How many items are there in case {position}?"
    reference = "secret_reference (scoring only)\n#### 1"
    return {"example_id": f"train:{position}", "position": position,
            "partition": "development", "source_split": "train", "row_id": position,
            "question": question, "reference_solution": reference,
            "content_hash": digest({"question": question, "reference_solution": reference})}


def manifest():
    return {"version": 1, "task": "gsm8k", "sample_seed": 1,
            "dataset": {"repo_id": "openai/gsm8k", "subset": "main", "revision": "a" * 40},
            "examples": [example(i) for i in range(3)]}


def config_for(tmp_path):
    config = load_config(ROOT / "configs/gsm8k/llada_decoder_baselines.yaml")
    config["data"]["num_samples"] = 2
    config["experiment_root"] = str(tmp_path)
    config["model"].update(device="cpu", dtype="float32")
    config["generation"].update(gen_length=4, steps=4, block_length=4)
    config["decoding"]["top_k_positions"] = 2
    config["decoding"]["best_of_five"]["score_block_size"] = 2
    return config


def wrapper():
    instance = LLaDA.__new__(LLaDA)
    instance.device, instance.mask_id, instance.family = "cpu", MASK, "llada"
    instance.model, instance.tokenizer = FakeModel(), FakeTokenizer()
    instance.identity = {**get_profile("llada"), "controlled_test_model": True}
    prompts = []
    instance.encode_prompt = lambda prompt: (prompts.append(prompt), PROMPT.clone())[1]
    instance.tokenizer.batch_decode = lambda *args, **kwargs: ["#### 1"]
    return instance, prompts


@pytest.mark.parametrize("method,expected_nfes,generation,scoring", [
    ("random_mdm", 4, 4, 0), ("topk_margin", 2, 2, 0),
    ("confidence_guided", 4, 4, 0), ("best_of_five", 30, 20, 10),
])
def test_separate_evaluations_share_frozen_examples_and_count_all_forwards(
        tmp_path, monkeypatch, method, expected_nfes, generation, scoring):
    monkeypatch.setattr(evaluation, "get_manifest", lambda *args, **kwargs: manifest())
    instance, prompts = wrapper()
    loads = []
    def factory(config):
        loads.append(config)
        return instance
    source = config_for(tmp_path)
    original = copy.deepcopy(source)
    directory, summary = evaluation.run_evaluation(method, source, model_factory=factory)
    assert source == original
    assert len(loads) == 1 and summary["status"] == "completed"
    assert summary["accuracy"] == summary["parse_rate"] == 1
    assert summary["evaluated_n"] == summary["correct"] == 2
    assert summary["accuracy_ci95_low"] < 1 == summary["accuracy_ci95_high"]
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    assert [row["example_id"] for row in rows] == ["train:0", "train:1"]
    assert [row["seed"] for row in rows] == [1, 2]
    assert {row["prompt_hash"] for row in rows} == {digest(prompt) for prompt in prompts}
    assert all("secret_reference" not in prompt for prompt in prompts)
    assert len(prompts) == 3  # One warm-up, one prompt per example.
    assert all(row["generation_nfes"] == generation for row in rows)
    assert all(row["scoring_nfes"] == scoring for row in rows)
    assert all(row["total_nfes"] == expected_nfes for row in rows)
    assert len(instance.model.seen) == 1 + 2 * expected_nfes
    assert summary["setup"]["warmup_nfes"] == 1
    assert all((directory / row["trace_file"]).is_file() for row in rows)
    assert all("logits" not in (directory / row["trace_file"]).read_text() for row in rows)
    assert len(list(csv.DictReader((directory / "results.csv").open()))) == 2
    assert (directory / "report.md").is_file()
    # Report-only regenerates metrics from saved artifacts without touching the model.
    monkeypatch.setattr(evaluation, "_load_model", lambda *args: pytest.fail("Loaded model in report-only"))
    monkeypatch.setattr(evaluation, "get_manifest", lambda *args, **kwargs: pytest.fail("Loaded dataset in report-only"))
    monkeypatch.setattr(sys, "argv", ["script", "--report-only", str(directory)])
    evaluation.main(method)
    assert json.loads((directory / "summary.json").read_text()) == summary


def test_incomplete_prediction_not_fabricated_and_reference_stays_out_of_decoder(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "get_manifest", lambda *args, **kwargs: manifest())
    instance, prompts = wrapper()
    original = evaluation._decode
    calls = 0
    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("controlled generation failure")
        return original(*args)
    monkeypatch.setattr(evaluation, "_decode", interrupted)
    with pytest.raises(RuntimeError, match="controlled generation failure"):
        evaluation.run_evaluation("random_mdm", config_for(tmp_path), model_factory=lambda _: instance)
    directory = next(tmp_path.rglob("run_state.json")).parent
    records = (directory / "predictions.jsonl").read_text().splitlines()
    assert len(records) == 1
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["status"] == "incomplete" and summary["evaluated_n"] == 1
    assert "secret_reference" not in "\n".join(prompts)


def test_config_validation_defaults_and_method_specific_limits(tmp_path):
    source = config_for(tmp_path)
    original = copy.deepcopy(source)
    for method in evaluation.METHODS:
        resolved = evaluation.resolve_settings(method, source)
        assert resolved["method"] == method
        assert resolved["data"]["num_samples"] == 2
    assert original == source
    source["generation"]["steps"] = 1
    with pytest.raises(ValueError, match="Top-K Margin"):
        evaluation.resolve_settings("topk_margin", source)
    with pytest.raises(ValueError, match="Confidence"):
        evaluation.resolve_settings("confidence_guided", source)
    source["generation"]["steps"] = 4
    source["decoding"]["top_k_positions"] = True
    with pytest.raises(ValueError, match="top_k_positions"):
        evaluation.resolve_settings("topk_margin", source)
