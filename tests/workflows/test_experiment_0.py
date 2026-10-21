"""One-load E0 integration, partial-run preservation, reports and Colab cells."""

import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import nbformat
import pytest
import torch
import yaml

from dlm_project.experiments.config import load_config
from dlm_project.models.profiles import get_profile
from dlm_project.paths import PROJECT_ROOT as ROOT
from dlm_project.tasks.gsm8k import experiment_0 as e0
from dlm_project.tasks.gsm8k.e0_prompt import build_e0_prompt
from dlm_project.tasks.gsm8k.e0_report import generate_report
from tests.decoding.helpers import FakeModel, PROMPT, MASK
from tests.tasks.test_e0_dataset import Tokenizer, manifest


def config(tmp_path):
    source = load_config(ROOT / "configs/gsm8k/experiment_0.yaml")
    source["experiment_root"] = str(tmp_path)
    source["model"].update(device="cpu", dtype="float32")
    source["data"]["num_samples"] = 2
    source["generation"] = dict(gen_length=8, steps=8, block_length=8)
    source["decoding"]["mcts"].update(init_tokens=2, top_k_tokens=2, top_k_actions=3)
    source["evaluation"]["search_budgets"] = [1, 8, 16]
    return source


def wrapper():
    prompts = []
    def encode(text):
        prompts.append(text)
        return PROMPT.clone()
    model = FakeModel(lambda _: [0, .05, .04, .6, .2, .11])
    model.config = SimpleNamespace(eos_token_id=1, vocab_size=6)
    return SimpleNamespace(model=model, tokenizer=Tokenizer(), device="cpu", mask_id=MASK,
                           identity={**get_profile(), "controlled_test_model": True}, encode_prompt=encode), prompts


@pytest.fixture
def prepared(monkeypatch):
    frozen = manifest()
    monkeypatch.setattr(e0, "get_manifest", lambda *args, **kwargs: frozen)
    return wrapper()


@pytest.mark.parametrize("run_order", ["example", "condition"])
def test_six_conditions_one_load_real_accounting_and_offline_report(tmp_path, monkeypatch, prepared,
                                                                  capsys, run_order):
    instance, prompts = prepared
    source = config(tmp_path)
    source["evaluation"]["run_order"] = run_order
    original = copy.deepcopy(source)
    loads = []
    def factory(config):
        loads.append(config)
        return instance
    receipt = tmp_path / "receipt.json"
    directory, summary = e0.run_experiment(source, model_factory=factory, run_manifest=receipt)
    assert source == original and len(loads) == 1
    assert summary["complete"] and summary["common_n"] == 2
    assert len(summary["conditions"]) == 6 and len(summary["gains"]) == 9
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 12
    examples = json.loads((directory / "selection.json").read_text())
    condition_ids = [condition["id"] for condition in loads[0]["conditions"]]
    expected = ([(example["example_id"], condition) for condition in condition_ids for example in examples]
                if run_order == "condition" else
                [(example["example_id"], condition) for example in examples for condition in condition_ids])
    assert [(row["example_id"], row["condition"]) for row in rows] == expected
    saved_receipt = json.loads(receipt.read_text())
    assert summary["run_order"] == saved_receipt["run_order"] == run_order
    assert saved_receipt["condition_ids"] == condition_ids
    streamed = capsys.readouterr().out
    if run_order == "condition":
        for previous, following in zip(condition_ids, condition_ids[1:]):
            assert streamed.index(f"Finished {previous} | n=2") < streamed.index(f"Starting {following} on 2 questions")
    assert len(instance.model.seen) == sum(row["total_nfes"] for row in rows) + 1
    assert json.loads(receipt.read_text())["run_dir"] == str(directory)
    assert all("REFERENCE_ONLY" not in prompt for prompt in prompts)
    assert len(prompts) == 3  # Warm-up + one tokenization per question, not per method.
    for row in rows:
        assert row["prompt"].startswith(
            "Decompose the question into three steps and solve the math problem following the examples.\n"
            "State the goal of each step clearly.\nShow your final numerical answer after ####."
        )
        assert row["prompt"].count("Example 1:") == row["prompt"].count("Example 2:") == 1
        assert "#### 27\n\nExample 2:" in row["prompt"]
        assert "#### 16\n\nNow solve the following question using the same decomposition format." in row["prompt"]
        assert row["prompt"].endswith(f"Problem: {row['question']}")
        assert "\nQuestion:\n" not in row["prompt"]
        assert row["protocol_version"] == "e0_fixed_budget_v4"
        assert row["temperature"] == 0.0
        if row["method"].startswith("medal"):
            assert row["details"]["temperature"] == 0.0
    for example in summary["common_example_ids"]:
        paired = [row for row in rows if row["example_id"] == example]
        assert len({row["prompt_ids_hash"] for row in paired}) == 1
        assert len({row["prompt_hash"] for row in paired}) == 1
        root = next(row for row in paired if row["condition"] == "medal_budget_b1")
        control = next(row for row in paired if row["method"] == "medal_no_search")
        greedy = next(row for row in paired if row["method"] == "confidence")
        assert control["condition"] == "medal_no_search_t0"
        assert control["full_output_ids"] == greedy["full_output_ids"]
        assert root["full_output_ids"] == control["full_output_ids"]
        assert root["total_nfes"] == control["total_nfes"] + 1
        assert root["fallback_decode_used"] and root["status"] == "completed"
    saved = json.loads((directory / "config.json").read_text())
    assert saved["prompt"]["medal_decomposition"] is False
    assert saved["decoding"]["completion_temperature"] == 0.0
    assert "The T=0 no-search MEDAL control" in (directory / "report/report.md").read_text()
    for stem in ("accuracy_vs_total_nfes", "accuracy_vs_runtime"):
        for extension in ("png", "pdf"):
            assert (directory / f"report/plots/{stem}.{extension}").stat().st_size > 100
    monkeypatch.setattr(e0, "_load_model", lambda *args: pytest.fail("Report loaded model"))
    monkeypatch.setattr(e0, "get_manifest", lambda *args, **kw: pytest.fail("Report downloaded dataset"))
    assert generate_report(directory) == summary
    monkeypatch.setattr(sys, "argv", ["e0", "--report-only", str(directory)])
    e0.main()
    rows[0]["total_nfes"] += 1
    (directory / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="accounting"):
        generate_report(directory)


def test_quota_failures_record_answers_as_incorrect_and_continue(tmp_path, prepared):
    instance, _ = prepared
    source = config(tmp_path)
    source["evaluation"].update(search_mode="candidates", search_budgets=[1, 16])
    directory, summary = e0.run_experiment(source, model_factory=lambda _: instance)
    assert summary["complete"] and summary["search_mode"] == "candidates"
    row = next(item for item in summary["conditions"] if item["id"] == "medal_candidates_b1")
    assert row["search_incomplete_count"] == 2 and row["accuracy_percent"] == 0
    assert row["mean_completion_nfes"] == 0 and row["fallback_rate"] == 0
    assert summary["common_n"] == 2


@pytest.mark.parametrize("failure", [RuntimeError("controlled error"), KeyboardInterrupt()])
@pytest.mark.parametrize("run_order", ["example", "condition"])
def test_errors_keep_partial_records_actual_cost_and_receipt(tmp_path, monkeypatch, prepared, failure, run_order):
    instance, _ = prepared
    def fail(model, ids, *args, **kwargs):
        with torch.inference_mode():
            model(ids)
        raise failure
    monkeypatch.setitem(e0.DECODERS, "entropy", fail)
    receipt = tmp_path / "receipt.json"
    source = config(tmp_path)
    source["evaluation"]["run_order"] = run_order
    with pytest.raises(type(failure)):
        e0.run_experiment(source, model_factory=lambda _: instance, run_manifest=receipt)
    saved = json.loads(receipt.read_text())
    assert saved["status"] == ("interrupted" if isinstance(failure, KeyboardInterrupt) else "failed")
    directory = Path(saved["run_dir"])
    summary = json.loads((directory / "report/summary.json").read_text())
    assert not summary["complete"] and summary["common_n"] == 0
    assert summary["unfinished_attempt_nfes"] == 1
    finished = 2 if run_order == "condition" else 1
    assert summary["terminal_predictions"] == finished
    assert summary["observed_conditions"][0]["n"] == finished
    assert all(item["n"] == 0 for item in summary["observed_conditions"][1:])


def test_config_defaults_and_overrides():
    source = load_config(ROOT / "configs/gsm8k/experiment_0.yaml")
    default = e0.resolve_settings(source)
    assert default["data"]["num_samples"] == 5
    assert default["decoding"]["mcts"]["init_tokens"] == 5
    assert default["decoding"]["mcts"]["stop_mode"] == "budget"
    assert default["evaluation"]["search_budgets"] == [512, 1024, 2048]
    assert default["evaluation"]["run_order"] == "example"
    assert default["decoding"]["completion_method"] == "llada"
    assert default["decoding"]["completion_temperature"] == 0.0
    assert all(condition["temperature"] == 0.0 for condition in default["conditions"])
    assert all(c["completion_method"] == "llada" for c in default["conditions"] if c["method"].startswith("medal"))
    question = "How many items are in {a, b, c}?"
    assert build_e0_prompt(question, default["prompt"]).endswith(f"Problem: {question}")
    assert build_e0_prompt(question, {"instructions": "Solve."}) == f"Solve.\n\nQuestion:\n{question}"
    for invalid_prompt in (
        {"instructions": "Solve."},
        {"template": "Missing question placeholder"},
        {"template": "{GSM8K_question} {GSM8K_question}"},
    ):
        invalid = copy.deepcopy(source)
        invalid["prompt"] = invalid_prompt
        with pytest.raises(ValueError, match="prompt.template"):
            e0.resolve_settings(invalid)
    extended = e0.resolve_settings(source, stage="evaluation", init_tokens=7,
                                   search_budgets=[256, 512, 1024], run_order="condition")
    assert extended["data"]["num_samples"] == 20
    assert extended["decoding"]["mcts"]["init_tokens"] == 7
    assert extended["evaluation"]["run_order"] == "condition"
    assert [condition["id"] for condition in extended["conditions"]] == [
        "confidence_t0", "entropy_t0", "medal_no_search_t0",
        "medal_budget_b256", "medal_budget_b512", "medal_budget_b1024"]
    assert e0.resolve_settings(source, stage="test")["data"]["num_samples"] == 25
    for override in ({"search_budgets": [1, 1]}, {"search_budgets": [0]}, {"init_tokens": 256},
                     {"run_order": "unknown"}):
        with pytest.raises(ValueError):
            e0.resolve_settings(source, **override)
    legacy = copy.deepcopy(source)
    legacy["decoding"]["completion_method"] = "medal"
    with pytest.raises(ValueError, match="positive completion_temperature"):
        e0.resolve_settings(legacy)
    legacy["decoding"]["completion_temperature"] = 1.0
    assert e0.resolve_settings(legacy)["conditions"][2]["id"] == "medal_no_search_t1"
    for invalid_temperature in (-1, float("nan"), True):
        invalid = copy.deepcopy(source)
        invalid["decoding"]["completion_temperature"] = invalid_temperature
        with pytest.raises(ValueError, match="temperature"):
            e0.resolve_settings(invalid)
    source["prompt"]["medal_decomposition"] = True
    with pytest.raises(ValueError, match="decomposition"):
        e0.resolve_settings(source)


def test_cli_sigterm_flushes_current_forward_count(tmp_path, monkeypatch, prepared):
    instance, _ = prepared
    source = tmp_path / "config.yaml"
    source.write_text(yaml.safe_dump(config(tmp_path)))
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(e0, "_load_model", lambda _: instance)
    def terminate(model, ids, *args, **kwargs):
        with torch.inference_mode():
            model(ids)
        signal.raise_signal(signal.SIGTERM)
    monkeypatch.setitem(e0.DECODERS, "entropy", terminate)
    monkeypatch.setattr(sys, "argv", ["e0", "--config", str(source), "--run-manifest", str(receipt),
                                     "--run-order", "condition"])
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        e0.main()
    assert signal.getsignal(signal.SIGTERM) == previous
    saved = json.loads(receipt.read_text())
    assert saved["status"] == "interrupted"
    assert saved["run_order"] == "condition"
    setup = json.loads((Path(saved["run_dir"]) / "setup.json").read_text())
    assert setup["active_total_nfes"] == 1


@pytest.mark.parametrize("label,stage,count", [("Dev", "dev", 5), ("Evaluation", "evaluation", 20),
                                              ("Testing", "test", 25)])
def test_notebooks_valid_controls_and_exact_subprocess_receipts(tmp_path, monkeypatch, label, stage, count):
    notebook = nbformat.read(ROOT / f"notebooks/gsm8k/Experiment_0_Baseline_Test_{label}.ipynb", as_version=4)
    nbformat.validate(notebook)
    cells = {cell.id: cell.source for cell in notebook.cells}
    assert len(cells) == len(notebook.cells)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
            compile(cell.source, cell.id, "exec")
            assert "from_pretrained(" not in cell.source
    assert 'BRANCH = "adi"' in cells["e0-source"] and 'SOURCE_MODE = "github"' in cells["e0-source"]
    assert 'SOURCE_MODE == "upload"' in cells["e0-source"]
    assert "requirements/llada-colab.txt" in cells["e0-dependencies"]
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "data"))
    (tmp_path / "content").mkdir()
    def path(value):
        value = Path(value)
        return tmp_path / value.relative_to("/") if value.is_relative_to("/content") else value
    scope = {"REPO_DIR": ROOT, "Path": path, "MODEL_PATH": tmp_path / "model", "os": os,
             "DRIVE_ROOT": tmp_path, "EXPERIMENT_TIMEZONE": "America/Los_Angeles", "profile": get_profile()}
    exec(cells["e0-config"], scope)
    assert scope["STAGE"] == stage and scope["NUM_SAMPLES"] == count
    assert scope["INIT_TOKENS"] == 5 and scope["RUN_QUOTA_COMPARISON"] is False
    assert scope["TEMPERATURE"] == scope["cfg"]["decoding"]["completion_temperature"] == 0.0
    assert all(condition["temperature"] == 0.0 for condition in e0.resolve_settings(scope["cfg"])["conditions"])
    expected_budgets = [512, 1024, 2048] if label == "Dev" else [256, 512, 1024]
    expected_order = "example" if label == "Dev" else "condition"
    assert scope["SEARCH_BUDGETS"] == expected_budgets
    assert scope["cfg"]["evaluation"].get("run_order", "example") == expected_order
    calls = []
    def launch(command, **kwargs):
        calls.append(command)
        receipt = Path(command[command.index("--run-manifest") + 1])
        mode = command[command.index("--search-mode") + 1]
        receipt.write_text(json.dumps({"run_dir": str(tmp_path / f"exact-{mode}-run"), "status": "completed",
                                      "stage": stage, "search_mode": mode, "protocol_version": e0.PROTOCOL_VERSION,
                                      "run_order": expected_order,
                                      "condition_ids": ["confidence_t0", "entropy_t0", "medal_no_search_t0",
                                                        *[f"medal_{mode}_b{b}" for b in expected_budgets]]}))
        return SimpleNamespace(stdout=io.StringIO("streamed progress\n"), wait=lambda: 0)
    monkeypatch.setattr(subprocess, "Popen", launch)
    exec(cells["e0-run"], scope)
    assert scope["E0_RUN_DIR"] == tmp_path / "exact-budget-run"
    assert calls[0][calls[0].index("--init-tokens") + 1] == "5"
    assert calls[0][calls[0].index("--num-samples") + 1] == str(count)
    budget_start = calls[0].index("--search-budgets") + 1
    assert calls[0][budget_start:budget_start + 3] == list(map(str, expected_budgets))
    if label != "Dev":
        assert calls[0][calls[0].index("--run-order") + 1] == "condition"
        assert "2048" not in cells["e0-config"]
    assert "rglob" not in cells["e0-report"] and "--report-only" in cells["e0-report"]
    assert scope["run_e0"]("candidates") == tmp_path / "exact-candidates-run"
    assert calls[-1][calls[-1].index("--init-tokens") + 1] == "5"

    def fail(command, **kwargs):
        receipt = Path(command[command.index("--run-manifest") + 1])
        receipt.write_text(json.dumps({"run_dir": str(tmp_path / "partial"), "status": "failed"}))
        return SimpleNamespace(stdout=io.StringIO("actual traceback\n"), wait=lambda: 1)
    monkeypatch.setattr(subprocess, "Popen", fail)
    with pytest.raises(subprocess.CalledProcessError):
        scope["run_e0"]()
    assert scope["LAST_PARTIAL_RUN_DIR"] == tmp_path / "partial"
