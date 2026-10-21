"""Single-question prompting, answer isolation and artifact inspection without weights."""

import copy
import json
from pathlib import Path
import sys

import nbformat
import pytest
import torch
import yaml

from dlm_project.tasks.gsm8k import runner as run_gsm8k
from dlm_project.tasks.gsm8k.dataset import load_example, prompt_preview
from dlm_project.tasks.gsm8k.evaluation import extract_answer, score_answer
from dlm_project.tasks.gsm8k.runner import run_prompt_experiment, validate_config
from dlm_project.experiments.config import load_config
from dlm_project.experiments.paths import experiment_parent

from dlm_project.paths import PROJECT_ROOT as ROOT


def config_for(tmp_path):
    config = load_config(ROOT / "configs/gsm8k/llada.yaml")
    config["experiment_root"] = str(tmp_path)
    return config


def example():
    return {"question": "Mia has 3 apples and buys 2 more. How many apples does she have?",
            "reference_solution": "REFERENCE_ONLY: 3 + 2 = 5\n#### 5", "source": {"kind": "gsm8k"}}


class Tokenizer:
    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False}
        return "<user>formatted question</user><assistant>"

    def convert_ids_to_tokens(self, ids):
        return [f"token-{token}" for token in ids]


class Model:
    tokenizer = Tokenizer()
    mask_id = 126336
    identity = {"test_model": True}

    def __init__(self):
        self.prompts = []

    def encode_prompt(self, prompt):
        self.encoded_prompt = prompt
        return torch.tensor([[10, 11, 12]])

    def predict_with_stats(self, prompt, **kwargs):
        assert prompt == self.encoded_prompt
        self.prompts.append(prompt)
        event = {"stage": "prediction", "text": "3 + 2 = 5.\n#### 5.0"}
        kwargs["on_event"](event)
        return {"text": event["text"], "stats": {"model_device": "cpu", "method": "baseline"}, "trace": [event]}


def test_dataset_loads_exactly_one_selected_question_and_keeps_reference_separate(tmp_path, monkeypatch):
    import datasets
    calls = []

    def loader(*args, **kwargs):
        calls.append((args, kwargs))
        return [{"question": example()["question"], "answer": example()["reference_solution"]}]

    monkeypatch.setattr(datasets, "load_dataset", loader)
    config = config_for(tmp_path)
    config["data"].update(question_index=7, cache_dir=str(tmp_path / "cache"))
    selected = load_example(config)
    assert selected["reference_solution"] == example()["reference_solution"]
    assert calls == [(("openai/gsm8k", "main"), {"split": "test[7:8]", "cache_dir": str(tmp_path / "cache")})]
    preview = prompt_preview(config, selected)
    assert selected["question"] in preview["user_prompt"]
    assert config["prompt"]["instructions"] in preview["user_prompt"]
    assert "REFERENCE_ONLY" not in json.dumps(preview)
    assert "reference_solution" not in preview
    assert preview["messages"] == [{"role": "user", "content": preview["user_prompt"]}]


def test_custom_question_never_loads_dataset_or_uses_its_reference(tmp_path, monkeypatch):
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: pytest.fail("Custom questions must not load GSM8K"))
    config = config_for(tmp_path)
    config["prompt"]["custom_question"] = " What is 3 + 2? "
    selected = load_example(config)
    assert selected == {"question": "What is 3 + 2?", "reference_solution": None, "source": {"kind": "custom"}}
    assert score_answer("#### 5", None)["exact_match"] is None


@pytest.mark.parametrize("text,expected", [
    ("Reasoning: 3 + 2 = 5.\n#### 5", "5"), ("#### $1,200.00", "1200"),
    ("#### -3.50", "-3.5"), ("#### +.5", "0.5"), ("#### -0.00", "0"),
    ("#### 2\nActually:\n#### 3.", "3"),
    ("There are 5 apples.", None), ("#### 1,2", None), ("#### 5 dollars", None),
    ("#### 5\n#### unknown", None), ("#### 1/2", None), ("#### NaN", None),
])
def test_explicit_final_answer_parsing(text, expected):
    assert extract_answer(text) == expected


def test_failed_parse_does_not_get_credit_and_invalid_reference_is_reported():
    assert score_answer("The answer is 5", "#### 5")["exact_match"] is False
    assert score_answer("#### 5.00", "#### 5")["exact_match"] is True
    assert score_answer("#### 6", "#### 5")["exact_match"] is False
    with pytest.raises(ValueError, match="reference"):
        score_answer("#### 5", "not a valid reference")


def test_prompt_run_saves_actual_input_raw_output_and_separate_evaluation(tmp_path):
    config, model = config_for(tmp_path), Model()
    before = copy.deepcopy(config)
    run_dir, metrics = run_prompt_experiment(config, example(), model)
    assert config == before
    assert run_dir.parent.parent.name == "GSM8K"
    assert model.prompts and all("REFERENCE_ONLY" not in prompt for prompt in model.prompts)
    record = json.loads((run_dir / "predictions.jsonl").read_text())
    prompt = json.loads((run_dir / "prompt.json").read_text())
    assert record["raw_output"] == (run_dir / "response.txt").read_text() == "3 + 2 = 5.\n#### 5.0"
    assert record["reference_solution"] == example()["reference_solution"]
    assert record["exact_match"] is True
    assert prompt["prompt_token_count"] == 3
    assert prompt["input_token_ids"] == [10, 11, 12]
    assert prompt["input_tokens"] == ["token-10", "token-11", "token-12"]
    assert prompt["output_mask_count"] == 256
    assert "REFERENCE_ONLY" not in json.dumps(prompt)
    assert metrics["status"] == "completed" and metrics["processed"] == 1
    assert json.loads((run_dir / "trace.jsonl").read_text())["text"] == record["raw_output"]
    assert json.loads((run_dir / "config.json").read_text())["run"]["dataset"] == "GSM8K"


def test_interrupted_generation_preserves_prompt_and_trace(tmp_path):
    class Interrupted(Model):
        def predict_with_stats(self, prompt, **kwargs):
            kwargs["on_event"]({"stage": "started"})
            raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        run_prompt_experiment(config_for(tmp_path), example(), Interrupted())
    run_dir = next(tmp_path.rglob("metrics.json")).parent
    assert (run_dir / "prompt.json").is_file()
    assert json.loads((run_dir / "trace.jsonl").read_text())["stage"] == "started"
    assert json.loads((run_dir / "metrics.json").read_text())["status"] == "running"
    assert not (run_dir / "predictions.jsonl").exists()


def test_prompt_only_cli_never_constructs_model(tmp_path, monkeypatch):
    from dlm_project.models import llada as Llada
    monkeypatch.setattr(Llada, "LLaDA", lambda **kwargs: pytest.fail("No weights in prompt preview"))
    config = config_for(tmp_path)
    config["prompt"]["custom_question"] = "What is 3 + 2?"
    config_path, preview_path = tmp_path / "config.yaml", tmp_path / "preview.json"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["run_gsm8k.py", "--config", str(config_path),
                                     "--prompt-only", "--preview-path", str(preview_path)])
    run_gsm8k.main()
    assert json.loads(preview_path.read_text())["question"] == "What is 3 + 2?"


def test_gsm8k_notebook_keeps_four_decoders_and_adds_mcts_confidence():
    path = ROOT / "notebooks/gsm8k/LLaDA_GSM8K_Colab.ipynb"
    notebook = nbformat.from_dict(json.loads(path.read_text()))
    # Ignore saved Colab stream metadata only in this in-memory validation copy.
    # The reference notebook's execution history stays intact on disk.
    for cell in notebook.cells:
        if cell.cell_type == "code" and not cell.get("id", "").startswith(("run-mcts", "preview-mcts", "mcts-")):
            cell.outputs = []
    notebook = nbformat.reads(json.dumps(notebook), as_version=4)
    nbformat.validate(notebook)
    cells = {c.id: c.source for c in notebook.cells}
    for c in notebook.cells:
        if c.cell_type == "code":
            # This reference notebook contains the user's saved Colab results.
            # Validate its sources without changing or rejecting those outputs.
            compile(c.source, c.id, "exec")
    assert 'BRANCH = "chong"' in cells["checkout"]
    assert "LLaDA_GSM8K_Colab.ipynb" in cells["colab-badge"]
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "from_pretrained(" not in code and "LLaDA(" not in code
    for method in ("confidence", "random_mdm", "top_k_margin", "best_of_5"):
        assert f'METHOD = "{method}"' in code
    assert "eval_gsm8k_{METHOD}.py" in code
    assert "configs/gsm8k/decoding.yaml" in code
    assert "random.sample(records, min(5, len(records)))" in code
    assert "reference_answer" in code and "raw_output" in code
    assert "eval_gsm8k_mcts_confidence.py" in code
    assert "MCTS_SETTINGS" in cells["mcts-confidence-config"]
    assert "--run-manifest" in cells["run-mcts-confidence"]
    assert "search_model_calls" in cells["preview-mcts-confidence"]


@pytest.mark.parametrize("section,key,value", [("data", "num_samples", 2), ("data", "question_index", -1),
                                               ("data", "question_index", True), ("data", "split", "validation"),
                                               ("prompt", "instructions", ""), ("prompt", "custom_question", "")])
def test_invalid_walkthrough_config_is_rejected(tmp_path, section, key, value):
    config = config_for(tmp_path)
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)
