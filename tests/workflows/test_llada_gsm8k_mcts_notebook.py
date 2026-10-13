"""Mock the appended MCTS notebook workflow without loading model weights."""

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import nbformat
import pytest
import yaml

from dlm_project.paths import PROJECT_ROOT as ROOT


@pytest.fixture
def cells():
    raw = json.loads((ROOT / "notebooks/gsm8k/LLaDA_GSM8K_Colab.ipynb").read_text())
    cells = {cell.get("id"): cell for cell in raw["cells"]}
    return {key: SimpleNamespace(**cell, text="".join(cell["source"])) for key, cell in cells.items()}


def scope_for(tmp_path, *, profile="smoke", count=2):
    (tmp_path / "content").mkdir()

    def path(value):
        actual = Path(value)
        if actual.is_absolute() and actual.parts[:2] == ("/", "content"):
            return tmp_path / "content" / actual.relative_to("/content")
        return actual

    config = yaml.safe_load((ROOT / "configs/gsm8k/decoding.yaml").read_text())
    config["data"].update(num_samples=count, cache_dir=str(tmp_path / "dataset-cache"))
    config["evaluation"]["profile"] = profile
    config["model"].update(model_path=str(tmp_path / "cached-model"), local_files_only=True)
    config["prompt"]["instructions"] = "Customized instructions. End with #### <number>."
    config["experiment_root"] = str(tmp_path / "drive")
    return {"cfg": config, "Path": path, "REPO_DIR": ROOT, "DRIVE_ROOT": tmp_path / "drive",
            "EXPERIMENT_TIMEZONE": "America/Los_Angeles", "sys": sys, "PROFILE": "full"}


def configuration_scope(cells, tmp_path, **kwargs):
    scope = scope_for(tmp_path, **kwargs)
    # The cell imports pathlib.Path; inject the mock path after that import.
    source = cells["mcts-confidence-config"].text.replace("from pathlib import Path\n", "")
    exec(source, scope)
    return scope


def run_definitions(cells, scope):
    source = cells["run-mcts-confidence"].text.removesuffix("run_mcts_confidence()\n")
    exec(source, scope)


def preview_definitions(cells, scope):
    source = cells["preview-mcts-confidence"].text.removesuffix("show_mcts_answers()\n")
    exec(source, scope)


def test_appended_cells_are_valid_cleared_and_preserve_existing_workflow(cells):
    raw = json.loads((ROOT / "notebooks/gsm8k/LLaDA_GSM8K_Colab.ipynb").read_text())
    validation_copy = copy.deepcopy(raw)
    # Saved Colab streams contain nonstandard metadata. Validate structure and
    # new outputs without modifying the user's original execution history.
    for cell in validation_copy["cells"][:-5]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
    notebook = nbformat.reads(json.dumps(validation_copy), as_version=4)
    nbformat.validate(notebook)
    ids = [cell.id for cell in notebook.cells]
    assert len(ids) == len(set(ids))
    new_ids = ("mcts-confidence-intro", "mcts-confidence-config", "run-mcts-confidence",
               "preview-mcts-confidence", "mcts-confidence-artifacts")
    assert ids[-5:] == list(new_ids)
    for cell_id in new_ids:
        cell = cells[cell_id]
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
            compile(cell.text, cell_id, "exec")
    assert any(cell["cell_type"] == "code" and cell.get("outputs") for cell in raw["cells"][:-5])
    old_code = "\n".join(cell.source for cell in notebook.cells[:-5] if cell.cell_type == "code")
    for method in ("confidence", "random_mdm", "top_k_margin", "best_of_5"):
        assert f'METHOD = "{method}"' in old_code
    assert "NUM_SAMPLES = 2" in old_code and 'BRANCH = "adi"' in old_code


@pytest.mark.parametrize("profile,count,expected", [("smoke", 2, 2), ("smoke", None, 20),
                                                      ("initial", None, 100), ("full", None, 1319)])
def test_configuration_preserves_active_benchmark(cells, tmp_path, profile, count, expected):
    scope = scope_for(tmp_path, profile=profile, count=count)
    original = copy.deepcopy(scope["cfg"])
    source = cells["mcts-confidence-config"].text.replace("from pathlib import Path\n", "")
    exec(source, scope)
    config = yaml.safe_load(scope["MCTS_CONFIG"].read_text())
    assert scope["cfg"] == original
    for section in ("model", "data", "prompt", "evaluation"):
        assert config[section] == original[section]
    assert config["decoding"]["seed"] == 1
    assert scope["mcts_resolved"]["data"]["num_samples"] == expected
    assert config["decoding"]["mcts"]["max_model_calls"] is None
    assert config["generation"]["block_length"] == config["generation"]["gen_length"]
    assert scope["MCTS_MANIFEST"] == tmp_path / "drive/manifests/gsm8k_test.json"
    assert scope["MCTS_RUN_DIR"] is None


def mock_process(scope, tmp_path, *, returncode=0, family="llada", interrupt=False):
    calls, processes = [], []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        waits = 0

        def wait():
            nonlocal waits
            waits += 1
            if interrupt and waits == 1:
                raise KeyboardInterrupt
            return returncode

        process = SimpleNamespace(stdout=io.StringIO("search output\nactual traceback\n"),
                                  wait=wait, terminated=False)
        process.terminate = lambda: setattr(process, "terminated", True)
        processes.append(process)
        if returncode == 0 and not interrupt:
            receipt = Path(command[command.index("--run-manifest") + 1])
            receipt.write_text(json.dumps({"run_dir": str(tmp_path / "exact-run"),
                "status": "completed", "model_family": family, "method": "mcts_confidence",
                "profile": scope["mcts_cfg"]["evaluation"]["profile"]}))
        return process

    scope["subprocess"] = SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT,
                                         CalledProcessError=subprocess.CalledProcessError)
    return calls, processes


def test_run_streams_traceback_and_selects_exact_receipt(cells, tmp_path, capsys):
    scope = configuration_scope(cells, tmp_path, profile="initial", count=None)
    run_definitions(cells, scope)
    calls, processes = mock_process(scope, tmp_path)
    capsys.readouterr()
    directory = scope["run_mcts_confidence"]()
    command, options = calls[0]
    assert command[2] == str(ROOT / "scripts/eval_gsm8k_mcts_confidence.py")
    assert command[command.index("--config") + 1] == str(scope["MCTS_CONFIG"])
    assert command[command.index("--profile") + 1] == "initial"  # Ignore a stale global PROFILE.
    assert command[command.index("--manifest") + 1] == str(scope["MCTS_MANIFEST"])
    assert directory == scope["MCTS_RUN_DIR"] == tmp_path / "exact-run"
    assert options["stderr"] == subprocess.STDOUT and options["text"] is True
    assert "actual traceback" in capsys.readouterr().out
    assert processes[0].stdout.closed
    scope["run_mcts_confidence"]()
    assert calls[0][0][-1] != calls[1][0][-1]


@pytest.mark.parametrize("failure", ["exit", "receipt", "interrupt"])
def test_failed_or_interrupted_run_cannot_preview_old_results(cells, tmp_path, failure):
    scope = configuration_scope(cells, tmp_path)
    run_definitions(cells, scope)
    preview_definitions(cells, scope)
    scope["MCTS_RUN_DIR"] = tmp_path / "old-run"
    _, processes = mock_process(scope, tmp_path, returncode=1 if failure == "exit" else 0,
                                family="illada" if failure == "receipt" else "llada",
                                interrupt=failure == "interrupt")
    expected = {"exit": subprocess.CalledProcessError, "receipt": RuntimeError,
                "interrupt": KeyboardInterrupt}[failure]
    with pytest.raises(expected):
        scope["run_mcts_confidence"]()
    assert scope["MCTS_RUN_DIR"] is None and processes[0].stdout.closed
    if failure == "interrupt":
        assert processes[0].terminated
    with pytest.raises(RuntimeError, match="Complete the MCTS"):
        scope["show_mcts_answers"]()


def test_random_previews_show_saved_answers_initialization_and_costs(cells, tmp_path, capsys):
    scope = configuration_scope(cells, tmp_path)
    preview_definitions(cells, scope)
    directory = tmp_path / "exact-run"
    directory.mkdir()
    scope["MCTS_RUN_DIR"] = directory
    records = [{"example_id": f"test:{i}", "prompt": f"instructions\n\nQuestion:\nQuestion {i}",
                "raw_output": f"response {i}\n#### 1", "predicted_answer": "1", "reference_answer": "1",
                "exact_match": True, "selected_depth": 4, "requested_depth": 20, "candidate_count": 0,
                "stop_reason": "model_call_budget", "fallback_used": True,
                "fallback_reason": "deepest_available_state", "search_model_calls": 512,
                "completion_model_calls": 252, "total_model_calls": 764} for i in range(6)]
    (directory / "metrics.json").write_text(json.dumps({"processed": 6, "accuracy": 1, "fallback_count": 6}))
    (directory / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in records))
    scope["subprocess"] = SimpleNamespace(Popen=lambda *a, **k: pytest.fail("Preview ran inference"))
    capsys.readouterr()
    selected = scope["show_mcts_answers"]()
    output = capsys.readouterr().out
    assert len(selected) == len({row["example_id"] for row in selected}) == 5
    assert all(row in records for row in selected)
    assert "instructions" not in output
    assert "Extracted answer:" in output and "Correct answer:" in output and "response" in output
    assert "Initialization depth: 4/20" in output and "Fallback: True" in output
    assert "search=512, completion=252, total=764" in output and "fallback_count" in output
    assert len(scope["show_mcts_answers"](count=10)) == 6
