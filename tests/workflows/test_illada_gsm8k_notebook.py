"""Exercise iLLaDA notebook configuration, streaming runs, and saved previews."""

import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import nbformat
import pytest
import yaml

from dlm_project.models.profiles import get_profile
from dlm_project.paths import PROJECT_ROOT as ROOT


@pytest.fixture
def cells():
    raw = json.loads((ROOT / "notebooks/gsm8k/iLLaDA_GSM8K_Colab.ipynb").read_text())
    # Validate an in-memory cleared copy of the supplied Colab history;
    # retained stream outputs can contain nonstandard Colab metadata.
    for cell in raw["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    notebook = nbformat.reads(json.dumps(raw), as_version=4)
    nbformat.validate(notebook)
    return {cell.id: cell for cell in notebook.cells}


def notebook_scope(tmp_path):
    (tmp_path / "content").mkdir(exist_ok=True)

    def path(value):
        actual = Path(value)
        if actual.is_absolute() and actual.parts[:2] == ("/", "content"):
            return tmp_path / "content" / actual.relative_to("/content")
        return actual

    return {"Path": path, "REPO_DIR": ROOT, "MODEL_FAMILY": "illada",
            "MODEL_PATH": tmp_path / "model", "DRIVE_ROOT": tmp_path / "drive",
            "EXPERIMENT_TIMEZONE": "America/Los_Angeles", "profile": get_profile("illada"),
            "os": SimpleNamespace(environ={"HF_DATASETS_CACHE": str(tmp_path / "datasets")}),
            "subprocess": subprocess, "sys": sys}


def test_notebook_structure_and_empty_outputs(cells):
    assert "iLLaDA_GSM8K_Colab.ipynb" in cells["colab-badge"].source
    assert 'BRANCH = "chong"' in cells["checkout"].source
    assert "requirements/illada-colab.txt" in cells["dependencies"].source
    assert 'MODEL_FAMILY = "illada"' in cells["drive-setup"].source
    code = "\n".join(cell.source for cell in cells.values() if cell.cell_type == "code")
    assert "from_pretrained(" not in code and "LLaDA(" not in code
    for cell in cells.values():
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
            compile(cell.source, cell.id, "exec")
    for method in ("confidence", "random_mdm", "top_k_margin", "best_of_5"):
        slug = method.replace("_", "-")
        assert f'run_decoding("{method}")' in cells[f"run-{slug}"].source
        assert f'show_random_answers("{method}")' in cells.get(f"preview-{slug}", SimpleNamespace(source="")).source or f'show_random_answers("{method}")' in cells[f"run-{slug}"].source


@pytest.mark.parametrize("profile,count", [("smoke", 20), ("initial", 100), ("full", 1319)])
def test_notebook_configuration_defaults(cells, tmp_path, profile, count):
    scope = notebook_scope(tmp_path)
    exec(cells["base-config"].source, scope)
    controls = cells["config-updater"].source.replace("NUM_SAMPLES = 2", "NUM_SAMPLES = None").replace("GEN_LENGTH = 1024", "GEN_LENGTH = 256").replace('PROFILE = "smoke"', f'PROFILE = "{profile}"')
    exec(controls, scope)
    config = yaml.safe_load(scope["BENCH_CONFIG"].read_text())
    assert config["model"]["family"] == "illada" and config["model"]["local_files_only"] is True
    assert config["model"]["model_path"] == str(tmp_path / "model")
    assert config["generation"] == {"gen_length": 256, "steps": 256, "block_length": 128, "temperature": 1.0}
    assert config["decoding"]["seed"] == 1 and config["data"]["num_samples"] is None
    assert scope["resolved"]["data"]["num_samples"] == count
    assert config["data"]["cache_dir"] == str(tmp_path / "datasets")
    assert scope["MANIFEST_PATH"] == tmp_path / "drive/manifests/gsm8k_test.json"
    assert config["prompt"] == scope["base_config"]["prompt"]  # Preserve supplied customized few-shot instructions.


def test_notebook_configuration_override(cells, tmp_path):
    scope = notebook_scope(tmp_path)
    exec(cells["base-config"].source, scope)
    controls = cells["config-updater"].source.replace("NUM_SAMPLES = 2", "NUM_SAMPLES = None").replace("GEN_LENGTH = 1024", "GEN_LENGTH = 256").replace("NUM_SAMPLES = None", "NUM_SAMPLES = 10")
    controls = controls.replace("TEMPERATURE = 1.0", "TEMPERATURE = 0.7").replace("SEED = 1", "SEED = 17")
    exec(controls, scope)
    config = yaml.safe_load(scope["BENCH_CONFIG"].read_text())
    assert config["data"]["num_samples"] == 10
    assert config["generation"]["temperature"] == .7 and config["decoding"]["seed"] == 17


def helper_scope(cells, tmp_path):
    scope = notebook_scope(tmp_path)
    exec(cells["base-config"].source, scope)
    exec(cells["config-updater"].source, scope)
    exec(cells["run-helpers"].source, scope)
    return scope


def mock_process(scope, tmp_path, *, returncode=0, receipt_family="illada"):
    calls = []
    processes = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        process = SimpleNamespace(stdout=io.StringIO("child output\nactual traceback\n"),
                                  wait=lambda: returncode)
        processes.append(process)
        if returncode == 0:
            receipt = Path(command[command.index("--run-manifest") + 1])
            method = Path(command[2]).stem.removeprefix("eval_illada_gsm8k_")
            directory = tmp_path / "exact-run" / method
            receipt.write_text(json.dumps({"run_dir": str(directory), "status": "completed",
                                           "model_family": receipt_family, "method": method,
                                           "profile": scope["PROFILE"]}))
        return process

    scope["subprocess"] = SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT,
                                         CalledProcessError=subprocess.CalledProcessError)
    return calls, processes


@pytest.mark.parametrize("method", ["confidence", "random_mdm", "top_k_margin", "best_of_5"])
def test_run_helper_streams_output_and_uses_exact_receipt(cells, tmp_path, capsys, method):
    scope = helper_scope(cells, tmp_path)
    calls, processes = mock_process(scope, tmp_path)
    capsys.readouterr()
    directory = scope["run_decoding"](method)
    command, options = calls[0]
    assert command[2] == str(ROOT / "scripts" / f"eval_illada_gsm8k_{method}.py")
    assert command[command.index("--config") + 1] == str(scope["BENCH_CONFIG"])
    assert command[command.index("--manifest") + 1] == str(scope["MANIFEST_PATH"])
    assert command[command.index("--profile") + 1] == "smoke"
    assert options["stderr"] == subprocess.STDOUT
    assert "actual traceback" in capsys.readouterr().out
    assert scope["RUN_DIRS"][method] == directory == tmp_path / "exact-run" / method
    assert processes[0].stdout.closed


def test_failed_run_does_not_select_old_answers(cells, tmp_path, capsys):
    scope = helper_scope(cells, tmp_path)
    scope["RUN_DIRS"]["confidence"] = tmp_path / "old-run"
    mock_process(scope, tmp_path, returncode=1)
    with pytest.raises(subprocess.CalledProcessError):
        scope["run_decoding"]("confidence")
    assert "actual traceback" in capsys.readouterr().out
    assert "confidence" not in scope["RUN_DIRS"]
    with pytest.raises(RuntimeError, match="Run this decoding method"):
        scope["show_random_answers"]("confidence")


def test_run_helper_rejects_wrong_model_receipt(cells, tmp_path):
    scope = helper_scope(cells, tmp_path)
    mock_process(scope, tmp_path, receipt_family="llada")
    with pytest.raises(RuntimeError, match="receipt"):
        scope["run_decoding"]("confidence")
    assert "confidence" not in scope["RUN_DIRS"]


def test_random_previews_read_only_saved_answers(cells, tmp_path, capsys):
    scope = helper_scope(cells, tmp_path)
    directory = tmp_path / "exact-run"
    directory.mkdir()
    scope["RUN_DIRS"]["best_of_5"] = directory
    records = [{"example_id": f"test:{i}", "prompt": f"instructions\n\nQuestion:\nQuestion {i}",
                "raw_output": f"response {i}\n#### 1", "predicted_answer": "1",
                "reference_answer": "1", "exact_match": True, "votes": {"1": 5}}
               for i in range(6)]
    (directory / "metrics.json").write_text(json.dumps({"accuracy": 1, "processed": 6}))
    (directory / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in records))
    scope["subprocess"] = SimpleNamespace(Popen=lambda *a, **k: pytest.fail("Inference ran during preview"))
    capsys.readouterr()
    selected = scope["show_random_answers"]("best_of_5")
    output = capsys.readouterr().out
    assert len(selected) == len({row["example_id"] for row in selected}) == 5
    assert all(row in records for row in selected)
    assert "instructions" not in output
    assert "Candidate answer votes:" in output and "Correct answer:" in output
    assert len(scope["show_random_answers"]("best_of_5", count=10)) == 6
