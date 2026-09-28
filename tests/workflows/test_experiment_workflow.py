import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import pytest

from dlm_project.experiments.paths import create_run_directory, experiment_name, experiment_parent
from dlm_project.models.profiles import get_profile
from dlm_project.experiments.config import load_config
from dlm_project.tasks.sudoku.runner import run_experiment

from dlm_project.paths import PROJECT_ROOT as ROOT


def cell_source(notebook, cell_id):
    return "".join(next(cell for cell in notebook["cells"] if cell["id"] == cell_id)["source"])


@pytest.mark.parametrize("family", ["llada", "illada"])
def test_model_dataset_naming_local_time_and_collisions(tmp_path, family):
    config = load_config(ROOT / "configs/sudoku/llada.yaml")
    config["experiment_root"] = str(tmp_path)
    config["model"]["family"] = family
    # 21:35 UTC is 14:35 in Los Angeles on this date.
    now = datetime(2026, 9, 20, 21, 35, 7, tzinfo=timezone.utc)
    run_dir, started = create_run_directory(config, now)
    parent = tmp_path / get_profile(family)["label"] / "Sudoku" / "gen256-steps256-blk128-temp0-lowconf"
    assert run_dir == parent / "20:09:26:14:35:07"
    assert started.utcoffset().total_seconds() == -7 * 3600
    marker = run_dir / "existing.json"
    marker.write_text("preserve me")
    second, _ = create_run_directory(config, now)
    assert second == parent / "20:09:26:14:35:07-02"
    assert marker.read_text() == "preserve me"
    config["generation"].update(steps=128, temperature=0.2)
    assert experiment_name(config["generation"]) == "gen256-steps128-blk128-temp0.2-lowconf"


def test_legacy_output_parent_remains_supported(tmp_path):
    config = load_config(ROOT / "configs/sudoku/llada.yaml")
    config.pop("experiment_root")
    config["output_dir"] = str(tmp_path)
    assert experiment_parent(config) == tmp_path


def test_plot_failure_preserves_record_and_cannot_mark_run_completed(tmp_path, monkeypatch):
    from dlm_project.tasks.sudoku import visualization as sudoku_visualization
    config = load_config(ROOT / "configs/sudoku/illada.yaml")
    config["experiment_root"] = str(tmp_path)

    class Model:
        def predict(self, prompt, **kwargs):
            return "123"

    def fail(*args):
        raise OSError("Simulated full output disk")

    monkeypatch.setattr(sudoku_visualization, "save_prediction_plot", fail)
    with pytest.raises(OSError, match="full output disk"):
        run_experiment(config, [{"puzzle": "0" * 81, "solution": "1" * 81}], Model())
    run_dir = next(tmp_path.rglob("metrics.json")).parent
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["status"] == "running"
    assert metrics["processed"] == 1
    assert metrics["plots_saved"] == 0
    record = json.loads((run_dir / "predictions.jsonl").read_text())
    assert record["raw_output"] == "123"


@pytest.mark.parametrize("family,label", [("llada", "LLaDA"), ("illada", "iLLaDA")])
def test_notebook_config_updater_uses_actual_parameters_and_family(tmp_path, family, label):
    notebook = json.loads((ROOT / f"notebooks/sudoku/{label}_Sudoku_Colab.ipynb").read_text())
    from dlm_project.experiments import paths as experiment_paths
    import yaml

    scope = {
        "yaml": yaml, "Path": Path, "REPO_DIR": ROOT, "MODEL_FAMILY": family,
        "MODEL_PATH": tmp_path / "models" / label, "DRIVE_ROOT": tmp_path,
        "EXPERIMENT_TIMEZONE": "America/Los_Angeles", "profile": get_profile(family),
        "os": type("Env", (), {"environ": {"HF_DATASETS_CACHE": str(tmp_path / "cache")}}),
    }
    exec(cell_source(notebook, "base-config"), scope)
    calls = []
    scope.update(RUN_CONFIG=tmp_path / "run.yaml", experiment_paths=experiment_paths,
                 sys=type("Python", (), {"executable": "python"}),
                 subprocess=type("Process", (), {"run": lambda *args, **kwargs: calls.append(args[0])}))
    exec(cell_source(notebook, "config-updater"), scope)
    saved = yaml.safe_load((tmp_path / "run.yaml").read_text())
    assert saved["model"]["family"] == family
    assert saved["data"]["num_samples"] == (1 if family == "llada" else 3)
    if family == "llada":
        assert saved["generation"]["block_length"] == 256
        assert saved["decoding"]["method"] == "mcts_entropy"
        assert saved["decoding"]["mcts"]["init_tokens"] == 2
    else:
        assert saved["generation"]["block_length"] == 128
    assert Path(saved["output_dir"]) == tmp_path / label / "Sudoku" / experiment_name(saved["generation"], saved.get("decoding"))
    assert calls[0][-1] == "--dataset-only"


def checkout_function(repo_dir, remote):
    notebook = json.loads((ROOT / "notebooks/sudoku/LLaDA_Sudoku_Colab.ipynb").read_text())
    code = cell_source(notebook, "checkout")
    function = next(node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef) and node.name == "ensure_checkout")
    scope = {"REPO_DIR": repo_dir, "REPO_URL": str(remote), "BRANCH": "adi", "subprocess": subprocess,
             "remote_git": lambda args: subprocess.run(["git", *args], check=True, capture_output=True)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "checkout-cell", "exec"), scope)
    return scope["ensure_checkout"]


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True, stderr=subprocess.DEVNULL).strip()


def test_notebook_switches_clean_main_to_adi_and_preserves_dirty_checkout(tmp_path):
    remote, seed, checkout = tmp_path / "origin.git", tmp_path / "seed", tmp_path / "checkout"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", "main", str(seed)], check=True, capture_output=True)
    (seed / "file.txt").write_text("main")
    git(seed, "add", "file.txt")
    git(seed, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "main")
    git(seed, "switch", "-c", "adi")
    (seed / "file.txt").write_text("adi")
    git(seed, "add", "file.txt")
    git(seed, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "adi")
    git(seed, "remote", "add", "origin", str(remote))
    git(seed, "push", "origin", "main", "adi")
    subprocess.run(["git", "clone", "--branch", "main", "--single-branch", str(remote), str(checkout)], check=True, capture_output=True)
    ensure = checkout_function(checkout, remote)
    ensure()
    assert git(checkout, "branch", "--show-current") == "adi"
    assert (checkout / "file.txt").read_text() == "adi"
    (checkout / "file.txt").write_text("my unsaved edits")
    with pytest.raises(RuntimeError, match="local changes"):
        ensure()
    assert (checkout / "file.txt").read_text() == "my unsaved edits"

    # Also reject unpublished local commits without discarding them.
    git(checkout, "add", "file.txt")
    git(checkout, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "local")
    local_head = git(checkout, "rev-parse", "HEAD")
    with pytest.raises(RuntimeError, match="differs from origin/adi"):
        ensure()
    assert git(checkout, "rev-parse", "HEAD") == local_head
