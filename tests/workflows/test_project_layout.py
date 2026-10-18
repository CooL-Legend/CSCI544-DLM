"""Exercise installed entry points and Colab setup after moving source files."""

import json
import os
from pathlib import Path
import site
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from dlm_project.paths import PROJECT_ROOT as ROOT


def test_carc_batch_writes_to_ignored_repository_results():
    script = (ROOT / "scripts/run_carc_llada_sudoku.sbatch").read_text()
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert 'RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results}"' in script
    assert "/results/" in ignored


@pytest.mark.parametrize("script", [
    "cache_model.py", "run_sudoku.py", "run_illada_sudoku.py",
    "run_gsm8k.py", "inspect_mcts.py", "compare_generation.py",
    "run_bgpo_sudoku.py",
    "run_gsm8k_sweep.py", "run_sudoku_sweep.py",
    "eval_gsm8k_random_mdm.py", "eval_gsm8k_topk_margin.py",
    "eval_gsm8k_confidence.py", "eval_gsm8k_best_of_five.py",
    "eval_gsm8k_top_k_margin.py", "eval_gsm8k_best_of_5.py",
    "eval_illada_gsm8k_random_mdm.py", "eval_illada_gsm8k_top_k_margin.py",
    "eval_illada_gsm8k_confidence.py", "eval_illada_gsm8k_best_of_5.py",
    "eval_gsm8k_mcts_confidence.py",
    "run_gsm8k_metrics.py",
])
def test_installed_entry_points_work_outside_checkout(tmp_path, script):
    # -I ignores PYTHONPATH and the script directory. Imports must come from
    # the installed package; no test conftest or source-folder path hacks help.
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts" / script), "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout and "--help" in result.stdout


@pytest.mark.parametrize("notebook,cell_id,profile", [
    ("sudoku/LLaDA_Sudoku_Colab.ipynb", "cell-05", "llada-colab.txt"),
    ("sudoku/iLLaDA_Sudoku_Colab.ipynb", "cell-05", "illada-colab.txt"),
    ("gsm8k/LLaDA_GSM8K_Colab.ipynb", "dependencies", "llada-colab.txt"),
    ("gsm8k/iLLaDA_GSM8K_Colab.ipynb", "dependencies", "illada-colab.txt"),
    ("sudoku/LLaDA_BGPO_Sudoku_Colab.ipynb", "dependencies", "bgpo-colab.txt"),
])
def test_notebook_installs_package_and_refreshes_kernel_paths(monkeypatch, notebook, cell_id, profile):
    cells = json.loads((ROOT / "notebooks" / notebook).read_text())["cells"]
    source = "".join(next(cell for cell in cells if cell.get("id") == cell_id)["source"])
    commands, refreshed = [], []

    def run(command, *, check):
        assert check is True
        commands.append(command)

    monkeypatch.setattr(site, "getsitepackages", lambda: ["runtime-site"])
    monkeypatch.setattr(site, "getusersitepackages", lambda: "user-site")
    monkeypatch.setattr(site, "addsitedir", refreshed.append)
    exec(source, {"REPO_DIR": ROOT, "sys": sys, "subprocess": SimpleNamespace(run=run)})
    assert len(commands) == 2
    requirements = [Path(commands[0][i + 1]) for i, arg in enumerate(commands[0]) if arg == "-r"]
    assert ROOT / "requirements" / profile in requirements
    assert all(path.is_file() for path in requirements)
    assert "--no-deps" in commands[1]
    assert Path(commands[1][commands[1].index("-e") + 1]) == ROOT
    assert refreshed == ["runtime-site", "user-site"]


def test_prompt_preview_runs_through_installed_cli_from_another_directory(tmp_path):
    config = yaml.safe_load((ROOT / "configs/gsm8k/llada.yaml").read_text())
    config["prompt"]["custom_question"] = "What is 3 + 2?"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    preview_path = tmp_path / "preview.json"
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts/run_gsm8k.py"),
         "--config", str(config_path), "--prompt-only", "--preview-path", str(preview_path)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stderr
    preview = json.loads(preview_path.read_text())
    assert preview["question"] == "What is 3 + 2?"
    assert preview["messages"] == [{"role": "user", "content": preview["user_prompt"]}]
    assert "reference_solution" not in preview
