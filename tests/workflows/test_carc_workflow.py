"""Resource selection uses measured pilots, not the synthetic runtime heuristic."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.benchmark_carc import choose_batch, estimate_seconds


def pilot(seconds, headroom=.5, parity=True):
    return {"strict_parity_passed": parity, "stress_passed": True, "summary": {
        "status": "completed", "fully_fresh_questions": 20, "resumed_question_methods": 0,
        "gpu_headroom_fraction": headroom, "evaluation_wall_seconds": seconds,
        "model_load_seconds": 10, "offline_analysis_wall_seconds": 2,
        "reporting_wall_seconds": 1,
    }}


def test_batch_selection_and_estimate():
    results = {"1": pilot(200), "4": pilot(100), "8": pilot(95)}
    assert choose_batch(results) == 4
    results["8"] = pilot(90)
    assert choose_batch(results) == 8
    results["8"]["summary"]["gpu_headroom_fraction"] = .19
    assert choose_batch(results) == 4
    results["4"]["strict_parity_passed"] = False
    assert choose_batch(results) == 1
    results["1"]["stress_passed"] = False
    assert choose_batch(results) is None
    assert estimate_seconds(pilot(100)["summary"]) == 10 + 1.25 * 25 * 103


@pytest.mark.parametrize("script", ["run_e1_test.py", "run_e1_e3.py", "run_e2.py"])
def test_profile_cli_flag(script):
    result = subprocess.run([sys.executable, f"scripts/{script}", "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--profile-resources" in result.stdout


def test_carc_notebooks_compile_and_default_to_four():
    for name in ("Experiment_1_Test.ipynb", "E2_Experiment_SRIG_Eval.ipynb"):
        notebook = json.loads((Path("notebooks/gsm8k") / name).read_text())
        source = []
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                text = "".join(cell["source"])
                compile(text, f"{name}:{i}", "exec")
                source.append(text)
        joined = "\n".join(source)
        assert "profile_resources=PROFILE_RESOURCES" in joined
        assert 'BATCH_SIZE = 4' in joined or "os.environ.get('BATCH_SIZE', '4')" in joined


def test_slurm_script_syntax():
    result = subprocess.run(["bash", "-n", "slurm/e1_test.sbatch"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
