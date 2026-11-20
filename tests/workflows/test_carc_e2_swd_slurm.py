"""Run the SWD Slurm wrapper locally with a fake scheduler and prepared assets."""

import ast
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_carc_e2_swd_test.sbatch"


@pytest.fixture
def job(tmp_path):
    repository = tmp_path / "checkout with spaces"
    repository.mkdir()
    (repository / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (repository / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
    environment = tmp_path / "prepared environment"
    (environment / "bin").mkdir(parents=True)
    interpreter = environment / "bin/python"
    interpreter.write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} "$@"\n')
    interpreter.chmod(0o755)
    model = tmp_path / "prepared model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "llada"}')
    mock_bin = tmp_path / "mock-bin"
    mock_bin.mkdir()
    capture = tmp_path / "srun.json"
    srun = mock_bin / "srun"
    srun.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps({"
        "'argv': sys.argv[1:], 'cwd': os.getcwd(), 'env': dict(os.environ)}))\n"
    )
    srun.chmod(0o755)
    env = {
        "REPO_DIR": str(repository), "DLM_ENV": str(environment), "MODEL_DIR": str(model),
        "HF_DATASETS_CACHE": str(tmp_path / "prepared datasets"),
        "PATH": f"{mock_bin}:{os.environ.get('PATH', '')}",
        "CAPTURE": str(capture), "SLURM_JOB_ID": "12345", "SLURM_CPUS_PER_TASK": "4",
    }
    return {"repo": repository, "environment": environment, "model": model,
            "capture": capture, "env": env, "tmp": tmp_path}


def invoke(job, args=(), *, overrides=None, remove=(), script=SCRIPT):
    env = {**job["env"], **(overrides or {})}
    for key in remove:
        env.pop(key, None)
    result = subprocess.run(["bash", str(script), *args], cwd=job["tmp"], env=env,
                            capture_output=True, text=True, timeout=30)
    record = json.loads(job["capture"].read_text()) if job["capture"].is_file() else None
    return result, record


def value_after(args, flag):
    return args[args.index(flag) + 1]


def test_slurm_resources_and_syntax_match_requested_500_question_run():
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    source = SCRIPT.read_text()
    directives = {line.removeprefix("#SBATCH ").strip() for line in source.splitlines()
                  if line.startswith("#SBATCH ")}
    assert {
        "--account=xiangren_1987", "--partition=gpu", "--gpus-per-task=a40:1",
        "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--mem=64G", "--time=24:00:00",
        "--output=results/slurm/e2-swd-%j.out", "--error=results/slurm/e2-swd-%j.err",
    } <= directives
    for location in ("envs/dlm", "model-cache", "datasets"):
        assert f"/project2/xiangren_1987/group_37/{location}" in source


def test_default_dispatch_is_offline_500_test_questions_with_notebook_swd_controls(job):
    result, record = invoke(job)
    assert result.returncode == 0, result.stdout + result.stderr
    args = record["argv"]
    assert args[:3] == [str(job["environment"] / "bin/python"), "-u",
                        str(job["repo"] / "scripts/run_e2.py")]
    assert record["cwd"] == str(job["repo"])
    expected = {
        "--stage": "test", "--num-samples": "500", "--batch-size": "8",
        "--max-depth": "10", "--num-candidates": "3", "--device": "cuda",
        "--reward-mode": "swd", "--reward-lambda": "1.0", "--reward-beta": "1.0",
        "--sim-cap-multiplier": "4", "--model-path": str(job["model"]),
        "--output-dir": str(job["repo"] / "results/e2_swd_test_500"),
    }
    for flag, value in expected.items():
        assert value_after(args, flag) == value
    assert args[args.index("--budgets") + 1:args.index("--budgets") + 3] == ["256", "512"]
    assert args[args.index("--methods") + 1:args.index("--methods") + 3] == ["medal", "ab_medal"]
    assert {"--run-e3", "--profile-resources", "--local-files-only"} <= set(args)
    assert "--no-resume" not in args and "--eval-split" not in args
    assert "same frozen 500-question test pool as E1" in result.stdout


def test_runtime_metadata_is_separate_from_read_only_prepared_assets(job):
    result, record = invoke(job, overrides={
        "HF_HOME": str(job["tmp"] / "wrong-home"), "TRANSFORMERS_CACHE": "wrong-cache",
        "HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0",
    })
    assert result.returncode == 0, result.stdout + result.stderr
    env = record["env"]
    runtime = job["repo"] / ".cache/carc/huggingface"
    assert env["HF_HOME"] == str(runtime)
    assert env["HF_MODULES_CACHE"] == str(runtime / "modules")
    assert env["TRANSFORMERS_CACHE"] == str(runtime / "hub")
    assert env["HF_HUB_CACHE"] == str(job["model"] / "huggingface/hub")
    assert env["HF_DATASETS_CACHE"] == job["env"]["HF_DATASETS_CACHE"]
    assert all(env[name] == "1" for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"))
    assert env["OMP_NUM_THREADS"] == "4"
    assert env["MPLBACKEND"] == "Agg" and env["TOKENIZERS_PARALLELISM"] == "false"
    assert str(job["repo"] / "src") in env["PYTHONPATH"].split(os.pathsep)
    assert not Path(env["HF_DATASETS_CACHE"]).exists()
    assert sorted(path.name for path in job["model"].iterdir()) == ["config.json"]


def test_reward_controls_match_the_swd_evaluation_notebook(job):
    notebook = json.loads((ROOT / "notebooks/gsm8k/E2_Experiment_SWD_Eval.ipynb").read_text())
    controls = next("".join(cell["source"]) for cell in notebook["cells"]
                    if cell["cell_type"] == "code" and "REWARD_MODE =" in "".join(cell["source"]))
    literals = {}
    for node in ast.parse(controls).body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    literals[target.id] = value
    result, record = invoke(job)
    assert result.returncode == 0, result.stdout + result.stderr
    args = record["argv"]
    for flag, variable in (("--reward-mode", "REWARD_MODE"), ("--reward-lambda", "REWARD_LAMBDA"),
                           ("--reward-beta", "REWARD_BETA"), ("--sim-cap-multiplier", "SIM_CAP_MULTIPLIER")):
        assert value_after(args, flag) == str(literals[variable])
    assert args[args.index("--methods") + 1:args.index("--methods") + 3] == list(literals["METHODS"])
    assert args[args.index("--budgets") + 1:args.index("--budgets") + 3] == [str(b) for b in literals["BUDGETS"]]


def test_smoke_arguments_and_prepared_path_overrides(job):
    model = job["tmp"] / "explicit snapshot"
    model.mkdir()
    (model / "config.json").write_text("{}")
    output = job["tmp"] / "smoke output"
    persist = job["tmp"] / "runtime metadata"
    result, record = invoke(job, ["2", "2"], overrides={
        "ENV_DIR": str(job["environment"]), "DLM_ENV": "/missing-env",
        "MODEL_PATH": str(model), "ARTIFACT_ROOT": str(output), "PERSIST_ROOT": str(persist),
        "NUM_SAMPLES": "200", "BATCH_SIZE": "4", "RESULTS_DIR": "/unused-result-root",
    })
    assert result.returncode == 0, result.stdout + result.stderr
    args = record["argv"]
    assert value_after(args, "--num-samples") == value_after(args, "--batch-size") == "2"
    assert value_after(args, "--model-path") == str(model)
    assert value_after(args, "--output-dir") == str(output)
    assert record["env"]["HF_HOME"] == str(persist / "huggingface")


def test_environment_sample_controls_and_results_root(job):
    results = job["tmp"] / "result root"
    result, record = invoke(job, overrides={"NUM_SAMPLES": "200", "BATCH_SIZE": "4",
                                           "RESULTS_DIR": str(results)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert value_after(record["argv"], "--num-samples") == "200"
    assert value_after(record["argv"], "--batch-size") == "4"
    assert value_after(record["argv"], "--output-dir") == str(results / "e2_swd_test_500")


def test_spooled_script_uses_slurm_submit_directory(job):
    spool = job["tmp"] / "slurm spool"
    spool.mkdir()
    script = spool / "slurm_script"
    shutil.copyfile(SCRIPT, script)
    result, record = invoke(job, script=script, remove=("REPO_DIR",),
                            overrides={"SLURM_SUBMIT_DIR": str(job["repo"])})
    assert result.returncode == 0, result.stdout + result.stderr
    assert record["cwd"] == str(job["repo"])


def test_explicit_repository_overrides_submit_directory(job):
    result, record = invoke(job, overrides={"SLURM_SUBMIT_DIR": "/wrong-submit-directory"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert record["cwd"] == str(job["repo"])


@pytest.mark.parametrize("remove,overrides", [
    (("SLURM_JOB_ID",), {}),
    ((), {"MODEL_PATH": "/missing-local-snapshot"}),
    ((), {"ENV_DIR": "/missing-prepared-environment"}),
])
def test_missing_allocation_or_prepared_assets_never_starts_srun(job, remove, overrides):
    result, record = invoke(job, remove=remove, overrides=overrides)
    assert result.returncode != 0
    assert record is None


@pytest.mark.parametrize("arguments", [
    ["0"], ["501"], ["-1"], ["two"], ["01"], ["1", "0"], ["1", "-1"],
    ["1", "2.5"], ["1", "eight"], ["1", "501"], ["1", "8", "extra"],
])
def test_invalid_sample_or_batch_arguments_fail_before_inference(job, arguments):
    result, record = invoke(job, arguments)
    assert result.returncode != 0
    assert record is None
