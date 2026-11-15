"""Exercise the CARC job wrapper without submitting Slurm jobs or loading a GPU."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from dlm_project.models.profiles import get_profile


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_carc_e1_test.sbatch"
PROFILE = get_profile("llada")
REVISION = PROFILE["revision"]
HUB_REPO = "models--" + PROFILE["repo_id"].replace("/", "--")
MODEL_NAME = PROFILE["repo_id"].split("/")[-1]


def snapshot(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text('{"model_type": "llada"}')
    return directory


@pytest.fixture
def job(tmp_path):
    repository = tmp_path / "checkout with spaces"
    repository.mkdir()
    (repository / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (repository / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
    (repository / "results/slurm").mkdir(parents=True)
    environment = tmp_path / "prepared environment"
    (environment / "bin").mkdir(parents=True)
    interpreter = environment / "bin/python"
    interpreter.write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} "$@"\n')
    interpreter.chmod(0o755)
    model = snapshot(tmp_path / "prepared model")
    data = tmp_path / "prepared datasets"
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
    removed = {"REPO_DIR", "ENV_DIR", "DLM_ENV", "MODEL_DIR", "MODEL_PATH", "RESULTS_DIR",
               "ARTIFACT_ROOT", "PERSIST_ROOT", "NUM_SAMPLES", "BATCH_SIZE"}
    env = {key: value for key, value in os.environ.items()
           if key not in removed and not key.startswith(("HF_", "TRANSFORMERS_", "SLURM_"))}
    env.update(
        REPO_DIR=str(repository), DLM_ENV=str(environment), MODEL_DIR=str(model),
        HF_DATASETS_CACHE=str(data), PATH=f"{mock_bin}:{os.environ.get('PATH', '')}",
        CAPTURE=str(capture), OMP_NUM_THREADS="1", SLURM_JOB_ID="12345",
    )
    return {"repo": repository, "environment": environment, "model": model,
            "data": data, "capture": capture, "env": env, "tmp": tmp_path}


def invoke(job, args=(), *, overrides=None, remove=(), script=SCRIPT):
    env = {**job["env"], **(overrides or {})}
    for key in remove:
        env.pop(key, None)
    result = subprocess.run(["bash", str(script), *args], cwd=job["tmp"], env=env,
                            capture_output=True, text=True, timeout=30)
    record = json.loads(job["capture"].read_text()) if job["capture"].is_file() else None
    return result, record


def value_after(arguments, flag):
    return arguments[arguments.index(flag) + 1]


def test_slurm_resources_and_syntax():
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    text = SCRIPT.read_text()
    directives = {line.removeprefix("#SBATCH ").strip() for line in text.splitlines()
                  if line.startswith("#SBATCH ")}
    assert {
        "--account=xiangren_1987", "--partition=gpu", "--gpus-per-task=l40s:1",
        "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--mem=64G", "--time=02:00:00",
        "--output=results/slurm/e1-%j.out", "--error=results/slurm/e1-%j.err",
    } <= directives
    assert "/project2/xiangren_1987/group_37" in text


def test_default_run_uses_prepared_environment_and_locked_e1_controls(job):
    result, record = invoke(job)
    assert result.returncode == 0, result.stdout + result.stderr
    args = record["argv"]
    assert args[0] == str(job["environment"] / "bin/python")
    assert "-u" in args
    runner = Path(next(arg for arg in args if arg.endswith("scripts/run_e1_test.py")))
    assert (Path(record["cwd"]) / runner).resolve() == (ROOT / "scripts/run_e1_test.py").resolve()
    assert record["cwd"] == str(job["repo"])
    assert value_after(args, "--num-samples") == "2"
    assert value_after(args, "--batch-size") == "4"
    assert args[args.index("--budgets") + 1:args.index("--budgets") + 3] == ["256", "512"]
    assert args[args.index("--methods") + 1:args.index("--methods") + 3] == ["medal", "ab_medal"]
    assert value_after(args, "--max-depth") == "10"
    assert value_after(args, "--num-candidates") == "3"
    assert value_after(args, "--device") == "cuda"
    assert value_after(args, "--model-path") == str(job["model"])
    assert "--local-files-only" in args
    assert "--profile-resources" in args
    assert "--no-resume" not in args and "--no-e3" not in args
    assert value_after(args, "--output-dir") == str(job["repo"] / "results/e1_e3_test_slurm")


def test_offline_and_runtime_cache_exports_preserve_model_and_data(job):
    result, record = invoke(job, overrides={
        "HF_HOME": str(job["tmp"] / "stale-home"),
        "HF_MODULES_CACHE": str(job["tmp"] / "stale-modules"),
        "TRANSFORMERS_CACHE": str(job["tmp"] / "stale-transformers"),
        "HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0",
    })
    assert result.returncode == 0, result.stdout + result.stderr
    env = record["env"]
    runtime = job["repo"] / ".cache/carc/huggingface"
    assert env["HF_HOME"] == str(runtime)
    assert env["HF_MODULES_CACHE"] == str(runtime / "modules")
    assert env["TRANSFORMERS_CACHE"] == str(runtime / "hub")
    assert env["HF_HUB_CACHE"] == str(job["model"] / "huggingface/hub")
    assert env["HF_DATASETS_CACHE"] == str(job["data"])
    assert all(env[name] == "1" for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"))
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert str(job["repo"] / "src") in env["PYTHONPATH"].split(os.pathsep)
    assert not job["data"].exists()
    assert sorted(path.name for path in job["model"].iterdir()) == ["config.json"]


@pytest.mark.parametrize("relative", [
    Path(MODEL_NAME), Path("models") / MODEL_NAME,
    Path(HUB_REPO) / "snapshots" / REVISION,
    Path("hub") / HUB_REPO / "snapshots" / REVISION,
    Path("huggingface/hub") / HUB_REPO / "snapshots" / REVISION,
    Path("snapshots") / REVISION,
])
def test_job_resolves_existing_model_layouts(job, relative):
    cache = job["tmp"] / "model cache root"
    expected = snapshot(cache / relative)
    result, record = invoke(job, overrides={"MODEL_DIR": str(cache)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert value_after(record["argv"], "--model-path") == str(expected)


def test_explicit_model_and_output_environment_overrides(job):
    model = snapshot(job["tmp"] / "selected snapshot")
    artifact = job["tmp"] / "selected output"
    result, record = invoke(job, ["12", "2"], overrides={
        "ENV_DIR": str(job["environment"]), "DLM_ENV": str(job["tmp"] / "wrong-env"),
        "MODEL_PATH": str(model), "RESULTS_DIR": str(job["tmp"] / "other-results"),
        "ARTIFACT_ROOT": str(artifact), "HF_HUB_CACHE": str(job["tmp"] / "separate-hub"),
        "NUM_SAMPLES": "99", "BATCH_SIZE": "8",
    })
    assert result.returncode == 0, result.stdout + result.stderr
    args = record["argv"]
    assert value_after(args, "--num-samples") == "12"
    assert value_after(args, "--batch-size") == "2"
    assert value_after(args, "--model-path") == str(model)
    assert value_after(args, "--output-dir") == str(artifact)
    assert record["env"]["HF_HUB_CACHE"] == str(job["tmp"] / "separate-hub")


def test_environment_sample_controls_and_results_root(job):
    results = job["tmp"] / "results root"
    result, record = invoke(job, overrides={"NUM_SAMPLES": "200", "BATCH_SIZE": "1",
                                           "RESULTS_DIR": str(results)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert value_after(record["argv"], "--num-samples") == "200"
    assert value_after(record["argv"], "--batch-size") == "1"
    assert value_after(record["argv"], "--output-dir") == str(results / "e1_e3_test_slurm")


def test_invalid_explicit_model_does_not_fall_back_or_launch_srun(job):
    result, record = invoke(job, overrides={"MODEL_PATH": str(job["tmp"] / "missing snapshot")})
    assert result.returncode != 0
    assert record is None
    assert "MODEL_PATH" in result.stdout + result.stderr


def test_wrong_cached_revision_does_not_launch_srun(job):
    cache = job["tmp"] / "wrong revision cache"
    snapshot(cache / HUB_REPO / "snapshots" / ("f" * 40))
    result, record = invoke(job, overrides={"MODEL_DIR": str(cache)})
    assert result.returncode != 0
    assert record is None


def test_spooled_script_uses_slurm_submit_directory(job):
    spool = job["tmp"] / "slurm spool"
    spool.mkdir()
    script = spool / "slurm_script"
    shutil.copyfile(SCRIPT, script)
    result, record = invoke(job, script=script, remove=("REPO_DIR",),
                            overrides={"SLURM_SUBMIT_DIR": str(job["repo"]), "SLURM_JOB_ID": "123"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert record["cwd"] == str(job["repo"])


def test_explicit_repository_wins_over_submit_directory(job):
    result, record = invoke(job, overrides={"SLURM_SUBMIT_DIR": str(job["tmp"] / "wrong-directory")})
    assert result.returncode == 0, result.stdout + result.stderr
    assert record["cwd"] == str(job["repo"])


def test_bash_on_allocated_node_uses_script_parent(job):
    # Copy rather than symlink so BASH_SOURCE identifies the temporary checkout.
    repository = job["tmp"] / "bash checkout"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    for filename in ("run_carc_e1_test.sbatch", "carc_paths.py", "run_e1_test.py"):
        shutil.copyfile(ROOT / "scripts" / filename, scripts / filename)
    (repository / "src").symlink_to(ROOT / "src", target_is_directory=True)
    result, record = invoke(job, script=scripts / SCRIPT.name, remove=("REPO_DIR",))
    assert result.returncode == 0, result.stdout + result.stderr
    assert record["cwd"] == str(repository)


def test_direct_execution_without_allocation_does_not_start_inference(job):
    result, record = invoke(job, remove=("SLURM_JOB_ID",))
    assert result.returncode != 0
    assert record is None
    assert "sbatch" in result.stdout + result.stderr


@pytest.mark.parametrize("arguments", [
    ["0"], ["501"], ["-1"], ["two"], ["1", "0"], ["1", "-1"],
    ["1", "2.5"], ["1", "four"], ["1", "501"], ["01"], ["1", "4", "extra"],
])
def test_invalid_arguments_fail_before_launch(job, arguments):
    result, record = invoke(job, arguments)
    assert result.returncode != 0
    assert record is None
