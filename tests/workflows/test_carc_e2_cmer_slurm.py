"""Exercise the CMER launcher with prepared local assets and a fake scheduler."""

from pathlib import Path
import shutil
import subprocess

import pytest

from tests.workflows.test_carc_e2_swd_slurm import ROOT, invoke, job, value_after


SCRIPT = ROOT / "scripts/run_carc_e2_cmer_test.sbatch"


def test_cmer_defaults_dispatch_500_questions_offline_for_48_hours(job):
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True).returncode == 0
    directives = {line.removeprefix("#SBATCH ") for line in SCRIPT.read_text().splitlines()
                  if line.startswith("#SBATCH ")}
    assert {
        "--account=xiangren_1987", "--partition=gpu", "--gpus-per-task=a40:1",
        "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--mem=64G", "--time=48:00:00",
        "--output=results/slurm/e2-cmer-%j.out", "--error=results/slurm/e2-cmer-%j.err",
    } <= directives
    for location in ("envs/dlm", "model-cache", "datasets"):
        assert f"/project2/xiangren_1987/group_37/{location}" in SCRIPT.read_text()

    result, record = invoke(job, script=SCRIPT)
    assert result.returncode == 0, result.stdout + result.stderr
    args, env = record["argv"], record["env"]
    assert args[:3] == [str(job["environment"] / "bin/python"), "-u",
                        str(job["repo"] / "scripts/run_e2.py")]
    expected = {
        "--stage": "test", "--num-samples": "500", "--batch-size": "8",
        "--reward-mode": "cmer", "--reward-lambda": "1.0", "--sim-cap-multiplier": "4",
        "--max-depth": "10", "--num-candidates": "3", "--device": "cuda",
        "--model-path": str(job["model"]),
        "--output-dir": str(job["repo"] / "results/e2_cmer_test_500"),
    }
    for flag, value in expected.items():
        assert value_after(args, flag) == value
    assert args[args.index("--methods") + 1:args.index("--methods") + 3] == ["medal", "ab_medal"]
    assert args[args.index("--budgets") + 1:args.index("--budgets") + 3] == ["256", "512"]
    assert {"--run-e3", "--profile-resources", "--local-files-only"} <= set(args)
    assert "--no-resume" not in args and "--reward-beta" not in args
    assert all(env[k] == "1" for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"))
    assert env["TRANSFORMERS_CACHE"] == str(job["repo"] / ".cache/carc/huggingface/hub")
    assert env["HF_MODULES_CACHE"] == str(job["repo"] / ".cache/carc/huggingface/modules")
    assert env["HF_DATASETS_CACHE"] == job["env"]["HF_DATASETS_CACHE"]
    assert not Path(env["HF_DATASETS_CACHE"]).exists()
    assert sorted(p.name for p in job["model"].iterdir()) == ["config.json"]


def test_cmer_smoke_uses_independent_output_and_explicit_model_override(job):
    snapshot = job["tmp"] / "explicit snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    output = job["tmp"] / "cmer smoke results"
    result, record = invoke(job, ["2", "2"], script=SCRIPT, overrides={
        "ARTIFACT_ROOT": str(output), "MODEL_PATH": str(snapshot),
        "NUM_SAMPLES": "500", "BATCH_SIZE": "8",
    })
    assert result.returncode == 0, result.stdout + result.stderr
    assert value_after(record["argv"], "--num-samples") == "2"
    assert value_after(record["argv"], "--batch-size") == "2"
    assert value_after(record["argv"], "--output-dir") == str(output)
    assert value_after(record["argv"], "--model-path") == str(snapshot)


def test_cmer_spooled_script_finds_the_submission_checkout(job):
    spool_script = job["tmp"] / "slurm_script"
    shutil.copyfile(SCRIPT, spool_script)
    result, record = invoke(job, script=spool_script, remove=("REPO_DIR",),
                            overrides={"SLURM_SUBMIT_DIR": str(job["repo"])})
    assert result.returncode == 0, result.stdout + result.stderr
    assert record["cwd"] == str(job["repo"])
    assert value_after(record["argv"], "--reward-mode") == "cmer"


@pytest.mark.parametrize("arguments", [["0"], ["501"], ["2", "0"], ["2", "eight"], ["2", "8", "extra"]])
def test_cmer_rejects_invalid_arguments_before_dispatch(job, arguments):
    result, record = invoke(job, arguments, script=SCRIPT)
    assert result.returncode == 2
    assert record is None


@pytest.mark.parametrize("remove,overrides", [
    (("SLURM_JOB_ID",), {}),
    ((), {"ENV_DIR": "/missing-prepared-environment"}),
    ((), {"MODEL_PATH": "/missing-local-snapshot"}),
])
def test_cmer_requires_allocation_and_prepared_assets(job, remove, overrides):
    result, record = invoke(job, script=SCRIPT, remove=remove, overrides=overrides)
    assert result.returncode != 0
    assert record is None
