"""Offline launcher wiring, paired condition order and subset reports."""

import json
import os

import pytest
import yaml

from scripts import run_gsm8k_experiments as launcher
from dlm_project.tasks.gsm8k import experiment_0 as e0
from dlm_project.tasks.gsm8k.e0_report import generate_report
from tests.tasks.test_e0_dataset import manifest
from tests.workflows.test_experiment_0 import config, wrapper


@pytest.mark.parametrize("method,conditions", [
    ("both", ["confidence_t0", "medal_budget_b8"]),
    ("llada", ["confidence_t0"]),
    ("mcts", ["medal_budget_b8"]),
])
def test_launcher_runs_only_requested_methods_offline_with_shared_inputs(tmp_path, monkeypatch, method, conditions):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    profile = launcher.get_profile()
    hub = tmp_path / "model cache"
    snapshot = hub / ("models--" + profile["repo_id"].replace("/", "--")) / "snapshots" / profile["revision"]
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    data = tmp_path / "datasets"
    data.mkdir()
    frozen = manifest()
    manifest_path = data / "gsm8k_e0.json"
    manifest_path.write_text(json.dumps(frozen))
    env = tmp_path / ".env"
    env.write_text(
        'export MODEL_CACHE_DIR="model cache" # paths are relative to .env\n'
        'DATASET_DIR=datasets\nRESULTS_DIR=results\n'
        'DATASET_MANIFEST=datasets/gsm8k_e0.json\n'
    )
    source = tmp_path / "config.yaml"
    source.write_text(yaml.safe_dump(config(tmp_path)))
    instance, prompts = wrapper()
    loads = []

    def load(settings):
        loads.append(settings)
        assert settings["model"]["local_files_only"] is True
        assert settings["model"]["model_path"] == profile["repo_id"]
        assert settings["data"]["cache_dir"] == str(data)
        assert os.environ["HF_HUB_CACHE"] == str(hub)
        assert os.environ["HF_DATASETS_CACHE"] == str(data)
        assert all(os.environ[key] == "1" for key in (
            "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"))
        return instance

    monkeypatch.setattr(e0, "_load_model", load)
    for unused in ("entropy", "medal_no_search"):
        monkeypatch.setitem(e0.DECODERS, unused, lambda *a, **k: pytest.fail("Unrequested decoder ran"))
    launcher.main([
        "--env-file", str(env), "--config", str(source), "--method", method,
        "--stage", "test", "--num-samples", "2", "--search-budget", "8",
        "--lc", "2", "--num-candidates", "1", "--top-k-tokens", "2", "--top-k-actions", "2",
    ])
    assert len(loads) == 1
    prediction_files = list((tmp_path / "results").rglob("predictions.jsonl"))
    assert len(prediction_files) == 1
    directory = prediction_files[0].parent
    rows = [json.loads(line) for line in prediction_files[0].read_text().splitlines()]
    expected_ids = [row["example_id"] for row in frozen["examples"] if row["stage"] == "test"][:2]
    assert [(row["condition"], row["example_id"]) for row in rows] == [
        (condition, example) for condition in conditions for example in expected_ids]
    assert len(prompts) == 3  # One warm-up and one tokenization per question.
    for example in expected_ids:
        paired = [row for row in rows if row["example_id"] == example]
        assert len({row["prompt_ids_hash"] for row in paired}) == 1
        assert len({row["seed"] for row in paired}) == 1
    for row in rows:
        assert row["temperature"] == 0.0
        assert row["completion_nfes"] == 8 - row["selected_depth"]
        assert row["search_nfes"] <= 8
        if row["method"] == "medal":
            assert row["completion_method"] == "llada"
            settings = row["details"]["mcts_config"]
            assert settings["stop_mode"] == "budget"
            assert settings["num_candidates"] == 1
            assert settings["top_k_tokens"] == settings["top_k_actions"] == 2
        else:
            assert row["search_nfes"] == 0
    summary = generate_report(directory)  # Also works without a no-search condition.
    assert summary["complete"] and summary["common_n"] == 2
    assert [condition["id"] for condition in summary["conditions"]] == conditions
    assert len(summary["gains"]) == (1 if method == "both" else 0)


def test_dry_run_does_not_load_model_dataset_or_create_results(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    env = tmp_path / ".env"
    env.write_text("MODEL_CACHE_DIR=model\nDATASET_DIR=data\nRESULTS_DIR=results\n")
    monkeypatch.setattr(e0, "run_experiment", lambda *a, **k: pytest.fail("Dry run executed experiment"))
    launcher.main(["--env-file", str(env), "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert [row["id"] for row in plan["conditions"]] == ["confidence_t0", "medal_budget_b256"]
    assert plan["decoding"]["completion_method"] == "llada"
    assert plan["decoding"]["mcts"]["init_tokens"] == 5
    assert not (tmp_path / "results").exists()


def test_env_paths_accept_parent_cache_and_override_old_shell_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    hub = tmp_path / "model-cache/huggingface/hub"
    hub.mkdir(parents=True)
    env = tmp_path / ".env"
    env.write_text("MODEL_CACHE_DIR=model-cache\nDATASET_DIR=data\nRESULTS_DIR=results\n"
                   "DATASET_REVISION=" + "a" * 40 + "\n")
    paths, revision = launcher.env_paths(env)
    assert paths["MODEL_CACHE_DIR"] == hub
    assert paths["DATASET_MANIFEST"] == tmp_path / "data/manifests/gsm8k_e0.json"
    assert revision == "a" * 40
    for key in ("HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE",
                "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE"):
        monkeypatch.setenv(key, "/old/cache")
    launcher.configure_offline(paths)
    assert os.environ["HF_HUB_CACHE"] == str(hub)
    assert os.environ["HF_DATASETS_CACHE"] == str(tmp_path / "data")
    assert "TRANSFORMERS_CACHE" not in os.environ
    assert "PYTORCH_TRANSFORMERS_CACHE" not in os.environ
    assert "PYTORCH_PRETRAINED_BERT_CACHE" not in os.environ


@pytest.mark.parametrize("failure", ["model", "dataset", "manifest"])
def test_missing_local_inputs_fail_before_experiment(tmp_path, monkeypatch, capsys, failure):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    env = tmp_path / ".env"
    env.write_text("MODEL_CACHE_DIR=hub\nDATASET_DIR=data\nRESULTS_DIR=results\n")
    if failure != "model":
        profile = launcher.get_profile()
        snapshot = (tmp_path / "hub" / ("models--" + profile["repo_id"].replace("/", "--"))
                    / "snapshots" / profile["revision"])
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
    if failure == "manifest":
        (tmp_path / "data").mkdir()
    monkeypatch.setattr(e0, "run_experiment", lambda *a, **k: pytest.fail("Missing input was ignored"))
    with pytest.raises(SystemExit) as error:
        launcher.main(["--env-file", str(env)])
    assert error.value.code == 2
    expected = {"model": "Local LLaDA snapshot not found", "dataset": "Dataset directory does not exist",
                "manifest": "No E0 manifest found"}
    assert expected[failure] in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("methods", [[], ["unknown"], ["confidence", "confidence"], "confidence"])
def test_e0_rejects_invalid_method_selection(tmp_path, methods):
    with pytest.raises(ValueError, match="methods"):
        e0.resolve_settings(config(tmp_path), methods=methods)
