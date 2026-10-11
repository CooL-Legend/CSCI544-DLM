"""Full test manifests, shared profiles, benchmark metrics, and four CLIs."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from dlm_project.experiments.config import load_config
from dlm_project.experiments.manifests import digest
from dlm_project.models.profiles import get_profile
from dlm_project.paths import PROJECT_ROOT as ROOT
from dlm_project.tasks.gsm8k import benchmark, benchmark_dataset
from tests.decoding.helpers import FakeModel, MASK, PROMPT


def manifest():
    examples = []
    for index in range(benchmark_dataset.TEST_SIZE):
        content = {"question": f"How many items are there in case {index}?",
                   "reference_solution": "REFERENCE_ONLY: hidden solution\n#### 1"}
        examples.append({"row_id": index, "example_id": f"test:{index}",
                         **content, "content_hash": digest(content)})
    return {"version": 1, "kind": "gsm8k_test_benchmark",
            "dataset": {"repo_id": "openai/gsm8k", "subset": "main", "split": "test",
                        "revision": "a" * 40, "fingerprint": "controlled"}, "examples": examples}


def config_for(tmp_path):
    config = load_config(ROOT / "configs/gsm8k/decoding.yaml")
    config["experiment_root"] = str(tmp_path)
    config["data"]["num_samples"] = 2
    config["model"].update(device="cpu", dtype="float32")
    config["generation"].update(gen_length=4, steps=4, block_length=2)
    return config


def wrapper(texts=None):
    prompts = []

    class Tokenizer:
        def batch_decode(self, *args, **kwargs):
            return [next(texts) if texts is not None else "#### 1"]

    def encode(prompt):
        prompts.append(prompt)
        return PROMPT.clone()

    profile = get_profile("llada")
    return SimpleNamespace(device="cpu", mask_id=MASK, model=FakeModel(),
                           identity={"repo_id": profile["repo_id"], "revision": profile["revision"],
                                     "controlled_test_model": True},
                           tokenizer=Tokenizer(), encode_prompt=encode), prompts


@pytest.mark.parametrize("method", tuple(benchmark.DECODERS))
def test_benchmark_one_load_reference_isolation_and_cost(tmp_path, monkeypatch, method):
    frozen = manifest()
    monkeypatch.setattr(benchmark, "get_manifest", lambda *a, **k: frozen)
    instance, prompts = wrapper()
    loads = []

    def factory(config):
        loads.append(config)
        return instance

    source = config_for(tmp_path)
    before = copy.deepcopy(source)
    directory, metrics = benchmark.run_evaluation(method, source, model_factory=factory)
    assert source == before
    assert len(loads) == metrics["model_load_count"] == 1
    assert metrics["status"] == "completed"
    assert metrics["accuracy"] == 1 and metrics["correct"] == metrics["processed"] == 2
    assert metrics["parse_failures"] == 0 and metrics["questions_per_second"] > 0
    forwards = 20 if method == "best_of_5" else 4
    assert metrics["mean_model_calls"] == forwards
    assert metrics["total_model_calls"] == len(instance.model.seen) == 2 * forwards
    records = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    assert [row["example_id"] for row in records] == ["test:0", "test:1"]
    assert [row["seed"] for row in records] == [1, 6]
    assert all("REFERENCE_ONLY" not in prompt for prompt in prompts)
    assert all(row["reference_solution"].startswith("REFERENCE_ONLY") for row in records)
    assert all(row["dataset_revision"] == "a" * 40 for row in records)
    assert all((directory / row["trace_file"]).is_file() for row in records)
    assert all(row["peak_gpu_memory_bytes"] is None for row in records)
    assert method in directory.parts and "smoke" in directory.parts
    saved = json.loads((directory / "config.json").read_text())
    assert saved["run"]["manifest_hash"] == digest(frozen)
    assert saved["provenance"]["model"] == instance.identity
    if method == "best_of_5":
        assert len(records[0]["candidates"]) == 5
        assert records[0]["votes"] == {"1": 5}
        assert records[0]["candidate_seeds"] == [1, 2, 3, 4, 5]


def test_accuracy_includes_wrong_and_unparseable_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "get_manifest", lambda *a, **k: manifest())
    instance, _ = wrapper(iter(["bad", "#### 2", "#### 1.0"]))
    config = config_for(tmp_path)
    config["data"]["num_samples"] = 3
    _, metrics = benchmark.run_evaluation("confidence", config, model_factory=lambda _: instance)
    assert metrics["processed"] == 3 and metrics["correct"] == 1
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["parse_failures"] == 1


def test_interruption_keeps_completed_prediction_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "get_manifest", lambda *a, **k: manifest())
    original = benchmark.DECODERS["random_mdm"]
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("controlled failure")
        return original(*args, **kwargs)

    monkeypatch.setitem(benchmark.DECODERS, "random_mdm", interrupted)
    instance, prompts = wrapper()
    with pytest.raises(RuntimeError, match="controlled failure"):
        benchmark.run_evaluation("random_mdm", config_for(tmp_path), model_factory=lambda _: instance)
    directory = next(tmp_path.rglob("metrics.json")).parent
    assert len((directory / "predictions.jsonl").read_text().splitlines()) == 1
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["status"] == "failed" and metrics["processed"] == 1
    assert metrics["requested_examples"] == 2 and metrics["accuracy"] == 1
    assert "controlled failure" in metrics["error"]
    assert all("REFERENCE_ONLY" not in prompt for prompt in prompts)


@pytest.mark.parametrize("profile,count", [("smoke", 20), ("initial", 100), ("full", 1319)])
def test_profiles_select_same_test_prefix(tmp_path, profile, count):
    source = config_for(tmp_path)
    source["data"]["num_samples"] = None
    config = benchmark.resolve_settings("confidence", source, profile=profile)
    assert config["data"]["num_samples"] == count
    examples = benchmark_dataset.select_examples(manifest(), profile)
    assert len(examples) == count
    assert [row["row_id"] for row in examples] == list(range(count))
    assert config["generation"]["temperature"] == 1


def test_manifest_is_pinned_and_reused_offline(tmp_path, monkeypatch):
    import datasets
    import huggingface_hub

    rows = [{"question": row["question"], "answer": row["reference_solution"]}
            for row in manifest()["examples"]]
    calls = []

    def loader(*args, **kwargs):
        calls.append((args, kwargs))
        return rows

    monkeypatch.setattr(datasets, "load_dataset", loader)
    monkeypatch.setattr(huggingface_hub.HfApi, "dataset_info", lambda *a: SimpleNamespace(sha="a" * 40))
    path = tmp_path / "shared.json"
    created = benchmark_dataset.get_manifest(path, cache_dir="cache")
    assert calls == [(("openai/gsm8k", "main"), {"split": "test", "revision": "a" * 40, "cache_dir": "cache"})]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: pytest.fail("Dataset loaded despite cached manifest"))
    monkeypatch.setattr(huggingface_hub.HfApi, "dataset_info", lambda *a: pytest.fail("Hub accessed offline"))
    assert benchmark_dataset.get_manifest(path) == created
    with pytest.raises(ValueError, match="differs"):
        benchmark_dataset.get_manifest(path, revision="b" * 40)
    assert not list(tmp_path.glob("shared.json.*"))


@pytest.mark.parametrize("corruption", ["content", "order", "count", "reference", "revision"])
def test_manifest_validation_detects_corruption(corruption):
    data = manifest()
    if corruption == "content":
        data["examples"][0]["question"] = "changed"
    elif corruption == "order":
        data["examples"][0]["row_id"] = 1
    elif corruption == "count":
        data["examples"].pop()
    elif corruption == "reference":
        row = data["examples"][0]
        row["reference_solution"] = "bad"
        row["content_hash"] = digest({key: row[key] for key in ("question", "reference_solution")})
    else:
        data["dataset"]["revision"] = "main"
    with pytest.raises(ValueError):
        benchmark_dataset.validate_manifest(data)


def test_cli_applies_overrides_without_loading_model(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(benchmark, "run_evaluation", lambda *a, **k: captured.append((a, k)))
    monkeypatch.setattr(sys, "argv", ["script", "--profile", "initial", "--num-samples", "10",
                                     "--seed", "17", "--temperature", "0", "--model-path", str(tmp_path),
                                     "--local-files-only", "--manifest", str(tmp_path / "manifest.json"),
                                     "--output-dir", str(tmp_path / "output")])
    benchmark.main("confidence")
    args, kwargs = captured[0]
    config = args[1]
    assert config["evaluation"]["profile"] == "initial" and config["data"]["num_samples"] == 10
    assert config["decoding"]["seed"] == 17 and config["generation"]["temperature"] == 0
    assert config["model"]["model_path"] == str(tmp_path) and config["model"]["local_files_only"] is True
    assert kwargs["manifest_path"] == tmp_path / "manifest.json"


@pytest.mark.parametrize("overrides", [
    {"profile": "wrong"}, {"num_samples": 0}, {"num_samples": True},
    {"num_samples": 1320}, {"seed": -1}, {"temperature": float("nan")},
])
def test_invalid_benchmark_settings(tmp_path, overrides):
    with pytest.raises(ValueError):
        benchmark.resolve_settings("confidence", config_for(tmp_path), **overrides)


def test_unpinned_model_fails_without_inference_and_records_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "get_manifest", lambda *a, **k: manifest())
    instance, _ = wrapper()
    instance.identity["revision"] = "untrusted"
    with pytest.raises(ValueError, match="pinned"):
        benchmark.run_evaluation("confidence", config_for(tmp_path), model_factory=lambda _: instance)
    directory = next(tmp_path.rglob("metrics.json")).parent
    assert not instance.model.seen
    assert (directory / "predictions.jsonl").read_text() == ""
    assert json.loads((directory / "metrics.json").read_text())["status"] == "failed"
