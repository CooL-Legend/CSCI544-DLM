"""Full test manifests, shared profiles, benchmark metrics, and decoder CLIs."""

import copy
from concurrent.futures import ThreadPoolExecutor
import json
import runpy
import sys
from threading import Barrier
from types import SimpleNamespace

import pytest
import torch

from dlm_project.experiments.config import load_config
from dlm_project.experiments.manifests import digest
from dlm_project.models.profiles import get_profile
from dlm_project.paths import PROJECT_ROOT as ROOT
from dlm_project.tasks.gsm8k import benchmark, benchmark_dataset
from tests.decoding.helpers import FakeModel, MASK, PROMPT


@pytest.fixture(autouse=True)
def decomposition_fixture(monkeypatch):
    def demonstrations(*a, revision, **kw):
        examples = []
        for i in (0,1):
            content = {"question": f"Training example {i}: what is one plus one?",
                       "reference_solution": "1 + 1 = 2.\n#### 2"}
            examples.append({"row_id": i, **content, "content_hash": digest(content)})
        return {"version": "medal_inline_v1", "dataset": {"repo_id": "openai/gsm8k", "subset": "main",
                "split": "train", "revision": revision}, "examples": examples}
    monkeypatch.setattr(benchmark, "get_gsm8k_demonstrations", demonstrations)

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


def config_for(tmp_path, family="llada"):
    filename = "illada_decoding.yaml" if family == "illada" else "decoding.yaml"
    config = load_config(ROOT / "configs/gsm8k" / filename)
    config["experiment_root"] = str(tmp_path)
    config["data"]["num_samples"] = 2
    config["model"].update(device="cpu", dtype="float32")
    config["generation"].update(gen_length=4, steps=4, block_length=4)
    config["decoding"]["mcts"] = {"init_tokens": 2, "num_candidates": 1,
                                  "top_k_tokens": 1, "top_k_actions": 1,
                                  "max_model_calls": 8}
    return config


def wrapper(texts=None, family="llada"):
    prompts = []
    mask_id = 5 if family == "illada" else MASK
    prompt_ids = torch.tensor([[3, 5]]) if family == "illada" else PROMPT
    model = FakeModel(lambda _: [.05, .6, .2, .1, .05, 0.]) if family == "illada" else FakeModel()

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return " ".join(str(token) for token in ids)

        def batch_decode(self, *args, **kwargs):
            return [next(texts) if texts is not None else "#### 1"]

    def encode(prompt):
        prompts.append(prompt)
        return prompt_ids.clone()

    profile = get_profile(family)
    return SimpleNamespace(device="cpu", mask_id=mask_id, model=model, family=family,
                           identity={"repo_id": profile["repo_id"], "revision": profile["revision"],
                                     "family": family, "mask_id": mask_id,
                                     "controlled_test_model": True},
                           tokenizer=Tokenizer(), encode_prompt=encode), prompts


@pytest.mark.parametrize("method", tuple(benchmark.DECODERS))
@pytest.mark.parametrize("family", benchmark.MODEL_FAMILIES)
def test_benchmark_one_load_reference_isolation_and_cost(tmp_path, monkeypatch, method, family):
    frozen = manifest()
    monkeypatch.setattr(benchmark, "get_manifest", lambda *a, **k: frozen)
    instance, prompts = wrapper(family=family)
    loads = []

    def factory(config):
        loads.append(config)
        return instance

    source = config_for(tmp_path, family)
    before = copy.deepcopy(source)
    directory, metrics = benchmark.run_evaluation(method, source, model_factory=factory)
    assert source == before
    assert len(loads) == metrics["model_load_count"] == 1
    assert metrics["status"] == "completed"
    assert metrics["model_family"] == family
    assert metrics["accuracy"] == 1 and metrics["correct"] == metrics["processed"] == 2
    assert metrics["parse_failures"] == 0 and metrics["questions_per_second"] > 0
    forwards = {"best_of_5": 20, "mcts_confidence": 5,
                "mcts_swd_puct": 7, "mcts_swd_reward": 7,
                "mcts_swd_adaptive": 6}.get(method, 4)
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
    assert all(row["model_family"] == family for row in records)
    assert all(row["mask_token_id"] == instance.mask_id for row in records)
    if family == "illada":
        assert all(torch.equal(state[:, :2], torch.tensor([[3, 5]])) for state in instance.model.seen)
    assert method in directory.parts and "smoke" in directory.parts
    assert get_profile(family)["label"] in directory.parts
    saved = json.loads((directory / "config.json").read_text())
    assert saved["run"]["manifest_hash"] == digest(frozen)
    assert saved["provenance"]["model"] == instance.identity
    if method == "best_of_5":
        assert len(records[0]["candidates"]) == 5
        assert records[0]["votes"] == {"1": 5}
        assert records[0]["candidate_seeds"] == [1, 2, 3, 4, 5]
    elif method in benchmark.MCTS_METHODS:
        expected_search = 3 if method in ("mcts_confidence", "mcts_swd_adaptive") else 5
        expected_depth = 1 if method == "mcts_swd_adaptive" else 2
        expected_completion = 4 - expected_depth
        assert metrics["total_search_model_calls"] == 2 * expected_search
        assert metrics["total_completion_model_calls"] == 2 * expected_completion
        assert metrics["fallback_count"] == 0 and metrics["mean_selected_depth"] == expected_depth
        assert metrics["search_latency_seconds"] >= 0 and metrics["completion_latency_seconds"] >= 0
        assert all(row["search_model_calls"] == expected_search
                   and row["completion_model_calls"] == expected_completion
                   for row in records)
        expected_version = "medal-v2" if method == "mcts_confidence" else "medal-v3"
        assert expected_version in directory.parent.name
        assert "lc2-c1-k1_1-k2_1-budget8" in directory.parent.name
        assert saved["decoding"]["mcts"]["init_tokens"] == 2
        trace = [json.loads(line) for line in (directory / records[0]["trace_file"]).read_text().splitlines()]
        assert {event.get("phase") for event in trace} >= {"tree_analysis", "simulation", "candidate_evaluation", "completion"}
        if method != "mcts_confidence":
            assert "stability_reference" in {event.get("phase") for event in trace}


def test_accuracy_includes_wrong_and_unparseable_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "get_manifest", lambda *a, **k: manifest())
    instance, _ = wrapper(iter(["bad", "#### 2", "#### 1.0"]))
    config = config_for(tmp_path)
    config["data"]["num_samples"] = 3
    _, metrics = benchmark.run_evaluation("confidence", config, model_factory=lambda _: instance)
    assert metrics["processed"] == 3 and metrics["correct"] == 1
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["parse_failures"] == 1


@pytest.mark.parametrize("method", ["random_mdm", "mcts_confidence"])
def test_interruption_keeps_completed_prediction_and_metrics(tmp_path, monkeypatch, method):
    monkeypatch.setattr(benchmark, "get_manifest", lambda *a, **k: manifest())
    original = benchmark.DECODERS[method]
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("controlled failure")
        return original(*args, **kwargs)

    monkeypatch.setitem(benchmark.DECODERS, method, interrupted)
    instance, prompts = wrapper()
    with pytest.raises(RuntimeError, match="controlled failure"):
        benchmark.run_evaluation(method, config_for(tmp_path), model_factory=lambda _: instance)
    directory = next(tmp_path.rglob("metrics.json")).parent
    assert len((directory / "predictions.jsonl").read_text().splitlines()) == 1
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["status"] == "failed" and metrics["processed"] == 1
    assert metrics["requested_examples"] == 2 and metrics["accuracy"] == 1
    assert "controlled failure" in metrics["error"]
    assert all("REFERENCE_ONLY" not in prompt for prompt in prompts)
    if method == "mcts_confidence":
        assert metrics["total_search_model_calls"] == 3
        assert metrics["total_completion_model_calls"] == 2
        assert metrics["total_model_calls"] == len(instance.model.seen) == 5


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


def test_manifest_publication_works_without_hard_links(tmp_path, monkeypatch):
    import datasets

    frozen = manifest()
    rows = [{"question": row["question"], "answer": row["reference_solution"]}
            for row in frozen["examples"]]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: rows)

    def unsupported_link(*args, **kwargs):
        raise PermissionError(1, "Operation not permitted on mounted Google Drive")

    monkeypatch.setattr(benchmark_dataset.os, "link", unsupported_link)
    path = tmp_path / "gsm8k_test.json"
    created = benchmark_dataset.get_manifest(path, revision="a" * 40)
    assert created == json.loads(path.read_text())
    assert created["dataset"]["revision"] == "a" * 40
    assert not list(tmp_path.glob("gsm8k_test.json.*"))


def test_concurrent_manifest_writers_reuse_first_snapshot(tmp_path, monkeypatch):
    import datasets
    import huggingface_hub

    rows = [{"question": row["question"], "answer": row["reference_solution"]}
            for row in manifest()["examples"]]
    barrier = Barrier(2)
    revisions = iter(["a" * 40, "b" * 40])
    monkeypatch.setattr(huggingface_hub.HfApi, "dataset_info",
                        lambda *a: SimpleNamespace(sha=next(revisions)))

    def loader(*args, **kwargs):
        barrier.wait(timeout=10)
        return rows

    monkeypatch.setattr(datasets, "load_dataset", loader)
    path = tmp_path / "shared.json"
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(benchmark_dataset.get_manifest, path) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert results[0] == results[1] == json.loads(path.read_text())
    assert not list(tmp_path.glob("shared.json.*"))


def test_failed_manifest_rename_cleans_up_files(tmp_path, monkeypatch):
    import datasets

    rows = [{"question": row["question"], "answer": row["reference_solution"]}
            for row in manifest()["examples"]]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: rows)

    def failed_rename(*args, **kwargs):
        raise OSError("controlled rename failure")

    monkeypatch.setattr(benchmark_dataset.os, "replace", failed_rename)
    path = tmp_path / "shared.json"
    with pytest.raises(OSError, match="controlled rename failure"):
        benchmark_dataset.get_manifest(path, revision="a" * 40)
    assert not path.exists()
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

    def run(*args, **kwargs):
        captured.append((args, kwargs))
        return tmp_path / "run", {"status": "completed"}

    monkeypatch.setattr(benchmark, "run_evaluation", run)
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


@pytest.mark.parametrize("family", benchmark.MODEL_FAMILIES)
@pytest.mark.parametrize("method", tuple(benchmark.DECODERS))
def test_cli_defaults_and_run_receipt(tmp_path, monkeypatch, family, method):
    captured = []
    directory = tmp_path / "exact-run"
    receipt = tmp_path / "receipts/run.json"

    def run(*args, **kwargs):
        captured.append((args, kwargs))
        return directory, {"status": "completed"}

    monkeypatch.setattr(benchmark, "run_evaluation", run)
    monkeypatch.setattr(sys, "argv", ["script", "--profile", "initial", "--run-manifest", str(receipt)])
    filename = "illada_decoding.yaml" if family == "illada" else "decoding.yaml"
    config_path = ROOT / "configs/gsm8k" / filename
    if method == "mcts_confidence":
        import yaml
        cfg = load_config(config_path)
        cfg["generation"]["block_length"] = cfg["generation"]["gen_length"]
        config_path = tmp_path / "medal.yaml"
        config_path.write_text(yaml.safe_dump(cfg))
    benchmark.main(method, default_config=config_path, expected_family=family)
    config = captured[0][0][1]
    assert config["model"]["family"] == family
    assert config["model"]["model_path"] == get_profile(family)["repo_id"]
    assert config["data"]["num_samples"] == 100
    assert json.loads(receipt.read_text()) == {
        "run_dir": str(directory), "status": "completed", "model_family": family,
        "method": method, "profile": "initial",
    }


@pytest.mark.parametrize("expected_family", benchmark.MODEL_FAMILIES)
def test_cli_rejects_other_model_family_before_loading(tmp_path, monkeypatch, capsys, expected_family):
    filename = "decoding.yaml" if expected_family == "illada" else "illada_decoding.yaml"
    monkeypatch.setattr(sys, "argv", ["script", "--config", str(ROOT / "configs/gsm8k" / filename)])
    monkeypatch.setattr(benchmark, "run_evaluation", lambda *a, **k: pytest.fail("Loaded the wrong family"))
    with pytest.raises(SystemExit) as error:
        benchmark.main("confidence", expected_family=expected_family)
    assert error.value.code == 2
    assert f"requires model.family: {expected_family}" in capsys.readouterr().err


def test_failed_evaluation_does_not_publish_run_receipt(tmp_path, monkeypatch):
    receipt = tmp_path / "run.json"

    def failed(*args, **kwargs):
        raise RuntimeError("controlled failure")

    monkeypatch.setattr(benchmark, "run_evaluation", failed)
    monkeypatch.setattr(sys, "argv", ["script", "--run-manifest", str(receipt)])
    with pytest.raises(RuntimeError, match="controlled failure"):
        benchmark.main("confidence")
    assert not receipt.exists()


@pytest.mark.parametrize("family", benchmark.MODEL_FAMILIES)
def test_family_specific_default_checkpoint(tmp_path, family):
    config = config_for(tmp_path, family)
    config["model"].pop("model_path")
    resolved = benchmark.resolve_settings("confidence", config)
    assert resolved["model"]["model_path"] == get_profile(family)["repo_id"]


@pytest.mark.parametrize("family", benchmark.MODEL_FAMILIES)
def test_checkpoint_verification_rejects_other_family(tmp_path, monkeypatch, family):
    monkeypatch.setattr(benchmark, "get_manifest", lambda *a, **k: manifest())
    other = "llada" if family == "illada" else "illada"
    instance, _ = wrapper(family=other)
    with pytest.raises(ValueError, match="pinned"):
        benchmark.run_evaluation("confidence", config_for(tmp_path, family), model_factory=lambda _: instance)
    assert not instance.model.seen


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


def test_mcts_and_confidence_reuse_shared_manifest_prompts_and_seeds(tmp_path):
    shared = tmp_path / "manifests/gsm8k_test.json"
    shared.parent.mkdir()
    shared.write_text(json.dumps(manifest()))
    runs = []
    for method in ("confidence", "mcts_confidence"):
        instance, prompts = wrapper()
        directory, _ = benchmark.run_evaluation(method, config_for(tmp_path), model_factory=lambda _: instance)
        config = json.loads((directory / "config.json").read_text())
        records = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
        runs.append((config, records, prompts))
    first, second = runs
    assert first[0]["run"]["manifest_hash"] == second[0]["run"]["manifest_hash"]
    assert first[0]["provenance"]["dataset"] == second[0]["provenance"]["dataset"]
    assert first[2] != second[2]
    assert [r["base_prompt"] for r in first[1]] == [r["base_prompt"] for r in second[1]]
    assert all(r["decomposition"]["demonstrations_dataset"]["split"] == "train" for r in second[1])
    assert [(row["example_id"], row["seed"], row["reference_answer"]) for row in first[1]] == [
        (row["example_id"], row["seed"], row["reference_answer"]) for row in second[1]]


def test_mcts_fallback_metrics_and_recomputed_completion_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "get_manifest", lambda *a, **k: manifest())
    source = config_for(tmp_path)
    source["decoding"]["mcts"].update(max_model_calls=1, stop_mode="budget")
    instance, _ = wrapper(iter(["bad", "#### 1"]))
    directory, metrics = benchmark.run_evaluation("mcts_confidence", source, model_factory=lambda _: instance)
    assert metrics["accuracy"] == .5 and metrics["parse_failures"] == 1
    assert metrics["fallback_count"] == 2 and metrics["mean_selected_depth"] == 0
    assert metrics["total_search_model_calls"] == 2
    assert metrics["total_completion_model_calls"] == 8
    assert metrics["total_model_calls"] == len(instance.model.seen) == 10
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    assert all(row["stop_reason"] == "model_call_budget" for row in rows)
    assert all(row["selected_initialization"]["committed"] == [] for row in rows)


@pytest.mark.parametrize("settings", [{"init_tokens": 5}, {"max_model_calls": 0},
                                      {"top_k_actions": False}, {"unknown": 1}, "invalid"])
def test_mcts_configuration_rejected_before_loading(tmp_path, settings):
    source = config_for(tmp_path)
    if isinstance(settings, dict):
        source["decoding"]["mcts"].update(settings)
    else:
        source["decoding"]["mcts"] = settings
    with pytest.raises(ValueError):
        benchmark.run_evaluation("mcts_confidence", source,
                                 model_factory=lambda _: pytest.fail("Loaded invalid search"))


def test_mcts_entrypoint_defaults_and_receipt(tmp_path, monkeypatch):
    captured = []
    receipt = tmp_path / "receipt.json"

    def run(method, config, **kwargs):
        captured.append((method, config, kwargs))
        return tmp_path / "exact-run", {"status": "completed"}

    monkeypatch.setattr(benchmark, "run_evaluation", run)
    monkeypatch.setattr(sys, "argv", ["script", "--run-manifest", str(receipt)])
    runpy.run_path(str(ROOT / "scripts/eval_gsm8k_mcts_confidence.py"), run_name="__main__")
    method, config, _ = captured[0]
    assert method == config["decoding"]["method"] == "mcts_confidence"
    assert config["model"]["family"] == "llada" and config["data"]["num_samples"] == 20
    assert config["generation"] == {"gen_length": 256, "steps": 256,
                                    "block_length": 128, "temperature": 1.0}
    assert config["decoding"]["mcts"] == {
        "init_tokens": 10, "num_candidates": 1, "top_k_tokens": 3, "top_k_actions": 2,
        "max_model_calls": 256, "margin_gamma": 10.0, "exploration_c": .01,
        "max_simulations": None,
        "epsilon": 1e-8, "stop_mode": "candidates_or_budget",
        "stability_mode": "none", "stability_lambda": 3.0,
        "stability_reward_beta": 1.0, "adaptive_min_depth": 1,
        "adaptive_min_model_calls": 1, "adaptive_patience": 1,
        "adaptive_low_uncertainty": .1, "adaptive_high_uncertainty": .35,
    }
    assert json.loads(receipt.read_text()) == {"run_dir": str(tmp_path / "exact-run"),
        "status": "completed", "model_family": "llada", "method": "mcts_confidence", "profile": "smoke"}


def test_mcts_entrypoint_rejects_illada_before_loading(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["script", "--config", str(ROOT / "configs/gsm8k/illada_decoding.yaml")])
    monkeypatch.setattr(benchmark, "run_evaluation", lambda *a, **k: pytest.fail("Loaded wrong family"))
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(ROOT / "scripts/eval_gsm8k_mcts_confidence.py"), run_name="__main__")
    assert error.value.code == 2 and "requires model.family: llada" in capsys.readouterr().err


def test_mcts_entrypoint_failure_does_not_publish_receipt(tmp_path, monkeypatch):
    receipt = tmp_path / "receipt.json"

    def fail(*args, **kwargs):
        raise RuntimeError("controlled MCTS failure")

    monkeypatch.setattr(benchmark, "run_evaluation", fail)
    monkeypatch.setattr(sys, "argv", ["script", "--run-manifest", str(receipt)])
    with pytest.raises(RuntimeError, match="controlled MCTS failure"):
        runpy.run_path(str(ROOT / "scripts/eval_gsm8k_mcts_confidence.py"), run_name="__main__")
    assert not receipt.exists()


def test_interrupted_child_evaluation_keeps_prior_record_and_active_cost_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, 'get_manifest', lambda *a, **k: manifest())
    instance, _ = wrapper()
    class Failing(FakeModel):
        def __call__(self, ids, **kw):
            if len(self.seen) == 7:
                raise RuntimeError('failure during second question child evaluation')
            return super().__call__(ids, **kw)
    instance.model = Failing()
    with pytest.raises(RuntimeError, match='second question'):
        benchmark.run_evaluation('mcts_confidence', config_for(tmp_path), model_factory=lambda _: instance)
    directory=next(tmp_path.rglob('metrics.json')).parent
    metrics=json.loads((directory/'metrics.json').read_text())
    assert metrics['status']=='failed' and metrics['processed']==1
    assert metrics['total_model_calls']==5
    assert metrics['active_prediction_model_calls']==3
    trace=[json.loads(line) for line in (directory/'traces/test_0001.jsonl').read_text().splitlines()]
    assert trace[-1]['stage']=='forward' and trace[-1]['search_model_calls']==3
    assert sum(e['stage']=='backpropagation' for e in trace)==1
