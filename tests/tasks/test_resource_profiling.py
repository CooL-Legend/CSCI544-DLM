"""Measurements must preserve algorithms and count actual, non-resumed work."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dlm_project.experiments import resources
from dlm_project.experiments.resume import resume_signature, validate_resume
from dlm_project.tasks.gsm8k import experiment_1 as e1, experiment_2 as e2
from tests.tasks.test_experiment_1 import mock_wrapper


def small_context(module, count=4):
    kwargs = {"reward_mode": "srig"} if module is e2 else {}
    ctx = module.context(stage="dev", num_samples=count, **kwargs)
    ctx["common"]["generation"].update(gen_length=12, steps=12, block_length=12)
    return ctx


def unequal_wrapper():
    wrapper = mock_wrapper()
    # Deliberately ensure all four prompts have different lengths in both runs.
    lengths = iter((2, 3, 5, 8, 9))
    wrapper.encode_prompt = lambda text: torch.tensor([[3] * next(lengths)])
    return wrapper


@pytest.mark.parametrize("module", [e1, e2])
def test_four_question_parity_profiling_and_actual_counts(tmp_path, module):
    stage = module.run_e1_stage if module is e1 else module.run_e2_stage
    outputs = []
    for label, batch, enabled in (("serial", 1, False), ("batch", 4, False), ("profile", 4, True)):
        wrapper = unequal_wrapper()
        original = wrapper.model
        ctx = small_context(module)
        rows, _ = stage(wrapper, ctx, tmp_path / label, budgets=(4, 8), max_depth=4,
                        num_candidates=2, batch_size=batch, profile_resources=enabled)
        outputs.append(rows)
        assert wrapper.model is original
        if enabled:
            summary = json.loads((tmp_path / label / "resource_summary.json").read_text())
            assert summary["prompt_token_lengths"] == [2, 3, 5, 8]
            assert summary["fully_fresh_questions"] == 4
            assert summary["fresh_questions_per_hour"] > 0
            phases = summary["phases"]
            assert sum(p["model_invocations"] for p in phases.values()) == len(original.calls)
            assert sum(p["sequence_evaluations"] for p in phases.values()) == sum(x.shape[0] for x, _ in original.calls)
            assert phases["search"]["max_active_sequences"] == 4
    for left, right, instrumented in zip(*outputs):
        for key in ("selected_state_hash", "selected_node_id", "selected_q", "predicted_answer",
                    "exact_match", "search_nfes", "completion_nfes", "total_nfes"):
            assert left[key] == right[key] == instrumented[key]
        for name, key in (("tree", "nodes"), ("completion", "token_ids")):
            data = [json.loads((Path(r["artifact_directory"]) / f"{name}.json").read_text())[key]
                    for r in (left, right, instrumented)]
            assert data[0] == data[1] == data[2]


@pytest.mark.parametrize("module", [e1, e2])
def test_eight_completion_sequences_dedup_and_cache(tmp_path, module):
    wrapper = unequal_wrapper()
    ctx = small_context(module)
    items = module.prepare(wrapper, ctx)
    requests = []
    for budget, token in ((4, 1), (8, 2)):
        for item in items:
            state = torch.cat((item["ids"], torch.tensor([[token, 0, 0]])), dim=1)
            requests.append((item, state, budget))
    # Ninth request is another budget selecting exactly the same first state.
    requests.append((items[0], requests[0][1].clone(), 12))
    cache = {}
    profiler = resources.ResourceProfiler(wrapper, ctx, tmp_path, methods=("medal", "ab_medal"),
                                          budgets=(4, 8), batch_size=4, resume=False)
    token = resources._CURRENT.set(profiler)
    try:
        first = module.complete_selected_states_batch(wrapper, requests, ctx, completion_cache=cache)
        before = len(wrapper.model.calls)
        second = module.complete_selected_states_batch(wrapper, requests, ctx, completion_cache=cache)
        assert len(wrapper.model.calls) == before
        assert len(first) == len(second) == 9
        assert sum(v[-1] for v in first.values()) == 1
        assert all(v[-1] for v in second.values())
    finally:
        resources._CURRENT.reset(token)
        profiler.finish("completed")
    summary = json.loads((tmp_path / "resource_summary.json").read_text())
    phase = summary["phases"]["completion"]
    assert phase["max_active_sequences"] == 8
    assert phase["unique_completion_requests"] == 8
    assert phase["cached_completion_requests"] == 10
    assert phase["model_invocations"] == before


@pytest.mark.parametrize("module", [e1, e2])
def test_resume_partial_batch_corruption_and_mismatch(tmp_path, module):
    stage = module.run_e1_stage if module is e1 else module.run_e2_stage
    ctx = small_context(module, 5)
    wrapper = mock_wrapper()
    options = dict(budgets=(4, 8), max_depth=4, num_candidates=2, batch_size=4, profile_resources=True)
    rows, _ = stage(wrapper, ctx, tmp_path, **options)
    initial = json.loads((tmp_path / "resource_summary.json").read_text())
    assert initial["fully_fresh_questions"] == 5
    events = [json.loads(line) for line in (tmp_path / "resource_batches.jsonl").read_text().splitlines()]
    assert any(r["phase"] == "search" and r["max_active_sequences"] == 1 for r in events)
    before = len(wrapper.model.calls)
    stage(wrapper, ctx, tmp_path, **options)
    resumed = json.loads((tmp_path / "resource_summary.json").read_text())
    assert resumed["fully_resumed_questions"] == 5
    assert resumed["fully_fresh_questions"] == 0
    assert resumed["fresh_questions_per_hour"] is None
    assert len(wrapper.model.calls) == before
    assert resumed["attempt_id"] != initial["attempt_id"]
    assert len(list((tmp_path / "resource_attempts").glob("*.json"))) == 2

    # A metrics file alone is insufficient; missing tree triggers exactly one method rerun.
    directory = Path(rows[0]["artifact_directory"])
    (directory / "tree.json").unlink()
    stage(wrapper, ctx, tmp_path, **options)
    mixed = json.loads((tmp_path / "resource_summary.json").read_text())
    assert mixed["mixed_questions"] == 1
    assert mixed["fresh_questions_per_hour"] is None
    assert len(wrapper.model.calls) > before
    assert (directory / "tree.json").exists()

    manifest = (tmp_path / "run_manifest.json").read_text()
    with pytest.raises(ValueError, match="configuration differs"):
        stage(wrapper, ctx, tmp_path, **{**options, "max_depth": 3})
    assert (tmp_path / "run_manifest.json").read_text() == manifest


def test_resume_allows_200_to_500_frozen_extension(tmp_path):
    old_ctx = e1.context(stage="test", num_samples=200)
    new_ctx = e1.context(stage="test", num_samples=500)
    wrapper = mock_wrapper()
    old_items, new_items = e1.prepare(wrapper, old_ctx), e1.prepare(wrapper, new_ctx)
    old = resume_signature(old_ctx, old_items, ("medal", "ab_medal"), (256, 512), 10, 3, 4)
    new = resume_signature(new_ctx, new_items, ("medal", "ab_medal"), (256, 512), 10, 3, 4)
    (tmp_path / "summaries").mkdir()
    (tmp_path / "summaries/run_manifest_test.json").write_text(json.dumps({"resume_signature": old}))
    validate_resume(tmp_path, "test", new, True)
    new["examples"][0]["identity_hash"] = "changed"
    with pytest.raises(ValueError, match="prefix"):
        validate_resume(tmp_path, "test", new, True)


def test_peak_counters_survive_cleanup_and_gpu_units(tmp_path, monkeypatch):
    counters = {"allocated": 5 * resources.GIB, "reserved": 7 * resources.GIB,
                "peak_allocated": 0, "peak_reserved": 0}
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: counters["allocated"])
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: counters["reserved"])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: counters["peak_allocated"])
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a: counters["peak_reserved"])
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: counters.update(peak_allocated=0, peak_reserved=0))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: SimpleNamespace(name="fake L40", total_memory=48 * resources.GIB))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (40 * resources.GIB, 48 * resources.GIB))
    wrapper = SimpleNamespace(model=SimpleNamespace(device="cuda"), identity={})
    profiler = resources.ResourceProfiler(wrapper, small_context(e1), tmp_path, methods=("medal",),
                                          budgets=(4, 8), batch_size=4, resume=False)

    @resources.resource_phase("search")
    def synthetic_phase(wrapper):
        counters.update(peak_allocated=16 * resources.GIB, peak_reserved=20 * resources.GIB)
        # Simulate deletion/empty_cache before the enclosing scope collects its peaks.
        counters.update(allocated=5 * resources.GIB, reserved=5 * resources.GIB)

    token = resources._CURRENT.set(profiler)
    try:
        synthetic_phase(wrapper)
    finally:
        resources._CURRENT.reset(token)
        profiler.finish("completed")
    summary = json.loads((tmp_path / "resource_summary.json").read_text())
    assert summary["model_loaded_baseline"]["reserved_gib"] == 7
    assert summary["phases"]["search"]["peak_allocated_gib"] == 16
    assert summary["phases"]["search"]["peak_reserved_gib"] == 20
    assert summary["gpu_headroom_fraction"] == pytest.approx(28 / 48)


def test_checkpoint_times_are_measured_when_captured(monkeypatch):
    tick = iter((3.0, 11.0))
    monkeypatch.setattr(resources, "perf_counter", lambda: next(tick))
    tree = SimpleNamespace(_snapshot_checkpoint=lambda budget: {"budget": budget})
    resources.watch_checkpoints(tree, SimpleNamespace(device="cpu"), 1.0)
    assert tree._snapshot_checkpoint(256)["search_elapsed_seconds"] == 2
    assert tree._snapshot_checkpoint(512)["search_elapsed_seconds"] == 10


def test_failure_restores_model_and_records_attempt(tmp_path):
    wrapper = mock_wrapper()
    original = wrapper.model
    original.failure = RuntimeError("deliberate failure")
    original.fail_at = 2
    with pytest.raises(RuntimeError, match="deliberate"):
        e1.run_e1_stage(wrapper, small_context(e1), tmp_path, budgets=(4, 8), batch_size=4,
                        profile_resources=True)
    assert wrapper.model is original
    assert resources._CURRENT.get() is None
    summary = json.loads((tmp_path / "resource_summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["fully_fresh_questions"] == 0
    assert summary["fresh_questions_per_hour"] is None
    assert summary["phases"]["search"]["model_invocations"] == 2
