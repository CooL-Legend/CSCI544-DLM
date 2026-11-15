"""E1 really dispatches reward B and cannot reuse original-IG artifacts."""

import json
from pathlib import Path

import pytest
import torch

from dlm_project.tasks.gsm8k import experiment_1 as e1
from dlm_project.tasks.gsm8k.e3_analysis import compute_e3_question_record, load_question_checkpoints
from tests.tasks.test_experiment_1 import mock_wrapper
from tests.decoding.test_ab_batch import BatchModel


@pytest.mark.parametrize("method", ["medal", "ab_medal"])
def test_reward_b_is_used_and_exported_at_every_checkpoint(tmp_path, method):
    wrapper = mock_wrapper()
    ctx = e1.context(stage="dev", methods=(method,), num_samples=1,
                     reward_mode="ig_kl", kl_weight=0.5, kl_top_k=2)
    item = e1.prepare(wrapper, ctx)[0]
    rows = e1.run_question_e1(wrapper, item, ctx, method, tmp_path,
                              budgets=(3, 8), max_depth=3, num_candidates=2)
    question_dir = tmp_path / method / item["example"]["example_id"]
    events = [json.loads(line) for line in (question_dir / "trace.jsonl").read_text().splitlines()]
    evaluations = [e for e in events if e["stage"] == "evaluation"]
    assert evaluations and all(e["reward_mode"] == "ig_kl" for e in evaluations)
    for e in evaluations:
        assert e["reward"] == pytest.approx(e["immediate_gain"] - 0.5 * e["action_history_kl"])
    for row in rows:
        assert row["reward_mode"] == "ig_kl"
        assert row["kl_extra_model_calls"] == 0
        assert row["kl_top_k"] == 2
        assert row["kl_estimator"] == "top_k_tail"
        checkpoint = question_dir / "checkpoints" / str(row["search_budget"])
        tree = json.loads((checkpoint / "tree.json").read_text())
        selected = next(n for n in tree["nodes"] if n["node_id"] == row["selected_node_id"])
        assert row["selected_ig_reward"] == selected["immediate_gain"]
        assert row["selected_kl_penalty"] == selected["kl_penalty"]
        gains = [n["immediate_gain"] for n in tree["nodes"] if n["parent_id"] is not None]
        assert tree["statistics"]["ig_mean"] == pytest.approx(sum(gains) / len(gains))
        assert tree["statistics"]["negative_ig_count"] == sum(g < 0 for g in gains)
        # The offline E3 consumer accepts the new artifacts without inference.
        data = load_question_checkpoints(question_dir, (row["search_budget"],))[row["search_budget"]]
        before = len(wrapper.model.calls)
        record = compute_e3_question_record(method, row["search_budget"], data, row["example_id"])
        assert record["ig_mean"] == tree["statistics"]["ig_mean"]
        assert len(wrapper.model.calls) == before


def test_resume_same_reward_but_reject_other_rewards_before_writing(tmp_path):
    wrapper = mock_wrapper()
    ctx = e1.context(stage="dev", methods=("medal",), num_samples=1,
                     reward_mode="ig_kl", kl_weight=0.05, kl_top_k=2)
    kwargs = dict(methods=("medal",), budgets=(3, 6), max_depth=3, num_candidates=2)
    expected, _ = e1.run_e1_stage(wrapper, ctx, tmp_path, **kwargs)
    calls = len(wrapper.model.calls)
    actual, _ = e1.run_e1_stage(wrapper, ctx, tmp_path, **kwargs)
    assert actual == json.loads(json.dumps(expected)) and len(wrapper.model.calls) == calls
    manifest_path = tmp_path / "run_manifest.json"
    manifest = manifest_path.read_text()
    assert json.loads(manifest)["reward"] == ctx["reward"]
    for options in ({"reward_mode": "ig"}, {"reward_mode": "ig_kl", "kl_weight": 0.1},
                    {"reward_mode": "ig_kl", "kl_top_k": 0}):
        other = e1.context(stage="dev", methods=("medal",), num_samples=1, **options)
        with pytest.raises(ValueError, match="Reward settings differ"):
            e1.run_e1_stage(wrapper, other, tmp_path, resume=False, **kwargs)
        assert manifest_path.read_text() == manifest
        assert len(wrapper.model.calls) == calls


def test_legacy_ig_results_cannot_be_resumed_as_ig_kl(tmp_path):
    wrapper = mock_wrapper()
    ctx = e1.context(stage="dev", methods=("medal",), num_samples=1)
    kwargs = dict(methods=("medal",), budgets=(2,), max_depth=3, num_candidates=2)
    e1.run_e1_stage(wrapper, ctx, tmp_path, **kwargs)
    for path in (tmp_path / "run_manifest.json", tmp_path / "summaries" / "run_manifest_dev.json"):
        record = json.loads(path.read_text())
        record.pop("reward")  # Historical manifests have no reward metadata.
        path.write_text(json.dumps(record))
    changed = e1.context(stage="dev", methods=("medal",), num_samples=1, reward_mode="ig_kl")
    calls = len(wrapper.model.calls)
    with pytest.raises(ValueError, match="Reward settings differ"):
        e1.run_e1_stage(wrapper, changed, tmp_path, **kwargs)
    assert len(wrapper.model.calls) == calls


def test_interrupted_question_keeps_reward_identity_and_can_resume(tmp_path, monkeypatch):
    wrapper = mock_wrapper()
    ctx = e1.context(stage="dev", methods=("medal",), num_samples=1, reward_mode="ig_kl")
    original = e1.execute_search_checkpointed
    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted search")
    monkeypatch.setattr(e1, "execute_search_checkpointed", interrupted)
    kwargs = dict(methods=("medal",), budgets=(2,), max_depth=3, num_candidates=2)
    with pytest.raises(RuntimeError, match="interrupted search"):
        e1.run_e1_stage(wrapper, ctx, tmp_path, **kwargs)
    q_dir = tmp_path / "dev" / "medal" / ctx["examples"][0]["example_id"]
    assert json.loads((q_dir / "config.json").read_text())["reward"] == ctx["reward"]
    monkeypatch.setattr(e1, "execute_search_checkpointed", original)
    rows, _ = e1.run_e1_stage(wrapper, ctx, tmp_path, **kwargs)
    assert len(rows) == 1 and rows[0]["reward_mode"] == "ig_kl"


@pytest.mark.parametrize("method", ["medal", "ab_medal"])
def test_cooperative_reward_b_matches_serial_outputs_and_uses_fewer_forwards(tmp_path, method):
    serial, batch = mock_wrapper(), mock_wrapper()
    for wrapper in (serial, batch):
        wrapper.encode_prompt = lambda text: torch.tensor([[3] * (2 + len(text) % 7)])
    ctx_serial = e1.context(stage="dev", methods=(method,), num_samples=3,
                            reward_mode="ig_kl", kl_weight=0.5, kl_top_k=2)
    ctx_batch = e1.context(stage="dev", methods=(method,), num_samples=3,
                           reward_mode="ig_kl", kl_weight=0.5, kl_top_k=2)
    prepared = e1.prepare(serial, ctx_serial)
    assert len({item["ids"].shape[1] for item in prepared}) > 1
    kwargs = dict(methods=(method,), budgets=(3, 12), max_depth=4, num_candidates=2)
    serial_rows, _ = e1.run_e1_stage(serial, ctx_serial, tmp_path / "serial", batch_size=1, **kwargs)
    batch_rows, _ = e1.run_e1_stage(batch, ctx_batch, tmp_path / "batch", batch_size=2, **kwargs)
    assert len(serial_rows) == len(batch_rows) == 6
    assert len(batch.model.calls) < len(serial.model.calls)
    assert any(ids.shape[0] > 1 for ids, _ in batch.model.calls)
    assert any(mask is not None and (mask == 0).any() for _, mask in batch.model.calls)
    keys = ("example_id", "search_budget", "search_nfes", "selected_node_id", "selected_depth",
            "selected_q", "selected_state_hash", "predicted_answer", "exact_match",
            "reward_mode", "kl_weight", "kl_top_k", "selected_ig_reward", "selected_kl_penalty",
            "selected_action_history_kl", "kl_extra_model_calls")
    for left, right in zip(serial_rows, batch_rows):
        assert {k: left[k] for k in keys} == {k: right[k] for k in keys}
        assert right["search_nfes"] == right["search_budget"]
        assert right["kl_extra_model_calls"] == 0
        serial_path, batch_path = Path(left["artifact_directory"]), Path(right["artifact_directory"])
        tree_left = json.loads((serial_path / "tree.json").read_text())
        tree_right = json.loads((batch_path / "tree.json").read_text())
        assert tree_left["nodes"] == tree_right["nodes"]
        completion_left = json.loads((serial_path / "completion.json").read_text())
        completion_right = json.loads((batch_path / "completion.json").read_text())
        assert completion_left["token_ids"] == completion_right["token_ids"]
        if method == "ab_medal":
            assert json.loads((serial_path / "posteriors.json").read_text()) == json.loads((batch_path / "posteriors.json").read_text())
    calls = len(batch.model.calls)
    resumed, _ = e1.run_e1_stage(batch, ctx_batch, tmp_path / "batch", batch_size=2, **kwargs)
    assert resumed == json.loads(json.dumps(batch_rows))
    assert len(batch.model.calls) == calls


@pytest.mark.parametrize("method", ["medal", "ab_medal"])
def test_lambda_zero_cooperative_search_matches_original_ig(method):
    baseline, variant = mock_wrapper(), mock_wrapper()
    baseline_ctx = e1.context(stage="dev", methods=(method,), num_samples=2)
    variant_ctx = e1.context(stage="dev", methods=(method,), num_samples=2,
                             reward_mode="ig_kl", kl_weight=0, kl_top_k=2)
    originals = e1.execute_search_batch_checkpointed(
        baseline, e1.prepare(baseline, baseline_ctx), baseline_ctx, method,
        budgets=(3, 12), max_depth=4, num_candidates=2)
    variants = e1.execute_search_batch_checkpointed(
        variant, e1.prepare(variant, variant_ctx), variant_ctx, method,
        budgets=(3, 12), max_depth=4, num_candidates=2)
    assert len(baseline.model.calls) == len(variant.model.calls)
    for (left, _), (right, _) in zip(originals, variants):
        assert left.model_calls == right.model_calls == 12
        assert right.peak_history_cache_bytes == 0
        assert len(left.nodes) == len(right.nodes)
        for a, b in zip(left.nodes, right.nodes):
            assert torch.equal(a.state, b.state)
            assert (a.immediate_reward, a.visits, a.reward_sum) == (b.immediate_reward, b.visits, b.reward_sum)
        for budget in (3, 12):
            assert torch.equal(left.checkpoints[budget]["selected"].state,
                               right.checkpoints[budget]["selected"].state)
            if method == "ab_medal":
                assert left.checkpoints[budget]["posteriors"] == right.checkpoints[budget]["posteriors"]


@pytest.mark.parametrize("method", ["medal", "ab_medal"])
@pytest.mark.parametrize("failure_source", ["forward", "submit"])
def test_cooperative_failure_clears_every_question_history(monkeypatch, method, failure_source):
    model = BatchModel(failure=RuntimeError("shared forward failed") if failure_source == "forward" else None)
    wrapper = mock_wrapper(model)
    ctx = e1.context(stage="dev", methods=(method,), num_samples=2, reward_mode="ig_kl")
    class_name = "CooperativeIGKLMCTSSearch" if method == "medal" else "IGKLABMEDALSearch"
    original = getattr(e1, class_name)
    trees = []

    def capture(*args, **kwargs):
        tree = original(*args, **kwargs)
        if failure_source == "submit":
            original_submit = tree.submit_evaluation
            def submit(request, **options):
                if tree.model_calls == 3:
                    raise RuntimeError("submission failed")
                return original_submit(request, **options)
            tree.submit_evaluation = submit
        trees.append(tree)
        return tree

    monkeypatch.setattr(e1, class_name, capture)
    with pytest.raises(RuntimeError, match="failed"):
        e1.execute_search_batch_checkpointed(wrapper, e1.prepare(wrapper, ctx), ctx, method,
                                             budgets=(6,), max_depth=4, num_candidates=2)
    assert len(trees) == 2
    assert trees[0]._histories is not trees[1]._histories
    assert all(tree.peak_history_cache_bytes > 0 for tree in trees)
    assert all(not tree._histories and not tree._historical_kl for tree in trees)


def test_interrupted_batch_keeps_reward_identity_and_resumes(tmp_path, monkeypatch):
    wrapper = mock_wrapper()
    ctx = e1.context(stage="dev", methods=("medal",), num_samples=2, reward_mode="ig_kl")
    original = e1.execute_search_batch_checkpointed
    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted batch")
    monkeypatch.setattr(e1, "execute_search_batch_checkpointed", interrupted)
    kwargs = dict(methods=("medal",), budgets=(3, 6), max_depth=3, num_candidates=2, batch_size=2)
    with pytest.raises(RuntimeError, match="interrupted batch"):
        e1.run_e1_stage(wrapper, ctx, tmp_path, **kwargs)
    for example in ctx["examples"]:
        q_dir = tmp_path / "dev" / "medal" / example["example_id"]
        assert json.loads((q_dir / "config.json").read_text())["reward"] == ctx["reward"]
    monkeypatch.setattr(e1, "execute_search_batch_checkpointed", original)
    rows, _ = e1.run_e1_stage(wrapper, ctx, tmp_path, **kwargs)
    assert len(rows) == 4 and all(row["reward_mode"] == "ig_kl" for row in rows)


def test_ig_kl_resume_rejects_batch_size_changes_before_writing(tmp_path):
    wrapper = mock_wrapper()
    ctx = e1.context(stage="dev", methods=("medal",), num_samples=2, reward_mode="ig_kl")
    kwargs = dict(methods=("medal",), budgets=(3,), max_depth=3, num_candidates=2)
    e1.run_e1_stage(wrapper, ctx, tmp_path, batch_size=2, **kwargs)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = manifest_path.read_text()
    calls = len(wrapper.model.calls)
    with pytest.raises(ValueError, match="Batch size differs"):
        e1.run_e1_stage(wrapper, ctx, tmp_path, batch_size=1, **kwargs)
    assert manifest_path.read_text() == manifest
    assert len(wrapper.model.calls) == calls
