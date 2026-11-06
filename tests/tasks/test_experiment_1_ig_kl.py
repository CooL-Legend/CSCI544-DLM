"""E1 really dispatches reward B and cannot reuse original-IG artifacts."""

import json

import pytest

from dlm_project.tasks.gsm8k import experiment_1 as e1
from dlm_project.tasks.gsm8k.e3_analysis import compute_e3_question_record, load_question_checkpoints
from tests.tasks.test_experiment_1 import mock_wrapper


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
