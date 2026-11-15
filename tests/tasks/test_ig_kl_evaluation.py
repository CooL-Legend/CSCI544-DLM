"""The 70-question adapter changes the frozen pool, not the IG-KL experiment."""

import pytest
import torch

from dlm_project.tasks.gsm8k import experiment_1 as e1
from dlm_project.tasks.gsm8k import ig_kl_evaluation as evaluation
from tests.tasks.test_experiment_1 import mock_wrapper


def test_canonical_twenty_are_exactly_the_original_ig_kl_examples():
    original = e1.context(stage="evaluation", reward_mode="ig_kl")
    canonical = evaluation.context(sample_slice=(0, 20))
    assert len(original["examples"]) == len(canonical["examples"]) == 20
    assert canonical["examples"] == original["examples"]
    for key in ("common", "reward", "methods", "budgets", "max_depth", "num_candidates",
                "prompt_template_hash", "stage"):
        assert canonical[key] == original[key], key


def test_seventy_questions_are_two_disjoint_ordered_parts():
    complete = evaluation.context()
    first = evaluation.context(sample_slice=(0, 20))
    extra = evaluation.context(sample_slice=(20, 70))
    assert len(complete["examples"]) == 70
    assert len(extra["examples"]) == 50
    assert first["examples"] + extra["examples"] == complete["examples"]
    ids = [example["example_id"] for example in complete["examples"]]
    assert len(set(ids)) == 70
    assert not {ex["example_id"] for ex in first["examples"]} & {
        ex["example_id"] for ex in extra["examples"]
    }
    assert complete["manifest_hash"] == first["manifest_hash"] == extra["manifest_hash"]
    assert complete["reward"]["reward_mode"] == "ig_kl"
    assert complete["reward"]["kl_weight"] == 0.05
    assert complete["reward"]["kl_top_k"] == 32


def test_num_samples_selects_a_frozen_prefix_and_eval_alias_is_canonical():
    all_examples = evaluation.context()["examples"]
    selected = evaluation.context(stage="eval", num_samples=3)
    assert selected["stage"] == "evaluation"
    assert selected["examples"] == all_examples[:3]


def test_reward_configuration_is_passed_to_the_existing_runner():
    configured = evaluation.context(kl_weight=0.2, kl_top_k=0)
    original = e1.context(stage="evaluation", reward_mode="ig_kl", kl_weight=0.2, kl_top_k=0)
    assert configured["reward"] == original["reward"]
    assert evaluation.load_model is e1.load_model
    assert evaluation.run_e1_stage is e1.run_e1_stage


@pytest.mark.parametrize("method", ["medal", "ab_medal"])
def test_canonical_search_is_unchanged_when_using_the_seventy_pool_adapter(method):
    """Compare actual synthetic-model searches, including backed-up KL rewards."""
    original = e1.context(stage="evaluation", num_samples=1, reward_mode="ig_kl")
    expanded = evaluation.context(sample_slice=(0, 1))
    searches = []
    for ctx in (original, expanded):
        wrapper = mock_wrapper()
        prepared = e1.prepare(wrapper, ctx)[0]
        search, _ = e1.execute_search_checkpointed(
            wrapper, prepared, ctx, method, budgets=(4, 8), max_depth=4, num_candidates=2,
        )
        searches.append(search)
    first, second = searches
    assert first.model_calls == second.model_calls == 8
    assert first.config.max_simulations is None
    assert second.config.max_simulations is None
    assert len(first.nodes) == len(second.nodes)
    for before, after in zip(first.nodes, second.nodes):
        assert torch.equal(before.state, after.state)
        assert before.visits == after.visits
        assert before.reward_sum == after.reward_sum
        assert before.immediate_reward == after.immediate_reward
    for budget in (4, 8):
        a, b = first.checkpoints[budget], second.checkpoints[budget]
        assert torch.equal(a["selected"].state, b["selected"].state)
        assert a["reward_metadata"] == b["reward_metadata"]


@pytest.mark.parametrize("sample_slice", [
    (0, 0), (20, 20), (30, 20), (-1, 20), (0, 71), (70, 71),
    (True, 20), (0, False), (0, 20.0), (0,), (0, 20, 70),
])
def test_invalid_slice_cannot_silently_change_the_experiment(sample_slice):
    with pytest.raises((ValueError, TypeError)):
        evaluation.context(sample_slice=sample_slice)


@pytest.mark.parametrize("count", [0, -1, 71, True, 1.5])
def test_invalid_sample_count_is_rejected(count):
    with pytest.raises((ValueError, TypeError)):
        evaluation.context(num_samples=count)


def test_slice_and_count_cannot_conflict():
    with pytest.raises(ValueError):
        evaluation.context(sample_slice=(0, 20), num_samples=3)


@pytest.mark.parametrize("stage", ["dev", "test", "unknown"])
def test_adapter_only_runs_the_evaluation_pool(stage):
    with pytest.raises(ValueError):
        evaluation.context(stage=stage)
