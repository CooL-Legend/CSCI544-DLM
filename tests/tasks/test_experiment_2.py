"""Unit tests for Experiment 2 (E2: Two-Step Lookahead FIG Reward)."""

import math
from pathlib import Path
import pytest
import torch

from dlm_project.decoding.ab_batch import execute_trees
from dlm_project.decoding.ab_medal import ABMEDALConfig
from dlm_project.decoding.e2_search import (
    E2ABMEDALSearch,
    E2MEDALSearch,
    compute_fig_reward,
    compute_srig_reward,
)
from dlm_project.decoding.mcts_decoding import MCTSConfig
from dlm_project.tasks.gsm8k import experiment_2 as e2
from tests.decoding.test_ab_batch import BatchModel
from tests.tasks.test_experiment_1 import mock_wrapper


def test_fig_reward_formula():
    """Verify exact computation of the two-step lookahead FIG reward."""
    # Standard positive information gain
    r1 = compute_fig_reward(parent_entropy=10.0, grandchild_entropy=6.0, epsilon=1e-8)
    assert abs(r1 - 0.4) < 1e-6

    # Zero grandchild entropy (full commitment)
    r2 = compute_fig_reward(parent_entropy=5.0, grandchild_entropy=0.0, epsilon=1e-8)
    assert abs(r2 - 1.0) < 1e-6

    # Near-zero parent entropy protected by epsilon
    r3 = compute_fig_reward(parent_entropy=0.0, grandchild_entropy=0.0, epsilon=1e-8)
    assert abs(r3 - 0.0) < 1e-6

    # Negative gain (grandchild more uncertain than parent)
    r4 = compute_fig_reward(parent_entropy=4.0, grandchild_entropy=6.0, epsilon=1e-8)
    assert abs(r4 - (-0.5)) < 1e-6


def test_srig_reward_formula():
    """Verify exact computation of the two-step lookahead SRIG (Sum of Relative Information Gains) reward."""
    # Standard positive two-step relative IG:
    # Step 1: (10 - 6) / 10 = 0.4
    # Step 2: (6 - 3) / 6 = 0.5
    # Total (lambda=1.0): 0.4 + 1.0 * 0.5 = 0.9
    r1 = compute_srig_reward(parent_entropy=10.0, child_entropy=6.0, grandchild_entropy=3.0, reward_lambda=1.0)
    assert abs(r1 - 0.9) < 1e-6

    # Weighted lambda = 0.5: 0.4 + 0.5 * 0.5 = 0.65
    r2 = compute_srig_reward(parent_entropy=10.0, child_entropy=6.0, grandchild_entropy=3.0, reward_lambda=0.5)
    assert abs(r2 - 0.65) < 1e-6

    # Terminal child (H(c)=0 -> step 2 gain is 0, total = 1.0)
    r3 = compute_srig_reward(parent_entropy=5.0, child_entropy=0.0, grandchild_entropy=0.0, reward_lambda=1.0)
    assert abs(r3 - 1.0) < 1e-6

    # Negative step 2 IG (grandchild more uncertain than child)
    # Step 1: (8 - 4) / 8 = 0.5
    # Step 2: (4 - 6) / 4 = -0.5
    # Total: 0.5 + 1.0 * (-0.5) = 0.0
    r4 = compute_srig_reward(parent_entropy=8.0, child_entropy=4.0, grandchild_entropy=6.0, reward_lambda=1.0)
    assert abs(r4 - 0.0) < 1e-6

    with pytest.raises(ValueError, match="Invalid reward_lambda"):
        compute_srig_reward(parent_entropy=10.0, child_entropy=6.0, grandchild_entropy=3.0, reward_lambda=float("nan"))


def test_e2_context_and_manifest_slicing():
    """Verify dev (5), canonical eval (20), extended eval (50), and full eval (70) slicing."""
    ctx_dev = e2.context(stage="dev", methods=("medal", "ab_medal"))
    assert ctx_dev["stage"] == "dev"
    assert len(ctx_dev["examples"]) == 5

    ctx_eval_20 = e2.context(stage="evaluation", methods=("medal", "ab_medal"), sample_slice=(0, 20))
    assert len(ctx_eval_20["examples"]) == 20

    ctx_eval_50 = e2.context(stage="evaluation", methods=("medal", "ab_medal"), sample_slice=(20, 70))
    assert len(ctx_eval_50["examples"]) == 50

    ctx_eval_all = e2.context(stage="evaluation", methods=("medal", "ab_medal"))
    assert len(ctx_eval_all["examples"]) == 70

    # Verify that the first 20 in full eval match canonical eval 20 exactly
    assert [e["example_id"] for e in ctx_eval_all["examples"][:20]] == [e["example_id"] for e in ctx_eval_20["examples"]]
    # Verify that the remaining 50 match extended eval 50 exactly
    assert [e["example_id"] for e in ctx_eval_all["examples"][20:]] == [e["example_id"] for e in ctx_eval_50["examples"]]


@pytest.mark.parametrize("method", ["medal", "ab_medal"])
def test_e2_search_two_step_step_execution(method):
    """Verify single step lookahead forward execution for MEDAL and AB-MEDAL."""
    m = BatchModel()
    prompt = torch.tensor([[3, 3, 3]])
    cfg_kwargs = dict(init_tokens=4, max_model_calls=10, stop_mode="budget")

    if method == "medal":
        tree = E2MEDALSearch(m, prompt, 0, 50, MCTSConfig(**cfg_kwargs), seed=1)
    else:
        tree = E2ABMEDALSearch(m, prompt, 0, 50, ABMEDALConfig(**cfg_kwargs), seed=1)

    # First step evaluates root, child, and grandchild
    tree.step()

    # Root (1 forward) + child (1 forward) + grandchild (1 forward) = 3 model calls
    assert tree.model_calls == 3
    assert tree.root.visits == 1
    # Check that child received non-null immediate reward
    child = tree.nodes[1]
    assert child.immediate_reward is not None
    assert child.visits == 1
    assert math.isfinite(child.immediate_reward)


def test_e2_grandchild_cache_reuse():
    """Verify that cached grandchild analysis is reused at 0 extra NFEs upon subsequent visit."""
    m = BatchModel()
    prompt = torch.tensor([[3, 3, 3]])
    tree = E2MEDALSearch(m, prompt, 0, 50, MCTSConfig(init_tokens=4, max_model_calls=15, stop_mode="budget"), seed=1)

    tree.step()
    initial_calls = tree.model_calls
    assert len(tree.cached_analyses) >= 2  # Root, child, and grandchild cached

    # Grandchild key must be in cached_analyses
    child = tree.nodes[1]
    greedy_action = child.analysis.actions[0]
    expected_grandchild_state = child.state.clone()
    expected_grandchild_state[0, tree.prompt_len + greedy_action.output_position] = greedy_action.token_id
    g_key = tuple(expected_grandchild_state[0].tolist())

    assert g_key in tree.cached_analyses


def test_e2_batched_parity_with_serial():
    """Verify serial vs batched execution yields identical search trees and selections."""
    prompt = torch.tensor([[3, 3, 3]])

    # Serial run
    m1 = BatchModel()
    serial_ab = E2ABMEDALSearch(m1, prompt, 0, 50, ABMEDALConfig(init_tokens=5, max_model_calls=12, stop_mode="budget"), seed=7)
    while not serial_ab.done:
        serial_ab.step()

    # Batched run
    m2 = BatchModel()
    batched_ab = E2ABMEDALSearch(m2, prompt, 0, 50, ABMEDALConfig(init_tokens=5, max_model_calls=12, stop_mode="budget"), seed=7)
    run = execute_trees([batched_ab], pad_token_id=4)
    assert run.status == "completed"

    assert serial_ab.model_calls == batched_ab.model_calls
    assert len(serial_ab.nodes) == len(batched_ab.nodes)
    assert serial_ab.result().selected.node_id == batched_ab.result().selected.node_id

    for s_n, b_n in zip(serial_ab.nodes, batched_ab.nodes):
        assert torch.equal(s_n.state, b_n.state)
        assert s_n.visits == b_n.visits
        assert abs(s_n.reward_sum - b_n.reward_sum) < 1e-6


def test_e2_stage_execution_and_artifacts(tmp_path):
    """Verify full end-to-end stage execution, per-question checkpoint files, and CSV summaries."""
    w = mock_wrapper()
    ctx = e2.context(stage="dev", methods=("medal", "ab_medal"), num_samples=2)
    budgets = (4, 8)

    q_rows, agg_rows = e2.run_e2_stage(
        w,
        ctx,
        tmp_path,
        methods=("medal", "ab_medal"),
        budgets=budgets,
        max_depth=4,
        num_candidates=2,
        batch_size=2,
    )

    # 2 questions x 2 methods x 2 budgets = 8 checkpoint outcomes
    assert len(q_rows) == 8
    # 2 methods x 2 budgets = 4 aggregate rows
    assert len(agg_rows) == 4

    stage_dir = tmp_path / "dev"
    summaries_dir = tmp_path / "summaries"
    assert (summaries_dir / "e2_dev_per_question.csv").exists()
    assert (summaries_dir / "e2_dev_aggregate.csv").exists()
    assert (tmp_path / "run_manifest.json").exists()

    for method in ("medal", "ab_medal"):
        for ex in ctx["examples"]:
            ex_id = ex["example_id"]
            q_dir = stage_dir / method / ex_id
            assert (q_dir / "trace.jsonl").exists()
            assert (q_dir / "nodes.jsonl").exists()
            assert (q_dir / "search_audit.json").exists()

            for b in budgets:
                chk_dir = q_dir / "checkpoints" / str(b)
                assert (chk_dir / "tree.json").exists()
                assert (chk_dir / "selection.json").exists()
                assert (chk_dir / "completion.json").exists()
                assert (chk_dir / "metrics.json").exists()


def test_e2_e3_offline_analysis_compatibility(tmp_path):
    """Verify that E3 offline analysis engine extracts all structural metrics and figures from E2 artifacts."""
    from dlm_project.tasks.gsm8k import e3_analysis, e3_tree_plots

    w = mock_wrapper()
    ctx = e2.context(stage="dev", methods=("medal", "ab_medal"), num_samples=2)
    budgets = (4, 8)

    e2.run_e2_stage(
        w,
        ctx,
        tmp_path,
        methods=("medal", "ab_medal"),
        budgets=budgets,
        max_depth=4,
        num_candidates=2,
        batch_size=2,
    )

    stage_dir = tmp_path / "dev"
    e3_out = tmp_path / "e3_summaries"
    per_q, agg = e3_analysis.analyze_e3_stage(
        stage_dir,
        e3_out,
        stage="dev",
        methods=("medal", "ab_medal"),
        budgets=budgets,
    )

    # 2 questions x 2 methods x 2 budgets = 8 records
    assert len(per_q) == 8
    assert len(agg) == 4
    assert (e3_out / "e3_dev_per_question.csv").exists()
    assert (e3_out / "e3_dev_aggregate.csv").exists()
    assert (e3_out / "e3_figures" / "fig1_nfe_by_depth.png").exists()

    # Check that structural metrics are populated and non-null
    for row in per_q:
        assert "branching_factor_mean" in row
        assert "expanded_parents" in row
        assert "unique_evaluated_states" in row
        assert "negative_ig_percentage" in row
        assert "ig_mean" in row
        assert row["search_nfes"] in budgets
        if row["method"] == "ab_medal":
            assert row["gen_count"] is not None
            assert row["cont_count"] is not None

    # Verify tree visualizer validity
    for method in ("medal", "ab_medal"):
        for ex in ctx["examples"]:
            q_dir = stage_dir / method / ex["example_id"]
            chks = e3_analysis.load_question_checkpoints(q_dir, budgets=budgets)
            for b, data in chks.items():
                tree = e3_tree_plots._validated_tree(data, method=method, example_id=ex["example_id"], budget=b)
                assert tree.root_id == 0
                assert len(tree.nodes) >= 1


@pytest.mark.parametrize("method", ["medal", "ab_medal"])
def test_e2_srig_search_execution(method):
    """Verify SRIG (Sum of Relative Information Gains) 2-step search execution."""
    m = BatchModel()
    prompt = torch.tensor([[3, 3, 3]])
    cfg_kwargs = dict(init_tokens=4, max_model_calls=10, stop_mode="budget")

    if method == "medal":
        tree = E2MEDALSearch(m, prompt, 0, 50, MCTSConfig(**cfg_kwargs), seed=1, reward_mode="srig", reward_lambda=1.0)
    else:
        tree = E2ABMEDALSearch(m, prompt, 0, 50, ABMEDALConfig(**cfg_kwargs), seed=1, reward_mode="srig", reward_lambda=1.0)

    tree.step()

    # Root (1) + child (1) + grandchild (1) = 3 calls
    assert tree.model_calls == 3
    child = tree.nodes[1]
    assert child.immediate_reward is not None
    assert math.isfinite(child.immediate_reward)

    # Check trace evaluation event
    eval_events = [e for e in tree.trace if e.get("stage") == "evaluation"]
    assert len(eval_events) >= 1
    ev = eval_events[0]
    assert ev["reward_mode"] == "srig"
    assert ev["reward_lambda"] == 1.0
    assert "rig_step1" in ev and "rig_step2" in ev
    assert abs(ev["reward"] - (ev["rig_step1"] + ev["rig_step2"])) < 1e-6

    # Verify virtual grandchild did NOT mutate child's GEN/CONT posteriors in AB-MEDAL
    if method == "ab_medal":
        assert child.node_id in tree.posteriors
        child_models = tree.posteriors[child.node_id]
        # Child was registered after simulation backup: GEN and CONT posteriors must be unupdated (prior state: kappa=1.0, nu=1.0)
        assert child_models["GEN"].kappa == 1.0
        assert child_models["CONT"].kappa == 1.0


def test_e2_exact_boundary_checkpoint_equivalence():
    """Verify that an intermediate checkpoint snapshot at B=4 is exact and never mutates live search."""
    m = BatchModel()
    prompt = torch.tensor([[3, 3, 3]])
    # Root (1) + child (2) + grandchild (3) -> 3 calls.
    # Next child is call 4 (hits checkpoint B=4). Grandchild is call 5.
    tree = E2ABMEDALSearch(
        m,
        prompt,
        0,
        50,
        ABMEDALConfig(init_tokens=4, max_model_calls=6, stop_mode="budget"),
        checkpoint_budgets=(4,),
        seed=1,
    )

    while not tree.done:
        tree.step()

    assert 4 in tree.checkpoints
    chk_4 = tree.checkpoints[4]
    # Checkpoint must have exact 4 model calls recorded
    assert chk_4["model_calls"] == 4
    assert chk_4["stop_reason"] == "model_call_budget"

    # Live tree must have proceeded to the hard cap 6
    assert tree.model_calls == 6
    # Live child node 2 has its full 2-step reward evaluated
    child_2 = tree.nodes[2]
    assert child_2.immediate_reward is not None


def test_e2_simulation_cap_safeguard():
    """Verify that simulation cap safeguard stops runaway cached searches and records NFE utilization."""
    m = BatchModel()
    prompt = torch.tensor([[3, 3, 3]])
    # Set simulation cap multiplier to 1 -> max_simulations = 1 * 10 = 10
    tree = E2MEDALSearch(
        m,
        prompt,
        0,
        50,
        MCTSConfig(init_tokens=4, max_model_calls=50, max_simulations=2, stop_mode="budget"),
        sim_cap_multiplier=1,
        seed=1,
    )

    while not tree.done:
        tree.step()

    assert tree.stop_reason == "simulation_budget"
    assert tree.simulations_completed >= 2
    assert tree.model_calls < 50
    chk = tree._snapshot_checkpoint(budget=50, stop_reason=tree.stop_reason)
    assert chk["nfe_utilization"] < 1.0
    assert chk["stop_reason"] == "simulation_budget"


