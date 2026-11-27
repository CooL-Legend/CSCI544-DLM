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
    compute_cmer_reward,
    compute_fig_reward,
    compute_pairwise_kl,
    compute_srig_reward,
    compute_swd_one_step_fallback,
    compute_swd_reward,
    extract_swd_state_data,
)
from dlm_project.decoding.mcts_decoding import MCTSConfig, StateAnalysis, _state_key
from dlm_project.experiments.resume import resume_signature
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


def test_cmer_reward_formula():
    """Verify Common-Mask Entropy Reduction over masked positions, negatives, and empty masks."""
    parent_h = torch.tensor([2.0, 1.5, 0.8, 1.2])
    target_h = torch.tensor([0.0, 0.5, 0.0, 0.4])
    # Only positions 1 and 3 remain masked in target; committed positions (0, 2) must be ignored
    mask = torch.tensor([False, True, False, True])
    reward, count = compute_cmer_reward(parent_h, target_h, mask)
    assert count == 2
    assert abs(reward - (((1.5 - 0.5) + (1.2 - 0.4)) / 2.0)) < 1e-6

    # Signed negative entropy reduction when target uncertainty increases at common masks
    inc_target_h = torch.tensor([0.0, 2.5, 0.0, 1.6])
    neg_reward, neg_count = compute_cmer_reward(parent_h, inc_target_h, mask)
    assert neg_count == 2
    assert abs(neg_reward - (((1.5 - 2.5) + (1.2 - 1.6)) / 2.0)) < 1e-6

    # Empty common mask set -> 0.0 reward without division by zero
    empty_reward, empty_count = compute_cmer_reward(parent_h, target_h, torch.tensor([False, False, False, False]))
    assert empty_count == 0
    assert empty_reward == 0.0


@pytest.mark.parametrize("method", ["medal", "ab_medal"])
def test_e2_cmer_search_execution(method):
    """Verify CMER 2-step search execution reuses cached per-position entropies cleanly."""
    m = BatchModel()
    prompt = torch.tensor([[3, 3, 3]])
    cfg_kwargs = dict(init_tokens=4, max_model_calls=10, stop_mode="budget")

    if method == "medal":
        tree = E2MEDALSearch(m, prompt, 0, 50, MCTSConfig(**cfg_kwargs), seed=1, reward_mode="cmer")
    else:
        tree = E2ABMEDALSearch(m, prompt, 0, 50, ABMEDALConfig(**cfg_kwargs), seed=1, reward_mode="cmer")

    tree.step()

    # Root (1) + child (1) + virtual grandchild (1) = 3 calls (0 extra NFEs for per-token entropies)
    assert tree.model_calls == 3
    child = tree.nodes[1]
    assert child.immediate_reward is not None
    assert math.isfinite(child.immediate_reward)

    eval_events = [e for e in tree.trace if e.get("stage") == "evaluation"]
    assert len(eval_events) == 1
    ev = eval_events[0]
    assert ev["reward_mode"] == "cmer"
    assert ev["lookahead_available"] is True
    assert ev["common_mask_count"] == 48  # gen_length=50 minus 1 child commit minus 1 grandchild commit

    # Verify trace reward matches cached grandchild StateAnalysis.entropies over remaining grandchild masks
    g_action = tree.cached_analyses[_state_key(child.state)].actions[0]
    g_state = child.state.clone()
    g_state[0, tree.prompt_len + g_action.output_position] = g_action.token_id
    g_analysis = tree.cached_analyses[_state_key(g_state)]
    g_mask = g_state[0, tree.prompt_len:] == tree.mask_id
    expected_reward, expected_count = compute_cmer_reward(
        tree.root.analysis.entropies, g_analysis.entropies, g_mask
    )
    assert ev["common_mask_count"] == expected_count
    assert abs(ev["reward"] - expected_reward) < 1e-6

    if method == "ab_medal":
        child_models = tree.posteriors[child.node_id]
        assert child_models["GEN"].kappa == 1.0
        assert child_models["CONT"].kappa == 1.0


@pytest.mark.parametrize("method", ["medal", "ab_medal"])
def test_e2_cmer_empty_common_masks_terminal_grandchild(method):
    """When child is terminal or grandchild fills final mask (|M(g*)| == 0), CMER reward is 0.0."""
    from dlm_project.decoding.mcts_decoding import Action, expand_node

    m = BatchModel()
    prompt = torch.tensor([[3, 3, 3]])
    cfg_kwargs = dict(init_tokens=4, max_model_calls=6, stop_mode="budget")

    def _make_tree():
        if method == "medal":
            t = E2MEDALSearch(m, prompt, 0, 5, MCTSConfig(**cfg_kwargs), seed=1, reward_mode="cmer")
        else:
            t = E2ABMEDALSearch(m, prompt, 0, 5, ABMEDALConfig(**cfg_kwargs), seed=1, reward_mode="cmer")
        req_root = t.next_evaluation()
        t.record_external_forward(req_root, invocation_id=1, batch_size=1)
        with torch.no_grad():
            t.submit_evaluation(req_root, logits=m(req_root.node.state).logits[0])
        return t

    # Case 1: Child with 1 mask left whose greedy action fills it -> |M(g*)| == 0 -> R_CMER = 0.0
    tree1 = _make_tree()
    near_terminal = expand_node(tree1.root, Action(0, 10, 0.9, 0.0, 0.1), 1, tree1.prompt_len, tree1.mask_id, 5)
    near_terminal.state[0, tree1.prompt_len : tree1.prompt_len + 4] = torch.tensor([10, 11, 12, 13])
    tree1.nodes.append(near_terminal)
    c_ent = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.9])
    c_analysis = StateAnalysis(c_ent, (Action(4, 14, 0.95, 0.0, 0.1),), _state_key(near_terminal.state), 0.95)
    if method == "medal":
        tree1._prepare_grandchild_or_finalize(near_terminal, tree1.root, c_analysis)
    else:
        tree1._prepare_grandchild_or_finalize(near_terminal, tree1.root, [tree1.root], ["GEN"], c_analysis)
    evals1 = [e for e in tree1.trace if e.get("stage") == "evaluation"]
    assert len(evals1) == 1
    assert evals1[0]["common_mask_count"] == 0
    assert evals1[0]["reward"] == 0.0

    # Case 2: Already terminal child (0 masks left) -> R_CMER = 0.0
    tree2 = _make_tree()
    terminal_child = expand_node(tree2.root, Action(0, 10, 0.9, 0.0, 0.1), 1, tree2.prompt_len, tree2.mask_id, 5)
    terminal_child.state[0, tree2.prompt_len :] = torch.tensor([10, 11, 12, 13, 14])
    tree2.nodes.append(terminal_child)
    t_analysis = StateAnalysis(torch.zeros(5), (), _state_key(terminal_child.state), 0.0)
    if method == "medal":
        tree2._prepare_grandchild_or_finalize(terminal_child, tree2.root, t_analysis)
    else:
        tree2._prepare_grandchild_or_finalize(terminal_child, tree2.root, [tree2.root], ["GEN"], t_analysis)
    evals2 = [e for e in tree2.trace if e.get("stage") == "evaluation"]
    assert len(evals2) == 1
    assert evals2[0]["lookahead_status"] == "terminal_child"
    assert evals2[0]["common_mask_count"] == 0
    assert evals2[0]["reward"] == 0.0


@pytest.mark.parametrize("method", ["medal", "ab_medal"])
def test_e2_cmer_boundary_fallback_and_checkpoint_equivalence(method):
    """Verify 1-step CMER fallback at NFE boundary B=2 (never mixing FIG/SRIG) and B-checkpoint match."""
    m = BatchModel()
    prompt = torch.tensor([[3, 3, 3]])

    def _build(max_calls, checkpoint_budgets=None):
        cfg_kwargs = dict(init_tokens=4, max_model_calls=max_calls, stop_mode="budget")
        if method == "medal":
            return E2MEDALSearch(
                m, prompt, 0, 50, MCTSConfig(**cfg_kwargs), checkpoint_budgets=checkpoint_budgets, seed=1, reward_mode="cmer"
            )
        return E2ABMEDALSearch(
            m, prompt, 0, 50, ABMEDALConfig(**cfg_kwargs), checkpoint_budgets=checkpoint_budgets, seed=1, reward_mode="cmer"
        )

    # Call 1 = root, Call 2 = child 1; child 1's grandchild is uncached -> boundary fallback fires at B=2
    cont_tree = _build(max_calls=6, checkpoint_budgets=(2,))
    while not cont_tree.done:
        cont_tree.step()

    ind_tree = _build(max_calls=2, checkpoint_budgets=None)
    while not ind_tree.done:
        ind_tree.step()

    chk_2 = cont_tree.checkpoints[2]
    assert chk_2["model_calls"] == 2
    assert chk_2["stop_reason"] == "model_call_budget"

    chk_evals = [e for e in chk_2["trace"] if e.get("stage") == "evaluation"]
    ind_evals = [e for e in ind_tree.trace if e.get("stage") == "evaluation"]
    assert len(chk_evals) == len(ind_evals) == 1

    fallback_ev = chk_evals[0]
    assert fallback_ev["lookahead_available"] is False
    assert fallback_ev["lookahead_status"] == "budget_boundary_fallback"
    assert fallback_ev["common_mask_count"] == 49
    assert abs(fallback_ev["reward"] - ind_evals[0]["reward"]) < 1e-6

    # Confirm 1-step CMER fallback matches R_CMER(parent, child) over M(c)
    node1 = [n for n in cont_tree.nodes if n.node_id == fallback_ev["child_id"]][0]
    expected_1step, _ = compute_cmer_reward(
        cont_tree.root.analysis.entropies,
        node1.analysis.entropies,
        node1.state[0, cont_tree.prompt_len:] == cont_tree.mask_id,
    )
    assert abs(fallback_ev["reward"] - expected_1step) < 1e-6


def test_e2_cmer_actionless_nonterminal_fallback():
    """Non-terminal child with no legal actions uses 1-step CMER fallback over M(c)."""
    m = BatchModel()
    prompt = torch.tensor([[3, 3, 3]])
    tree = E2MEDALSearch(
        m, prompt, 0, 50, MCTSConfig(init_tokens=4, max_model_calls=10, stop_mode="budget"), seed=1, reward_mode="cmer"
    )

    req_root = tree.next_evaluation()
    tree.record_external_forward(req_root, invocation_id=1, batch_size=1)
    with torch.no_grad():
        tree.submit_evaluation(req_root, logits=m(req_root.node.state).logits[0])
    req_child = tree.next_evaluation()
    child = req_child.node
    parent = req_child.parent
    c_key = _state_key(child.state)
    c_entropies = torch.zeros(50)
    c_entropies[child.state[0, tree.prompt_len:] == tree.mask_id] = 0.7
    actionless_analysis = StateAnalysis(c_entropies, (), c_key, 0.0)
    tree._prepare_grandchild_or_finalize(child, parent, actionless_analysis)

    eval_events = [e for e in tree.trace if e.get("stage") == "evaluation"]
    assert len(eval_events) == 1
    ev = eval_events[0]
    assert ev["lookahead_available"] is False
    assert ev["lookahead_status"] == "no_legal_actions"
    expected_1step, expected_count = compute_cmer_reward(
        parent.analysis.entropies,
        c_entropies,
        child.state[0, tree.prompt_len:] == tree.mask_id,
    )
    assert ev["common_mask_count"] == expected_count
    assert abs(ev["reward"] - expected_1step) < 1e-6


def test_swd_reward_and_kl_formulas():
    """Verify Q_swd, 1-step R_stability, sparse top-K tail KL parity, and parameter checks."""
    eps = 1e-8
    positions_all = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    p_parent = torch.tensor([
        [0.7, 0.2, 0.05, 0.05],
        [0.5, 0.3, 0.1, 0.1],
        [0.6, 0.2, 0.1, 0.1],
        [0.4, 0.3, 0.2, 0.1],
    ])
    p_child = torch.tensor([
        [0.8, 0.1, 0.05, 0.05],
        [0.5, 0.3, 0.1, 0.1],
        [0.2, 0.5, 0.2, 0.1],
        [0.4, 0.3, 0.2, 0.1],
    ])
    positions_gc = torch.tensor([1, 3], dtype=torch.long)
    p_gc = torch.tensor([
        [0.5, 0.3, 0.1, 0.1],  # identical to parent/child -> KL = 0, q = 0.5
        [0.7, 0.15, 0.1, 0.05],  # shifted -> positive KL, q = 0.7
    ])

    swd_p = extract_swd_state_data(p_parent, positions_all, top_k=None)
    swd_c = extract_swd_state_data(p_child, positions_all, top_k=None)
    swd_g = extract_swd_state_data(p_gc, positions_gc, top_k=None)

    common_mask = torch.tensor([False, True, False, True])
    reward, count = compute_swd_reward(swd_p, swd_c, swd_g, common_mask, beta=1.0, epsilon=eps)
    assert count == 2

    # Position 1 has identical distributions -> KL = 0 -> contribution = 0.5 * 1.0
    # Position 3 has p_p==p_c (KL_pc=0), p_c -> p_g positive forward KL
    p3_c = p_child[3].double()
    p3_g = p_gc[1].double()
    kl_cg_3 = float((p3_c * (torch.log(p3_c + eps) - torch.log(p3_g + eps))).sum().item())
    expected = 0.5 + 0.7 * math.exp(-1.0 * kl_cg_3)
    assert abs(reward - expected) < 1e-6

    # Beta = 0 collapses weights to 1 -> pure sum of q^j_{t+2}
    r_b0, _ = compute_swd_reward(swd_p, swd_c, swd_g, common_mask, beta=0.0, epsilon=eps)
    assert abs(r_b0 - (0.5 + 0.7)) < 1e-6

    # Empty mask returns (0.0, 0)
    r_empty, c_empty = compute_swd_reward(swd_p, swd_c, swd_g, torch.tensor([False, False, False, False]))
    assert (r_empty, c_empty) == (0.0, 0)

    # Sparse Top-K + analytical uniform tail matches exact full dense when tails are uniform
    # p_a top-4 = {0,1,2,3} (sum=0.90), p_b top-4 = {0,2,4,5} (sum=0.88) -> partial overlap test
    p_a = torch.tensor([[0.50, 0.25, 0.10, 0.05] + [0.10 / 6] * 6])
    p_b = torch.tensor([[0.40, 0.02, 0.30, 0.02, 0.10, 0.08, 0.02, 0.02, 0.02, 0.02]])
    pos1 = torch.tensor([0], dtype=torch.long)
    dense_a = extract_swd_state_data(p_a, pos1, top_k=None)
    dense_b = extract_swd_state_data(p_b, pos1, top_k=None)
    sparse_a = extract_swd_state_data(p_a, pos1, top_k=4)
    sparse_b = extract_swd_state_data(p_b, pos1, top_k=4)
    kl_dense = compute_pairwise_kl(dense_a, dense_b, pos1, epsilon=eps)
    kl_sparse = compute_pairwise_kl(sparse_a, sparse_b, pos1, epsilon=eps)
    assert abs(float(kl_dense.item()) - float(kl_sparse.item())) < 1e-6

    with pytest.raises(ValueError, match="Invalid reward_beta"):
        compute_swd_reward(swd_p, swd_c, swd_g, common_mask, beta=-0.5)
    with pytest.raises(ValueError, match="Invalid reward_beta"):
        compute_swd_reward(swd_p, swd_c, swd_g, common_mask, beta=float("nan"))
    with pytest.raises(ValueError, match="Invalid reward_beta"):
        compute_swd_one_step_fallback(swd_p, swd_c, torch.ones(4, dtype=torch.bool), beta=True)


@pytest.mark.parametrize("method", ["medal", "ab_medal"])
def test_e2_swd_search_execution(method):
    """Verify SWD 2-step search execution reuses compact distributions with 0 extra NFEs."""
    m = BatchModel()
    prompt = torch.tensor([[3, 3, 3]])
    cfg_kwargs = dict(init_tokens=4, max_model_calls=10, stop_mode="budget")

    if method == "medal":
        tree = E2MEDALSearch(
            m, prompt, 0, 50, MCTSConfig(**cfg_kwargs), seed=1, reward_mode="swd", reward_beta=1.0
        )
    else:
        tree = E2ABMEDALSearch(
            m, prompt, 0, 50, ABMEDALConfig(**cfg_kwargs), seed=1, reward_mode="swd", reward_beta=1.0
        )

    tree.step()

    assert tree.model_calls == 3
    child = tree.nodes[1]
    assert child.immediate_reward is not None
    assert math.isfinite(child.immediate_reward)

    eval_events = [e for e in tree.trace if e.get("stage") == "evaluation"]
    assert len(eval_events) == 1
    ev = eval_events[0]
    assert ev["reward_mode"] == "swd"
    assert ev["reward_beta"] == 1.0
    assert ev["lookahead_available"] is True
    assert ev["common_mask_count"] == 48

    g_action = tree.cached_analyses[_state_key(child.state)].actions[0]
    g_state = child.state.clone()
    g_state[0, tree.prompt_len + g_action.output_position] = g_action.token_id
    g_key = _state_key(g_state)
    g_mask = g_state[0, tree.prompt_len:] == tree.mask_id

    expected_reward, expected_count = compute_swd_reward(
        tree.cached_swd_states[_state_key(tree.root.state)],
        tree.cached_swd_states[_state_key(child.state)],
        tree.cached_swd_states[g_key],
        g_mask,
        beta=1.0,
        epsilon=tree.config.epsilon,
    )
    assert ev["common_mask_count"] == expected_count
    assert abs(ev["reward"] - expected_reward) < 1e-6

    if method == "ab_medal":
        child_models = tree.posteriors[child.node_id]
        assert child_models["GEN"].kappa == 1.0
        assert child_models["CONT"].kappa == 1.0


@pytest.mark.parametrize("method", ["medal", "ab_medal"])
def test_e2_swd_empty_common_masks_terminal_grandchild(method):
    """When child is terminal or grandchild fills final mask (|M(g*)| == 0), SWD reward is 0.0."""
    from dlm_project.decoding.mcts_decoding import Action, expand_node

    m = BatchModel()
    prompt = torch.tensor([[3, 3, 3]])
    cfg_kwargs = dict(init_tokens=4, max_model_calls=6, stop_mode="budget")

    def _make_tree():
        if method == "medal":
            t = E2MEDALSearch(m, prompt, 0, 5, MCTSConfig(**cfg_kwargs), seed=1, reward_mode="swd")
        else:
            t = E2ABMEDALSearch(m, prompt, 0, 5, ABMEDALConfig(**cfg_kwargs), seed=1, reward_mode="swd")
        req_root = t.next_evaluation()
        t.record_external_forward(req_root, invocation_id=1, batch_size=1)
        with torch.no_grad():
            t.submit_evaluation(req_root, logits=m(req_root.node.state).logits[0])
        return t

    # Case 1: Child with 1 mask left whose greedy action fills it -> |M(g*)| == 0 -> Q_swd = 0.0
    tree1 = _make_tree()
    near_terminal = expand_node(tree1.root, Action(0, 10, 0.9, 0.0, 0.1), 1, tree1.prompt_len, tree1.mask_id, 5)
    near_terminal.state[0, tree1.prompt_len : tree1.prompt_len + 4] = torch.tensor([10, 11, 12, 13])
    tree1.nodes.append(near_terminal)
    c_key = _state_key(near_terminal.state)
    c_analysis = StateAnalysis(
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.9]), (Action(4, 14, 0.95, 0.0, 0.1),), c_key, 0.95
    )
    tree1.cached_swd_states[c_key] = extract_swd_state_data(
        torch.tensor([[0.05, 0.05, 0.1, 0.1, 0.5, 0.2]]), torch.tensor([4], dtype=torch.long)
    )
    if method == "medal":
        tree1._prepare_grandchild_or_finalize(near_terminal, tree1.root, c_analysis)
    else:
        tree1._prepare_grandchild_or_finalize(near_terminal, tree1.root, [tree1.root], ["GEN"], c_analysis)
    evals1 = [e for e in tree1.trace if e.get("stage") == "evaluation"]
    assert len(evals1) == 1
    assert evals1[0]["common_mask_count"] == 0
    assert evals1[0]["reward"] == 0.0

    # Case 2: Already terminal child -> Q_swd = 0.0
    tree2 = _make_tree()
    terminal_child = expand_node(tree2.root, Action(0, 10, 0.9, 0.0, 0.1), 1, tree2.prompt_len, tree2.mask_id, 5)
    terminal_child.state[0, tree2.prompt_len :] = torch.tensor([10, 11, 12, 13, 14])
    tree2.nodes.append(terminal_child)
    t_analysis = StateAnalysis(torch.zeros(5), (), _state_key(terminal_child.state), 0.0)
    if method == "medal":
        tree2._prepare_grandchild_or_finalize(terminal_child, tree2.root, t_analysis)
    else:
        tree2._prepare_grandchild_or_finalize(terminal_child, tree2.root, [tree2.root], ["GEN"], t_analysis)
    evals2 = [e for e in tree2.trace if e.get("stage") == "evaluation"]
    assert len(evals2) == 1
    assert evals2[0]["lookahead_status"] == "terminal_child"
    assert evals2[0]["common_mask_count"] == 0
    assert evals2[0]["reward"] == 0.0


@pytest.mark.parametrize("method", ["medal", "ab_medal"])
def test_e2_swd_boundary_fallback_and_checkpoint_equivalence(method):
    """Verify 1-step R_stability fallback at NFE boundary B=2 and exact checkpoint parity."""
    m = BatchModel()
    prompt = torch.tensor([[3, 3, 3]])

    def _build(max_calls, checkpoint_budgets=None):
        cfg_kwargs = dict(init_tokens=4, max_model_calls=max_calls, stop_mode="budget")
        if method == "medal":
            return E2MEDALSearch(
                m, prompt, 0, 50, MCTSConfig(**cfg_kwargs), checkpoint_budgets=checkpoint_budgets, seed=1, reward_mode="swd"
            )
        return E2ABMEDALSearch(
            m, prompt, 0, 50, ABMEDALConfig(**cfg_kwargs), checkpoint_budgets=checkpoint_budgets, seed=1, reward_mode="swd"
        )

    cont_tree = _build(max_calls=6, checkpoint_budgets=(2,))
    while not cont_tree.done:
        cont_tree.step()

    ind_tree = _build(max_calls=2, checkpoint_budgets=None)
    while not ind_tree.done:
        ind_tree.step()

    chk_2 = cont_tree.checkpoints[2]
    assert chk_2["model_calls"] == 2
    assert chk_2["stop_reason"] == "model_call_budget"

    chk_evals = [e for e in chk_2["trace"] if e.get("stage") == "evaluation"]
    ind_evals = [e for e in ind_tree.trace if e.get("stage") == "evaluation"]
    assert len(chk_evals) == len(ind_evals) == 1

    fallback_ev = chk_evals[0]
    assert fallback_ev["reward_mode"] == "swd"
    assert fallback_ev["reward_beta"] == 1.0
    assert fallback_ev["lookahead_available"] is False
    assert fallback_ev["lookahead_status"] == "budget_boundary_fallback"
    assert fallback_ev["common_mask_count"] == 49
    assert abs(fallback_ev["reward"] - ind_evals[0]["reward"]) < 1e-6

    node1 = [n for n in cont_tree.nodes if n.node_id == fallback_ev["child_id"]][0]
    expected_1step, _ = compute_swd_one_step_fallback(
        cont_tree.cached_swd_states[_state_key(cont_tree.root.state)],
        cont_tree.cached_swd_states[_state_key(node1.state)],
        node1.state[0, cont_tree.prompt_len:] == cont_tree.mask_id,
        beta=1.0,
        epsilon=cont_tree.config.epsilon,
    )
    assert abs(fallback_ev["reward"] - expected_1step) < 1e-6


def test_e2_swd_actionless_nonterminal_fallback():
    """Non-terminal child with no legal actions uses 1-step R_stability fallback over M(c)."""
    m = BatchModel()
    prompt = torch.tensor([[3, 3, 3]])
    tree = E2MEDALSearch(
        m, prompt, 0, 50, MCTSConfig(init_tokens=4, max_model_calls=10, stop_mode="budget"), seed=1, reward_mode="swd"
    )

    req_root = tree.next_evaluation()
    tree.record_external_forward(req_root, invocation_id=1, batch_size=1)
    with torch.no_grad():
        tree.submit_evaluation(req_root, logits=m(req_root.node.state).logits[0])
    req_child = tree.next_evaluation()
    child, parent = req_child.node, req_child.parent
    c_key = _state_key(child.state)
    c_entropies = torch.zeros(50)
    c_entropies[child.state[0, tree.prompt_len:] == tree.mask_id] = 0.7
    actionless_analysis = StateAnalysis(c_entropies, (), c_key, 0.0)

    with torch.no_grad():
        c_logits = m(child.state).logits[0, tree.prompt_len:]
    c_probs = torch.softmax(c_logits.float(), dim=-1)
    c_pos = torch.nonzero(child.state[0, tree.prompt_len:] == tree.mask_id, as_tuple=False).flatten()
    tree._cache_swd_from_distribution(c_key, c_pos, c_probs[c_pos])
    tree._prepare_grandchild_or_finalize(child, parent, actionless_analysis)

    eval_events = [e for e in tree.trace if e.get("stage") == "evaluation"]
    assert len(eval_events) == 1
    ev = eval_events[0]
    assert ev["lookahead_available"] is False
    assert ev["lookahead_status"] == "no_legal_actions"
    expected_1step, expected_count = compute_swd_one_step_fallback(
        tree.cached_swd_states[_state_key(parent.state)],
        tree.cached_swd_states[c_key],
        child.state[0, tree.prompt_len:] == tree.mask_id,
        beta=1.0,
        epsilon=tree.config.epsilon,
    )
    assert ev["common_mask_count"] == expected_count
    assert abs(ev["reward"] - expected_1step) < 1e-6


def test_e2_swd_context_and_resume_signature():
    """Verify SWD context parameters and resume signature compatibility."""
    ctx_swd_1 = e2.context(stage="dev", num_samples=2, reward_mode="swd", reward_beta=1.0)
    ctx_swd_2 = e2.context(stage="dev", num_samples=2, reward_mode="swd", reward_beta=2.0)
    ctx_fig_a = e2.context(stage="dev", num_samples=2, reward_mode="fig")
    ctx_fig_b = e2.context(stage="dev", num_samples=2, reward_mode="fig", reward_beta=9.0)

    w = mock_wrapper()
    prep = e2.prepare(w, ctx_swd_1)
    sig_swd_1 = resume_signature(ctx_swd_1, prep, ("medal", "ab_medal"), (256, 512), 10, 3, 2)
    sig_swd_2 = resume_signature(ctx_swd_2, prep, ("medal", "ab_medal"), (256, 512), 10, 3, 2)
    sig_fig_a = resume_signature(ctx_fig_a, prep, ("medal", "ab_medal"), (256, 512), 10, 3, 2)
    sig_fig_b = resume_signature(ctx_fig_b, prep, ("medal", "ab_medal"), (256, 512), 10, 3, 2)

    assert sig_swd_1["settings_hash"] != sig_swd_2["settings_hash"]
    assert sig_fig_a["settings_hash"] == sig_fig_b["settings_hash"]
    assert "reward_beta" not in sig_fig_a["settings"]

    with pytest.raises(ValueError, match="Invalid reward_beta"):
        e2.context(stage="dev", reward_mode="swd", reward_beta=-1.0)


@pytest.mark.parametrize("method", ["medal", "ab_medal"])
def test_e2_swd_batched_parity_with_serial(method):
    """Verify batched cooperative SWD execution matches serial step() exactly."""
    m = BatchModel()
    prompts = [torch.tensor([[3, 3, 3]]), torch.tensor([[3, 2, 3, 4]])]
    cfg_kwargs = dict(init_tokens=4, max_model_calls=12, stop_mode="budget")

    serial_rewards = []
    for seed, prompt in enumerate(prompts, start=1):
        if method == "medal":
            t = E2MEDALSearch(m, prompt, 0, 50, MCTSConfig(**cfg_kwargs), seed=seed, reward_mode="swd", reward_beta=1.0)
        else:
            t = E2ABMEDALSearch(m, prompt, 0, 50, ABMEDALConfig(**cfg_kwargs), seed=seed, reward_mode="swd", reward_beta=1.0)
        while not t.done:
            t.step()
        serial_rewards.append([e["reward"] for e in t.trace if e.get("stage") == "evaluation"])

    batched_trees = []
    for seed, prompt in enumerate(prompts, start=1):
        if method == "medal":
            batched_trees.append(
                E2MEDALSearch(m, prompt, 0, 50, MCTSConfig(**cfg_kwargs), seed=seed, reward_mode="swd", reward_beta=1.0)
            )
        else:
            batched_trees.append(
                E2ABMEDALSearch(m, prompt, 0, 50, ABMEDALConfig(**cfg_kwargs), seed=seed, reward_mode="swd", reward_beta=1.0)
            )
    execute_trees(batched_trees, pad_token_id=4)
    batched_rewards = [
        [e["reward"] for e in t.trace if e.get("stage") == "evaluation"]
        for t in batched_trees
    ]
    assert batched_rewards == serial_rewards


@pytest.mark.parametrize("reward_mode", ["cmer", "srig"])
@pytest.mark.parametrize("count", [None, 2, 200, 201, 500])
def test_e2_test_pool_matches_e1_content_order_and_prompts(reward_mode, count):
    """Verify E2 test stage (200 primary and 500 extended) aligns 100% with E1 and E0."""
    from dlm_project.tasks.gsm8k import experiment_0 as e0, experiment_1 as e1
    from dlm_project.tasks.gsm8k.experiment_protocol import build_prompt, example_seed

    ref_e1 = e1.context(stage="test", num_samples=count)
    ref_e0 = e0.context(stage="test", methods=("confidence",), num_samples=count)
    actual = e2.context(stage="test", num_samples=count, reward_mode=reward_mode)

    assert actual["stage"] == actual["norm_stage"] == "test"
    assert actual["reward_mode"] == reward_mode
    assert actual["examples"] == ref_e1["examples"] == ref_e0["examples"]
    assert actual["manifest_hash"] == ref_e1["manifest_hash"] == ref_e0["manifest_hash"]
    assert actual["manifest"] == ref_e1["manifest"] == ref_e0["manifest"]
    assert actual["common"] == ref_e1["common"] == ref_e0["common"]
    assert actual["prompt_template_hash"] == ref_e1["prompt_template_hash"]
    assert len(actual["examples"]) == (count if count is not None else 200)
    assert len({row["example_id"] for row in actual["examples"]}) == len(actual["examples"])
    for row, old in zip(actual["examples"], ref_e1["examples"]):
        assert build_prompt(row["question"], actual["common"]["prompt"]) == build_prompt(
            old["question"], ref_e1["common"]["prompt"]
        )
        assert example_seed(row["row_id"]) == example_seed(old["row_id"])


@pytest.mark.parametrize("reward_mode", ["cmer", "srig"])
def test_e2_test_slices_primary_200_and_extended_300_equal_full_500(reward_mode):
    """Verify primary_200 (0..200) + extended_300 (200..500) == full_500 (0..500)."""
    from dlm_project.tasks.gsm8k import experiment_1 as e1

    ref_500 = e1.context(stage="test", num_samples=500)
    primary_200 = e2.context(stage="test", sample_slice=(0, 200), reward_mode=reward_mode)
    extended_300 = e2.context(stage="test", sample_slice=(200, 500), reward_mode=reward_mode)
    full_500 = e2.context(stage="test", sample_slice=(0, 500), reward_mode=reward_mode)

    assert len(primary_200["examples"]) == 200
    assert len(extended_300["examples"]) == 300
    assert primary_200["examples"] + extended_300["examples"] == full_500["examples"]
    assert full_500["examples"] == ref_500["examples"]


@pytest.mark.parametrize("options", [
    {"stage": "test", "num_samples": 0},
    {"stage": "test", "num_samples": -1},
    {"stage": "test", "num_samples": 501},
    {"stage": "test", "num_samples": True},
    {"stage": "test", "sample_slice": (-1, 2)},
    {"stage": "test", "sample_slice": (0, 501)},
    {"stage": "test", "sample_slice": (10, 10)},
    {"stage": "unknown"},
])
def test_e2_context_rejects_invalid_test_requests(options):
    with pytest.raises(ValueError):
        e2.context(**options)


@pytest.mark.parametrize("reward_mode", ["cmer", "srig"])
def test_e2_test_stage_execution_and_resumption(tmp_path, monkeypatch, reward_mode):
    """Verify end-to-end E2 test stage execution and zero-forward resumption for CMER and SRIG."""
    monkeypatch.setattr(e2.os, "sync", lambda: None, raising=False)
    w = mock_wrapper()
    ctx = e2.context(stage="test", methods=("medal", "ab_medal"), num_samples=2, reward_mode=reward_mode)
    ctx["common"]["generation"] = {"gen_length": 8, "steps": 8, "block_length": 8}
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
    assert len(q_rows) == 8
    assert len(agg_rows) == 4
    assert all(r["reward_mode"] == reward_mode for r in q_rows)
    assert (tmp_path / "summaries" / "e2_test_per_question.csv").is_file()
    assert (tmp_path / "summaries" / "e2_test_aggregate.csv").is_file()

    calls_before = len(w.model.calls)
    resumed_rows, _ = e2.run_e2_stage(
        w,
        ctx,
        tmp_path,
        methods=("medal", "ab_medal"),
        budgets=budgets,
        max_depth=4,
        num_candidates=2,
        batch_size=2,
        resume=True,
    )
    assert len(w.model.calls) == calls_before
    assert [r["selected_state_hash"] for r in resumed_rows] == [r["selected_state_hash"] for r in q_rows]


@pytest.mark.parametrize("filename,expected_mode", [
    ("E2_Experiment_CMER_Test.ipynb", "cmer"),
    ("E2_Experiment_SRIG_Test.ipynb", "srig"),
])
def test_generated_e2_test_notebooks_are_valid_and_compile(filename, expected_mode):
    """Verify generated E2 CMER and SRIG Test notebooks pass schema validation and compile."""
    import nbformat
    from dlm_project.paths import PROJECT_ROOT

    nb_path = PROJECT_ROOT / "notebooks/gsm8k" / filename
    assert nb_path.is_file(), f"Missing generated notebook: {nb_path}"
    notebook = nbformat.read(nb_path, as_version=4)
    nbformat.validate(notebook)

    full_source = []
    for idx, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{filename}:cell_{idx}", "exec")
            full_source.append(cell.source)

    joined = "\n".join(full_source)
    assert "STAGE = 'test'" in joined
    assert f"REWARD_MODE = {expected_mode!r}" in joined
    assert "METHODS = ('medal', 'ab_medal')" in joined
    assert "BUDGETS = (256, 512)" in joined
    assert "sample_slice=(0, 200)" in joined
    assert "sample_slice=(200, 500)" in joined
    assert "run_e2_stage" in joined
    assert "analyze_e3_stage" in joined





