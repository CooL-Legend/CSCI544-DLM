"""Tests for offline E3 Analysis Engine and publication figure generation."""
from pathlib import Path
import pytest

from dlm_project.tasks.gsm8k import experiment_1 as e1
from dlm_project.tasks.gsm8k import e3_analysis as e3
from tests.tasks.test_experiment_1 import mock_wrapper


def test_e3_offline_analysis_and_plots(tmp_path):
    w = mock_wrapper()
    ctx = e1.context(stage="dev", methods=("medal", "ab_medal"), num_samples=2)
    budgets = (2, 4)

    # 1. Produce mock E1 results
    e1.run_e1_stage(
        w, ctx, tmp_path, methods=("medal", "ab_medal"), budgets=budgets, max_depth=4, num_candidates=2
    )

    stage_dir = tmp_path / "dev"
    summaries_dir = tmp_path / "summaries"

    # 2. Run offline E3 analysis
    per_q_rows, agg_rows = e3.analyze_e3_stage(
        results_dir=stage_dir,
        output_dir=summaries_dir,
        stage="dev",
        methods=("medal", "ab_medal"),
        budgets=budgets,
    )

    assert len(per_q_rows) == 2 * 2 * 2  # 2 questions * 2 methods * 2 budgets
    assert len(agg_rows) == 4

    assert (summaries_dir / "e3_dev_per_question.csv").exists()
    assert (summaries_dir / "e3_dev_aggregate.csv").exists()

    # 3. Verify all 6 publication figures exist in both PNG and PDF
    fig_dir = summaries_dir / "e3_figures"
    assert fig_dir.is_dir()
    for i in range(1, 7):
        name = [
            "fig1_nfe_by_depth",
            "fig2_branching_factors",
            "fig3_candidate_density",
            "fig4_root_subtrees",
            "fig5_reward_redundancy",
            "fig6_accuracy_vs_budget",
        ][i - 1]
        png_file = fig_dir / f"{name}.png"
        pdf_file = fig_dir / f"{name}.pdf"
        assert png_file.exists(), f"Missing {png_file}"
        assert pdf_file.exists(), f"Missing {pdf_file}"
        assert png_file.stat().st_size > 1000, f"{png_file} too small"
        assert pdf_file.stat().st_size > 1000, f"{pdf_file} too small"

    # 4. Check specific E3 metrics
    for r in per_q_rows:
        assert "depth_efficiency_gap" in r
        assert "feasible_candidate_depth_dc" in r
        assert r["max_depth"] >= r["selected_depth"]
        assert r["depth_efficiency_gap"] == r["max_depth"] - r["selected_depth"]
        if r["method"] == "ab_medal":
            assert r["gen_count"] is not None
            assert r["cont_count"] is not None
        else:
            assert r["gen_count"] is None
            assert r["cont_count"] is None
