"""Offline E3 Analysis Engine: Tree allocation, structural dynamics, and publication figures.

Pure offline evaluation reading E1 search traces and checkpoints with ZERO model calls.
Generates:
  - e3_per_question.csv
  - e3_aggregate.csv
  - 7 publication figures (PNG and PDF)
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dlm_project.decoding import search_audit
from dlm_project.experiments.sweep_report import wilson_interval, write_csv
from dlm_project.tasks.gsm8k.e1_variants import METHODS, is_adaptive, variant


def load_question_checkpoints(q_dir: Path, budgets: Sequence[int] = (256, 512, 1024)):
    """Load audit and metrics for each budget checkpoint in a question directory."""
    checkpoints = {}
    checkpoints_dir = q_dir / "checkpoints"
    for b in budgets:
        b_dir = checkpoints_dir / str(b)
        if not b_dir.is_dir():
            continue
        metrics_file = b_dir / "metrics.json"
        tree_file = b_dir / "tree.json"
        selection_file = b_dir / "selection.json"
        comp_file = b_dir / "completion.json"

        if metrics_file.exists() and tree_file.exists():
            metrics = json.loads(metrics_file.read_text())
            tree = json.loads(tree_file.read_text())
            selection = json.loads(selection_file.read_text()) if selection_file.exists() else {}
            completion = json.loads(comp_file.read_text()) if comp_file.exists() else {}
            checkpoints[b] = {
                "metrics": metrics,
                "tree": tree,
                "selection": selection,
                "completion": completion,
            }
    return checkpoints


def compute_e3_question_record(method: str, budget: int, q_data: Dict[str, Any], example_id: str):
    """Compute detailed E3 metrics for one question-budget outcome."""
    metrics = q_data["metrics"]
    tree = q_data["tree"]
    stats = tree.get("statistics", {})
    nodes = tree.get("nodes", [])

    # Depth distribution
    depth_counts = Counter(n["depth"] for n in nodes)
    children = [n for n in nodes if n.get("parent_id") is not None]
    max_depth = max(depth_counts.keys(), default=0)
    mean_child_depth = (sum(n["depth"] for n in children) / len(children)) if children else 0.0

    # Branching
    expanded_parents = stats.get("expanded_parents", 0)
    branching_mean = stats.get("branching_factor_mean", 0.0)
    branching_max = stats.get("branching_factor_max", 0)

    # Redundancy
    unique_states = stats.get("unique_evaluated_states", len(nodes))
    duplicates = stats.get("duplicate_state_count", 0)
    duplicate_pct = stats.get("duplicate_state_percentage", 0.0)

    # Rewards
    neg_ig_count = stats.get("negative_ig_count", 0)
    neg_ig_pct = stats.get("negative_ig_percentage", 0.0)
    ig_mean = stats.get("ig_mean", 0.0)
    selected_q = metrics.get("selected_q", 0.0)

    # Candidate density & efficiency
    distinct_by_depth = metrics.get("distinct_candidates_by_depth", {})
    if isinstance(distinct_by_depth, str):
        distinct_by_depth = json.loads(distinct_by_depth)
    d_selected = metrics.get("selected_depth", 0)
    depth_gap = max_depth - d_selected

    # D_C: deepest depth with at least 3 distinct states
    d_c = 0
    for d_str, count in distinct_by_depth.items():
        d_int = int(d_str)
        if count >= metrics.get("num_candidates", 3) and d_int > d_c:
            d_c = d_int

    # Candidate diversity
    diversity = stats.get("candidate_diversity", {})
    mean_jaccard = diversity.get("mean_jaccard_distance")
    shared_prefix = diversity.get("shared_prefix_length")

    # Compute concentration: largest root subtree share
    root_subtrees = stats.get("nfes_per_root_subtree", {})
    subtree_counts = list(root_subtrees.values()) if root_subtrees else []
    largest_subtree_share = (max(subtree_counts) / sum(subtree_counts)) if subtree_counts and sum(subtree_counts) else 0.0

    # Adaptive policy (AB-MEDAL only)
    decision_counts = stats.get("decision_counts") or {}
    forced_counts = stats.get("forced_decision_counts") or {}
    gen_count = decision_counts.get("GEN", 0) if is_adaptive(method) else None
    cont_count = decision_counts.get("CONT", 0) if is_adaptive(method) else None
    forced_gen = forced_counts.get("GEN", 0) if is_adaptive(method) else None
    forced_cont = forced_counts.get("CONT", 0) if is_adaptive(method) else None

    return {
        "example_id": example_id,
        "method": method,
        **{k: variant(method)[k] for k in ("base_method", "reward_mode")},
        "search_budget": budget,
        "exact_match": metrics.get("exact_match", False),
        "parse_success": metrics.get("parse_success", False),
        "total_nfes": metrics.get("total_nfes", budget + 256 - d_selected),
        "search_nfes": metrics.get("search_nfes", budget),
        "completion_nfes": metrics.get("completion_nfes", 256 - d_selected),
        "tree_nfes": sum(depth_counts.values()),
        "evaluated_children": len(children),
        **{key: stats.get(key, 0) for key in (
            "temporary_candidate_nfes", "adalook_extra_nfes", "adalook_total_extra_nfes",
            "adalook_group_count", "adalook_trigger_rate", "adalook_mean_depth")},
        "max_depth": max_depth,
        "selected_depth": d_selected,
        "depth_efficiency_gap": depth_gap,
        "feasible_candidate_depth_dc": d_c,
        "mean_evaluated_child_depth": mean_child_depth,
        "expanded_parents": expanded_parents,
        "branching_factor_mean": branching_mean,
        "branching_factor_max": branching_max,
        "unique_evaluated_states": unique_states,
        "duplicate_state_count": duplicates,
        "duplicate_state_percentage": duplicate_pct,
        "negative_ig_count": neg_ig_count,
        "negative_ig_percentage": neg_ig_pct,
        "ig_mean": ig_mean,
        "selected_q": selected_q,
        "mean_jaccard_distance": mean_jaccard,
        "shared_prefix_length": shared_prefix,
        "largest_root_subtree_share": largest_subtree_share,
        "root_subtree_count": len(subtree_counts),
        "gen_count": gen_count,
        "cont_count": cont_count,
        "forced_gen_count": forced_gen,
        "forced_cont_count": forced_cont,
        "fallback_used": metrics.get("fallback_used", False),
        "nfes_per_depth": json.dumps({str(k): v for k, v in sorted(depth_counts.items())}),
    }


def analyze_e3_stage(results_dir: Path, output_dir: Path, stage: str = "dev",
                     methods: Sequence[str] = METHODS,
                     budgets: Sequence[int] = (256, 512, 1024)):
    """Analyze all saved traces for a stage, output E3 CSVs and 6 figures."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    figures_dir = output_dir / "e3_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    e3_per_question_rows = []

    for method in methods:
        m_dir = results_dir / method
        if not m_dir.is_dir():
            continue
        for q_dir in sorted(m_dir.iterdir()):
            if not q_dir.is_dir():
                continue
            example_id = q_dir.name
            chk_data = load_question_checkpoints(q_dir, budgets=budgets)
            for b in budgets:
                if b in chk_data:
                    row = compute_e3_question_record(method, b, chk_data[b], example_id)
                    e3_per_question_rows.append(row)

    # 1. Write e3_per_question.csv
    per_q_path = output_dir / f"e3_{stage}_per_question.csv"
    write_csv(per_q_path, e3_per_question_rows)
    write_csv(output_dir / "e3_per_question.csv", e3_per_question_rows)

    # 2. Write e3_aggregate.csv
    agg_rows = []
    for b in sorted(budgets):
        for method in methods:
            subset = [r for r in e3_per_question_rows if r["method"] == method and r["search_budget"] == b]
            if not subset:
                continue
            n = len(subset)
            acc = sum(r["exact_match"] for r in subset) / n
            ci_low, ci_high = wilson_interval(sum(r["exact_match"] for r in subset), n)

            agg_rows.append({
                "stage": stage,
                "method": method,
                **{k: variant(method)[k] for k in ("base_method", "reward_mode")},
                "search_budget": b,
                "sample_count": n,
                "accuracy": acc,
                "accuracy_percent": acc * 100.0,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "mean_max_depth": sum(r["max_depth"] for r in subset) / n,
                "mean_selected_depth": sum(r["selected_depth"] for r in subset) / n,
                "mean_depth_gap": sum(r["depth_efficiency_gap"] for r in subset) / n,
                "mean_feasible_depth_dc": sum(r["feasible_candidate_depth_dc"] for r in subset) / n,
                "mean_child_depth": sum(r["mean_evaluated_child_depth"] for r in subset) / n,
                "mean_expanded_parents": sum(r["expanded_parents"] for r in subset) / n,
                "mean_branching_factor": sum(r["branching_factor_mean"] for r in subset) / n,
                "mean_duplicate_pct": sum(r["duplicate_state_percentage"] for r in subset) / n,
                "mean_negative_ig_pct": sum(r["negative_ig_percentage"] for r in subset) / n,
                "mean_ig": sum(r["ig_mean"] for r in subset) / n,
                "mean_selected_q": sum(r["selected_q"] for r in subset) / n,
                **{f"mean_{key}": sum(r[key] for r in subset) / n for key in (
                    "tree_nfes", "evaluated_children", "temporary_candidate_nfes",
                    "adalook_extra_nfes", "adalook_total_extra_nfes", "adalook_trigger_rate",
                    "adalook_mean_depth")},
                "mean_largest_subtree_share": sum(r["largest_root_subtree_share"] for r in subset) / n,
                "mean_jaccard_diversity": sum(r["mean_jaccard_distance"] for r in subset if r["mean_jaccard_distance"] is not None) / max(1, sum(1 for r in subset if r["mean_jaccard_distance"] is not None)),
                "gen_proportion": (sum(r["gen_count"] for r in subset if r["gen_count"] is not None) /
                                  max(1, sum((r["gen_count"] or 0) + (r["cont_count"] or 0) for r in subset))) if is_adaptive(method) else None,
            })

    agg_path = output_dir / f"e3_{stage}_aggregate.csv"
    write_csv(agg_path, agg_rows)
    write_csv(output_dir / "e3_aggregate.csv", agg_rows)

    # 3. Generate publication plots
    if e3_per_question_rows:
        generate_e3_figures(e3_per_question_rows, agg_rows, figures_dir, budgets=budgets, methods=methods)

    print(f"E3 offline analysis complete: saved {per_q_path}, {agg_path}, and figures to {figures_dir}")
    return e3_per_question_rows, agg_rows


def generate_e3_figures(rows: List[Dict[str, Any]], agg_rows: List[Dict[str, Any]],
                        figures_dir: Path, budgets: Sequence[int] = (256, 512, 1024),
                        methods: Sequence[str] = METHODS):
    """Generate four-variant tree figures and the explicit compute breakdown."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    palette = {"medal": "#2b5c8f", "ab_medal": "#d95f02",
               "medal_adalook": "#368650", "ab_medal_adalook": "#9656a8"}
    labels = {method: variant(method)["label"] for method in methods}

    # Figure 1: NFE allocation by depth
    fig, axes = plt.subplots(1, len(budgets), figsize=(5 * len(budgets), 4), sharey=True)
    if len(budgets) == 1:
        axes = [axes]
    for idx, b in enumerate(sorted(budgets)):
        ax = axes[idx]
        for method in methods:
            sub = [r for r in rows if r["method"] == method and r["search_budget"] == b]
            if not sub:
                continue
            # Accumulate depth NFEs
            depth_totals = defaultdict(int)
            for r in sub:
                counts = json.loads(r["nfes_per_depth"])
                for d_str, c in counts.items():
                    depth_totals[int(d_str)] += c
            depths = sorted(depth_totals.keys())
            means = [depth_totals[d] / len(sub) for d in depths]
            ax.plot(depths, means, marker="o", label=labels[method], color=palette[method], lw=2)
        ax.set_title(f"Search Budget B = {b}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Depth", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Mean Real-Tree NFEs (excludes AdaLook)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True)
    fig.suptitle("Fig 1: Real-Tree NFE Allocation by Search Depth", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, figures_dir / "fig1_nfe_by_depth")

    # Figure 2: Branching factor distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_indices = np.arange(len(budgets))
    width = 0.8 / len(methods)
    for m_idx, method in enumerate(methods):
        means = []
        for b in sorted(budgets):
            sub = [r["branching_factor_mean"] for r in rows if r["method"] == method and r["search_budget"] == b]
            means.append(np.mean(sub) if sub else 0.0)
        ax.bar(x_indices + (m_idx - (len(methods) - 1) / 2) * width, means, width, label=labels[method], color=palette[method], alpha=0.9)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"B={b}" for b in sorted(budgets)], fontsize=10)
    ax.set_ylabel("Mean Children per Expanded Parent", fontsize=10)
    ax.set_title("Fig 2: Tree Branching Factor by Search Budget", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    _save_fig(fig, figures_dir / "fig2_branching_factors")

    # Figure 3: Candidate density vs depth
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in methods:
        sub = [r for r in rows if r["method"] == method and r["search_budget"] == max(budgets)]
        if not sub:
            continue
        max_d = [r["max_depth"] for r in sub]
        sel_d = [r["selected_depth"] for r in sub]
        feas_d = [r["feasible_candidate_depth_dc"] for r in sub]
        ax.plot(["Max Depth", "Feasible D_C", "Selected d*"],
                [np.mean(max_d), np.mean(feas_d), np.mean(sel_d)],
                marker="s", lw=2, label=f"{labels[method]} (B={max(budgets)})", color=palette[method])
    ax.set_ylabel("Tree Depth", fontsize=10)
    ax.set_title(f"Fig 3: Candidate Depth Feasibility Gap (B={max(budgets)})", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    _save_fig(fig, figures_dir / "fig3_candidate_density")

    # Figure 4: Root-subtree NFE allocation
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m_idx, method in enumerate(methods):
        shares = []
        for b in sorted(budgets):
            sub = [r["largest_root_subtree_share"] * 100 for r in rows if r["method"] == method and r["search_budget"] == b]
            shares.append(np.mean(sub) if sub else 0.0)
        ax.plot([f"B={b}" for b in sorted(budgets)], shares, marker="o", lw=2, label=labels[method], color=palette[method])
    ax.set_ylabel("Largest Root-Subtree Share of Tree NFEs (%)", fontsize=10)
    ax.set_title("Fig 4: Root Subtree Compute Concentration", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    _save_fig(fig, figures_dir / "fig4_root_subtrees")

    # Figure 5: Reward and redundancy analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for method in methods:
        neg_ig = [np.mean([r["negative_ig_percentage"] for r in rows if r["method"] == method and r["search_budget"] == b]) for b in sorted(budgets)]
        dup_pct = [np.mean([r["duplicate_state_percentage"] for r in rows if r["method"] == method and r["search_budget"] == b]) for b in sorted(budgets)]
        b_labels = [f"B={b}" for b in sorted(budgets)]
        ax1.plot(b_labels, neg_ig, marker="o", lw=2, label=labels[method], color=palette[method])
        ax2.plot(b_labels, dup_pct, marker="s", lw=2, label=labels[method], color=palette[method])
    ax1.set_ylabel("Negative IG Rate (%)", fontsize=10)
    ax1.set_title("Negative Immediate IG (all reward modes)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()
    ax2.set_ylabel("Duplicate State Rate (%)", fontsize=10)
    ax2.set_title("Search State Redundancy", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    fig.suptitle("Fig 5: Reward Dynamics and Search Redundancy", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, figures_dir / "fig5_reward_redundancy")

    # Figure 6: Accuracy vs search budget
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in methods:
        sub_agg = [r for r in agg_rows if r["method"] == method]
        sub_agg.sort(key=lambda r: r["search_budget"])
        if not sub_agg:
            continue
        budgets_x = [r["search_budget"] for r in sub_agg]
        accs = [r["accuracy_percent"] for r in sub_agg]
        yerr_low = [r["accuracy_percent"] - r["ci_95_low"] * 100 for r in sub_agg]
        yerr_high = [r["ci_95_high"] * 100 - r["accuracy_percent"] for r in sub_agg]
        ax.errorbar(budgets_x, accs, yerr=[yerr_low, yerr_high], marker="o", lw=2, capsize=4,
                    label=labels[method], color=palette[method])
    ax.set_xlabel("Search Forward Budget B", fontsize=10)
    ax.set_ylabel("GSM8K Accuracy (%) [Wilson 95% CI]", fontsize=10)
    ax.set_title("Fig 6: Exact-Match Accuracy vs MCTS Compute Budget", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)
    plt.tight_layout()
    _save_fig(fig, figures_dir / "fig6_accuracy_vs_budget")


    # Figure 7: all search costs, including temporary states absent from the tree.
    fig, axes = plt.subplots(1, len(budgets), figsize=(6 * len(budgets), 4.5), squeeze=False)
    for ax, b in zip(axes[0], sorted(budgets)):
        bottom = np.zeros(len(methods))
        for key, label, color in (
            ("tree_nfes", "Real tree", "#2b5c8f"),
            ("temporary_candidate_nfes", "Temporary comparisons", "#d95f02"),
            ("adalook_extra_nfes", "AdaLook rollout", "#368650"),
        ):
            values = []
            for method in methods:
                sub = [r[key] for r in rows if r["method"] == method and r["search_budget"] == b]
                values.append(np.mean(sub) if sub else 0.0)
            ax.bar(np.arange(len(methods)), values, bottom=bottom, label=label, color=color)
            bottom += np.asarray(values)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels([labels[m] for m in methods], rotation=25, ha="right", fontsize=9)
        ax.set_title(f"B={b}")
        ax.set_ylabel("Mean Search NFEs")
        ax.legend(fontsize=8)
    fig.suptitle("Fig 7: Search Compute Breakdown")
    fig.tight_layout()
    _save_fig(fig, figures_dir / "fig7_compute_breakdown")


def _save_fig(fig, base_path: Path):
    fig.savefig(f"{base_path}.png", dpi=300)
    fig.savefig(f"{base_path}.pdf")
    plt.close(fig)
