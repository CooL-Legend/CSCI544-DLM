"""Closed-setup CLI runner for the locked Experiment 1 (E1) + Experiment 3 (E3) Test Protocol.

Protocol:
  - Evaluates the predetermined GSM8K test questions under all 4 locked configurations:
      1. MEDAL    @ B = 256 (Condition A, L_c = 10, C_min = 3)
      2. AB-MEDAL @ B = 256 (Condition A, L_c = 10, C_min = 3)
      3. MEDAL    @ B = 512 (Condition B, L_c = 10, C_min = 3)
      4. AB-MEDAL @ B = 512 (Condition B, L_c = 10, C_min = 3)
  - Continuous search up to B = 512 with verified prefix checkpointing at B in {256, 512}.
  - Cooperative multi-question batching (--batch-size) with varlen FlashAttention-2.
  - Supports 200 as the planned primary test size (--num-samples 200) and seamless
    extension to 500 (--num-samples 500) using the exact same sample_seed=1 ordering
    and automatic checkpoint resumption (--resume).
  - Runs offline E3 diagnostic analysis (0 extra NFEs) to explain why or why not
    AB-MEDAL improves accuracy at a matched search budget.

Examples:
    # 2-example B=512 closed-setup verification check across all 4 configs:
    python scripts/run_e1_test.py --num-samples 2 --batch-size 2

    # Planned primary test run (200 predetermined test questions):
    python scripts/run_e1_test.py --num-samples 200 --batch-size 4

    # Seamless extension to 500 predetermined test questions (resumes at question 201):
    python scripts/run_e1_test.py --num-samples 500 --batch-size 4
"""

import argparse
from math import comb
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence

import torch

from dlm_project.experiments.resources import finish_reporting

from dlm_project.experiments.sweep_report import write_csv
from dlm_project.tasks.gsm8k.e3_analysis import analyze_e3_stage
from dlm_project.tasks.gsm8k.experiment_1 import (
    MAX_SEARCH_DEPTH,
    MIN_CANDIDATES,
    context,
    load_model,
    run_e1_stage,
)

LOCKED_TEST_METHODS = ("medal", "ab_medal")
LOCKED_TEST_BUDGETS = (256, 512)


def mcnemar_exact_pvalue(b_wins: int, c_losses: int) -> float:
    """Two-sided exact binomial McNemar p-value on discordant pairs (b_wins vs c_losses)."""
    n = b_wins + c_losses
    if n == 0:
        return 1.0
    k = min(b_wins, c_losses)
    tail = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def compute_paired_comparison(
    q_rows: Sequence[Dict[str, Any]],
    budgets: Sequence[int] = LOCKED_TEST_BUDGETS,
) -> List[Dict[str, Any]]:
    """Compute paired AB-MEDAL vs MEDAL accuracy & McNemar stats at each matched budget."""
    by_key = {(r["example_id"], r["method"], int(r["search_budget"])): r for r in q_rows}
    ex_ids = sorted({r["example_id"] for r in q_rows})
    paired_rows = []

    for budget in sorted(budgets):
        pairs = [
            (by_key[(ex_id, "medal", budget)], by_key[(ex_id, "ab_medal", budget)])
            for ex_id in ex_ids
            if (ex_id, "medal", budget) in by_key and (ex_id, "ab_medal", budget) in by_key
        ]
        if not pairs:
            continue
        n = len(pairs)
        both_correct = sum(1 for m, ab in pairs if m["exact_match"] and ab["exact_match"])
        ab_only = sum(1 for m, ab in pairs if (not m["exact_match"]) and ab["exact_match"])
        medal_only = sum(1 for m, ab in pairs if m["exact_match"] and (not ab["exact_match"]))
        both_wrong = sum(1 for m, ab in pairs if (not m["exact_match"]) and (not ab["exact_match"]))

        medal_acc = (both_correct + medal_only) / n
        ab_acc = (both_correct + ab_only) / n
        delta_acc = ab_acc - medal_acc
        p_val = mcnemar_exact_pvalue(ab_only, medal_only)

        paired_rows.append({
            "search_budget": budget,
            "paired_questions": n,
            "medal_accuracy": medal_acc,
            "ab_medal_accuracy": ab_acc,
            "accuracy_delta": delta_acc,
            "accuracy_delta_percent": 100.0 * delta_acc,
            "both_correct": both_correct,
            "ab_medal_only_correct": ab_only,
            "medal_only_correct": medal_only,
            "both_wrong": both_wrong,
            "mcnemar_exact_p": p_val,
            "medal_mean_total_nfe": sum(m["total_nfes"] for m, _ in pairs) / n,
            "ab_medal_mean_total_nfe": sum(ab["total_nfes"] for _, ab in pairs) / n,
        })
    return paired_rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of predetermined GSM8K test questions: 2 (smoke test), 200 (primary), or 500 (extended).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Cooperative multi-question batch size (default: 4).",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=list(LOCKED_TEST_BUDGETS),
        help="Matched search budgets to checkpoint (default: 256 512).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("medal", "ab_medal"),
        default=list(LOCKED_TEST_METHODS),
        help="Methods to evaluate (default: medal ab_medal).",
    )
    parser.add_argument("--max-depth", type=int, default=MAX_SEARCH_DEPTH)
    parser.add_argument("--num-candidates", type=int, default=MIN_CANDIDATES)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH"),
        help="Local checkpoint directory or HuggingFace repo ID (defaults to MODEL_PATH env var or pinned Hub ID).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=bool(os.environ.get("HF_HUB_OFFLINE", "") == "1"),
        help="Require all model/tokenizer files to be read strictly from local cache/path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("ARTIFACT_ROOT", "results/e1_test_locked")),
        help="Directory for saving locked E1 and E3 test artifacts.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        default=True,
        help="Disable per-question checkpoint resumption.",
    )
    parser.add_argument(
        "--no-e3",
        action="store_false",
        dest="run_e3",
        default=True,
        help="Skip offline E3 diagnostic analysis after E1 completes.",
    )
    parser.add_argument("--profile-resources", action="store_true",
                        help="Measure actual batch memory, forwards and elapsed time in resource sidecars.")
    args = parser.parse_args(argv)
    entrypoint_started = perf_counter()

    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but unavailable on this host.")

    stage = "test"
    methods = tuple(args.methods)
    budgets = tuple(sorted(args.budgets))

    print("=" * 78)
    print("LOCKED E1 + E3 TEST PROTOCOL (CLOSED-SETUP RUNNER)")
    print(f"  Stage          : {stage} (predetermined seed=1 ordering)")
    print(f"  Test Questions : {args.num_samples} (200 primary / 500 extended protocol)")
    print(f"  Configurations : {len(methods) * len(budgets)} ({[f'{m}@{b}' for b in budgets for m in methods]})")
    print(f"  Search Depth   : L_c = {args.max_depth} | Min Candidates C = {args.num_candidates}")
    print(f"  Batch Size     : {args.batch_size} | Resume: {args.resume}")
    print(f"  Output Dir     : {args.output_dir.resolve()}")
    print("=" * 78, flush=True)

    ctx = context(
        stage=stage,
        methods=methods,
        budgets=budgets,
        max_depth=args.max_depth,
        num_candidates=args.num_candidates,
        num_samples=args.num_samples,
    )

    print(f"\nLoading model on {args.device} (local_files_only={args.local_files_only})...", flush=True)
    wrapper = load_model(
        ctx,
        model_path=args.model_path,
        device=args.device,
        local_files_only=args.local_files_only,
    )

    started = perf_counter()
    q_rows, agg_rows = run_e1_stage(
        wrapper,
        ctx,
        args.output_dir,
        methods=methods,
        budgets=budgets,
        max_depth=args.max_depth,
        num_candidates=args.num_candidates,
        resume=args.resume,
        batch_size=args.batch_size,
        profile_resources=args.profile_resources,
    )
    total_time = perf_counter() - started

    reporting_started = perf_counter()
    offline_seconds = 0.0
    summaries_dir = args.output_dir / "summaries"
    paired_rows = compute_paired_comparison(q_rows, budgets=budgets)
    if paired_rows:
        write_csv(summaries_dir / f"e1_{stage}_paired_comparison.csv", paired_rows)
        write_csv(summaries_dir / "e1_paired_comparison.csv", paired_rows)

    print(f"\nE1 Test Run Completed in {total_time:.2f}s ({len(q_rows)} configuration outcomes):")
    for row in agg_rows:
        print(
            f"  [{row['method'].upper():8s} @ B={row['search_budget']:4d}] "
            f"Acc={row['accuracy_percent']:5.1f}% (95% CI [{row['ci_95_low']*100:5.1f}%, {row['ci_95_high']*100:5.1f}%]) | "
            f"N={row['sample_count']} | Mean Total NFE={row['mean_total_nfe']:.1f} | "
            f"Mean Depth={row['mean_selected_depth']:.2f}"
        )

    if paired_rows:
        print("\nPaired Comparison (AB-MEDAL vs MEDAL at Matched Search Budget):")
        for p in paired_rows:
            print(
                f"  [Budget B={p['search_budget']:4d}] "
                f"Delta={p['accuracy_delta_percent']:+5.1f}% "
                f"(AB-MEDAL {p['ab_medal_accuracy']*100:5.1f}% vs MEDAL {p['medal_accuracy']*100:5.1f}%) | "
                f"Wins/Losses={p['ab_medal_only_correct']}/{p['medal_only_correct']} | "
                f"McNemar p={p['mcnemar_exact_p']:.4f}"
            )

    if args.run_e3:
        offline_started = perf_counter()
        print("\nLaunching Offline E3 Diagnostic Analysis (0 extra NFEs)...", flush=True)
        e3_rows, e3_agg = analyze_e3_stage(
            results_dir=args.output_dir / stage,
            output_dir=summaries_dir,
            stage=stage,
            methods=methods,
            budgets=budgets,
        )
        print(f"E3 Analysis Complete: {len(e3_rows)} question-budget records and 6 figures saved to {summaries_dir / 'figures'}.")

        offline_seconds = perf_counter() - offline_started

    if args.profile_resources:
        finish_reporting(args.output_dir,
                         reporting_seconds=perf_counter() - reporting_started - offline_seconds,
                         offline_seconds=offline_seconds,
                         total_seconds=perf_counter() - entrypoint_started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

