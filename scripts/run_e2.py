"""CLI entry point for Experiment 2 (E2: Two-Step Lookahead FIG Reward).

Compares MEDAL vs. AB-MEDAL under a two-step lookahead Fractional Information Gain (FIG)
reward function across matched search budgets (B in {256, 512}).

Examples:
    # 2-example smoke test on Dev (B=256, B=512):
    python scripts/run_e2.py --stage dev --num-samples 2 --batch-size 2

    # Canonical Dev run (5 questions):
    python scripts/run_e2.py --stage dev --batch-size 4

    # Canonical Evaluation run (original 20 questions):
    python scripts/run_e2.py --stage eval --eval-split canonical_20 --batch-size 4

    # Extended Evaluation run (additional 50 questions):
    python scripts/run_e2.py --stage eval --eval-split extended_50 --batch-size 4

    # Full Evaluation run (all 70 questions):
    python scripts/run_e2.py --stage eval --eval-split all_70 --batch-size 4
"""

import argparse
from math import comb
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from dlm_project.experiments.sweep_report import write_csv
from dlm_project.tasks.gsm8k.experiment_2 import (
    CHECKPOINT_BUDGETS,
    MAX_SEARCH_DEPTH,
    MIN_CANDIDATES,
    context,
    load_model,
    run_e2_stage,
)

LOCKED_METHODS = ("medal", "ab_medal")


def mcnemar_exact_pvalue(b_wins: int, c_losses: int) -> float:
    """Two-sided exact binomial McNemar p-value on discordant pairs."""
    n = b_wins + c_losses
    if n == 0:
        return 1.0
    k = min(b_wins, c_losses)
    tail = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def compute_paired_comparison(
    q_rows: Sequence[Dict[str, Any]],
    budgets: Sequence[int] = CHECKPOINT_BUDGETS,
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
        "--stage",
        choices=("dev", "eval", "evaluation"),
        default="dev",
        help="Experiment stage: 'dev' (5 questions) or 'eval'/'evaluation' (up to 70 questions).",
    )
    parser.add_argument(
        "--eval-split",
        choices=("canonical_20", "extended_50", "all_70"),
        default="all_70",
        help="Evaluation split slice: 'canonical_20' (first 20 questions), 'extended_50' (questions 20 to 70), or 'all_70' (all 70).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("medal", "ab_medal"),
        default=list(LOCKED_METHODS),
        help="Methods to evaluate (default: medal ab_medal).",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=list(CHECKPOINT_BUDGETS),
        help="Matched search budgets to checkpoint (default: 256 512).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Cooperative multi-question batch size (default: 4).",
    )
    parser.add_argument("--max-depth", type=int, default=MAX_SEARCH_DEPTH)
    parser.add_argument("--num-candidates", type=int, default=MIN_CANDIDATES)
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Optional prefix count to limit the number of questions for testing.",
    )
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
        default=Path("results/e2"),
        help="Directory for saving E2 artifacts (default: results/e2).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        default=True,
        help="Disable per-question checkpoint resumption.",
    )
    parser.add_argument(
        "--run-e3",
        action="store_true",
        help="Launch offline E3 diagnostic analysis after E2 completes.",
    )
    args = parser.parse_args(argv)

    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but unavailable on this host.")

    norm_stage = "evaluation" if args.stage in ("eval", "evaluation") else args.stage
    sample_slice: Optional[Tuple[int, int]] = None
    if norm_stage == "evaluation":
        if args.eval_split == "canonical_20":
            sample_slice = (0, 20)
        elif args.eval_split == "extended_50":
            sample_slice = (20, 70)
        elif args.eval_split == "all_70":
            sample_slice = (0, 70)

    methods = tuple(args.methods)
    budgets = tuple(sorted(args.budgets))

    print("=" * 78)
    print("EXPERIMENT 2: TWO-STEP LOOKAHEAD MEDAL vs. AB-MEDAL")
    print(f"  Stage          : {args.stage} (normalized: {norm_stage})")
    if norm_stage == "evaluation":
        print(f"  Eval Split     : {args.eval_split} (slice: {sample_slice})")
    print(f"  Configurations : {len(methods) * len(budgets)} ({[f'{m}@{b}' for b in budgets for m in methods]})")
    print(f"  Search Depth   : L_c = {args.max_depth} | Min Candidates C = {args.num_candidates}")
    print(f"  Batch Size     : {args.batch_size} | Resume: {args.resume}")
    print(f"  Output Dir     : {args.output_dir.resolve()}")
    print("=" * 78, flush=True)

    ctx = context(
        stage=args.stage,
        methods=methods,
        budgets=budgets,
        max_depth=args.max_depth,
        num_candidates=args.num_candidates,
        num_samples=args.num_samples,
        sample_slice=sample_slice,
    )

    print(f"\nPrepared {len(ctx['examples'])} questions for evaluation.")
    print(f"Loading model on {args.device} (local_files_only={args.local_files_only})...", flush=True)
    wrapper = load_model(
        ctx,
        model_path=args.model_path,
        device=args.device,
        local_files_only=args.local_files_only,
    )

    started = perf_counter()
    q_rows, agg_rows = run_e2_stage(
        wrapper,
        ctx,
        args.output_dir,
        methods=methods,
        budgets=budgets,
        max_depth=args.max_depth,
        num_candidates=args.num_candidates,
        resume=args.resume,
        batch_size=args.batch_size,
    )
    total_time = perf_counter() - started

    summaries_dir = args.output_dir / "summaries"
    paired_rows = compute_paired_comparison(q_rows, budgets=budgets)
    if paired_rows:
        write_csv(summaries_dir / f"e2_{args.stage}_paired_comparison.csv", paired_rows)
        write_csv(summaries_dir / "e2_paired_comparison.csv", paired_rows)

    print(f"\nE2 Run Completed in {total_time:.2f}s ({len(q_rows)} configuration outcomes):")
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
        print("\nLaunching Offline Diagnostic Analysis...", flush=True)
        from dlm_project.tasks.gsm8k.e3_analysis import analyze_e3_stage
        analyze_e3_stage(
            results_dir=args.output_dir / args.stage,
            output_dir=summaries_dir,
            stage=args.stage,
            methods=methods,
            budgets=budgets,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

