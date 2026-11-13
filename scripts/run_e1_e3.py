"""CLI entry point for running unified Experiment 1 (and triggering E3 analysis).

Example:
    python scripts/run_e1_e3.py --stage dev --device cuda --run-e3
    python scripts/run_e1_e3.py --stage evaluation --device cuda --run-e3
"""

import argparse
from pathlib import Path
from time import perf_counter

import torch

from dlm_project.experiments.resources import finish_reporting

from dlm_project.tasks.gsm8k.experiment_1 import (
    CHECKPOINT_BUDGETS, MAX_SEARCH_DEPTH, MIN_CANDIDATES,
    context, load_model, run_e1_stage
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("dev", "evaluation", "test"), default="dev")
    parser.add_argument("--methods", nargs="+", choices=("medal", "ab_medal"), default=["medal", "ab_medal"])
    parser.add_argument("--budgets", nargs="+", type=int, default=list(CHECKPOINT_BUDGETS))
    parser.add_argument("--max-depth", type=int, default=MAX_SEARCH_DEPTH)
    parser.add_argument("--num-candidates", type=int, default=MIN_CANDIDATES)
    parser.add_argument("--num-samples", type=int, help="Optional prefix count for quick testing.")
    parser.add_argument("--batch-size", type=int, default=4, help="Cooperative batch size across questions (default: 4).")
    parser.add_argument("--model-path", help="Local checkpoint or HuggingFace repo ID.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/e1_e3"))
    parser.add_argument("--no-resume", action="store_false", dest="resume", default=True,
                        help="Disable checkpoint resumption and force re-running all questions from scratch.")
    parser.add_argument("--run-e3", action="store_true", help="Run E3 offline analysis on completion.")
    parser.add_argument("--profile-resources", action="store_true",
                        help="Measure actual batch memory, forwards and elapsed time in resource sidecars.")
    args = parser.parse_args(argv)
    entrypoint_started = perf_counter()

    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but unavailable.")

    print(f"Initializing E1 run: stage={args.stage}, methods={args.methods}, budgets={args.budgets}, depth={args.max_depth}, batch_size={args.batch_size}, resume={args.resume}")
    ctx = context(
        stage=args.stage,
        methods=tuple(args.methods),
        budgets=tuple(args.budgets),
        max_depth=args.max_depth,
        num_candidates=args.num_candidates,
        num_samples=args.num_samples,
    )

    print(f"Loading model on {args.device}...")
    wrapper = load_model(ctx, model_path=args.model_path, device=args.device, local_files_only=args.local_files_only)

    started = perf_counter()
    q_rows, agg_rows = run_e1_stage(
        wrapper, ctx, args.output_dir,
        methods=tuple(args.methods),
        budgets=tuple(args.budgets),
        max_depth=args.max_depth,
        num_candidates=args.num_candidates,
        resume=args.resume,
        batch_size=args.batch_size,
        profile_resources=args.profile_resources,
    )
    total_time = perf_counter() - started

    reporting_started = perf_counter()
    offline_seconds = 0.0
    print(f"\nE1 Stage '{args.stage}' Completed in {total_time:.2f}s:")
    print(f"  Total outcomes: {len(q_rows)}")
    print(f"  Artifacts saved to: {args.output_dir}")
    print("\nSummary Results:")
    for row in agg_rows:
        print(f"  [{row['method'].upper()}] Budget={row['search_budget']}: "
              f"Acc={row['accuracy_percent']:.1f}% [{row['ci_95_low']*100:.1f}%, {row['ci_95_high']*100:.1f}%], "
              f"Total NFE={row['mean_total_nfe']:.1f}, "
              f"Selected Depth={row['mean_selected_depth']:.1f}, "
              f"Time={row['mean_total_wall_seconds']:.2f}s")

    if args.run_e3:
        offline_started = perf_counter()
        print("\nLaunching E3 Offline Analysis...")
        from dlm_project.tasks.gsm8k.e3_analysis import analyze_e3_stage
        analyze_e3_stage(
            results_dir=args.output_dir / args.stage,
            output_dir=args.output_dir / "summaries",
            stage=args.stage,
            methods=tuple(args.methods),
            budgets=tuple(args.budgets),
        )

        offline_seconds = perf_counter() - offline_started

    if args.profile_resources:
        finish_reporting(args.output_dir,
                         reporting_seconds=perf_counter() - reporting_started - offline_seconds,
                         offline_seconds=offline_seconds,
                         total_seconds=perf_counter() - entrypoint_started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
