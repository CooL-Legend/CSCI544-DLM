"""CLI entry point for running standalone E3 offline analysis on saved E1 results.

Example:
    python scripts/run_e3_analysis.py --stage dev --results-dir results/e1_e3/dev --output-dir results/e1_e3/summaries
    python scripts/run_e3_analysis.py --stage evaluation --results-dir results/e1_e3/evaluation --output-dir results/e1_e3/summaries
"""

import argparse
from pathlib import Path

from dlm_project.tasks.gsm8k.e3_analysis import analyze_e3_stage
from dlm_project.tasks.gsm8k.experiment_1 import CHECKPOINT_BUDGETS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("dev", "evaluation", "test"), default="dev")
    parser.add_argument("--results-dir", type=Path, help="Directory containing method subdirs (defaults to results/e1_e3/<stage>).")
    parser.add_argument("--output-dir", type=Path, default=Path("results/e1_e3/summaries"))
    parser.add_argument("--methods", nargs="+", choices=("medal", "ab_medal"), default=["medal", "ab_medal"])
    parser.add_argument("--budgets", nargs="+", type=int, default=list(CHECKPOINT_BUDGETS))
    args = parser.parse_args(argv)

    results_dir = args.results_dir or Path("results/e1_e3") / args.stage

    if not results_dir.is_dir():
        parser.error(f"Results directory does not exist: {results_dir}")

    print(f"Running E3 offline analysis on: {results_dir}")
    print(f"Output directory: {args.output_dir}")

    analyze_e3_stage(
        results_dir=results_dir,
        output_dir=args.output_dir,
        stage=args.stage,
        methods=tuple(args.methods),
        budgets=tuple(args.budgets),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
