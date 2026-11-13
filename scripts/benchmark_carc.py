"""Run full E1 pilots on one allocated GPU; never extrapolate from E2 SRIG.

Each batch size runs in a fresh process with empty result/cache state. The
length stress mode is a separate forward microbenchmark, not a runtime forecast.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from dlm_project.experiments.artifacts import save_json
from dlm_project.paths import PROJECT_ROOT


def compare_runs(reference, candidate):
    """Strict comparisons intentionally include discrete trees and completion tokens."""
    differences = []
    paths = sorted((Path(reference) / "evaluation").glob("*/*/checkpoints/*/metrics.json"))
    if len(paths) != 80:
        raise ValueError(f"Expected 80 outcomes in the 20-question reference, found {len(paths)}.")
    for path in paths:
        other = Path(candidate) / path.relative_to(reference)
        for name, keys in (
            ("metrics", ("predicted_answer", "exact_match", "selected_state_hash", "search_nfes", "completion_nfes")),
            ("completion", ("token_ids",)), ("tree", ("nodes",)),
        ):
            left = json.loads(path.with_name(f"{name}.json").read_text())
            right = json.loads(other.with_name(f"{name}.json").read_text())
            for key in keys:
                if left[key] != right[key]:
                    differences.append({"checkpoint": str(path.parent.relative_to(reference)), "field": key})
    return {"strict_parity_passed": not differences, "differences": differences}


def choose_batch(results):
    """Prefer the validated working batch 4; require 10% gain to select 8."""
    def safe(size):
        result = results.get(str(size), {})
        summary = result.get("summary", {})
        return (summary.get("status") == "completed"
                and summary.get("fully_fresh_questions") == 20
                and summary.get("resumed_question_methods", 0) == 0
                and (summary.get("gpu_headroom_fraction") or 0) >= .20
                and result.get("stress_passed", False)
                and result.get("strict_parity_passed", False))
    if safe(4):
        if safe(8):
            t4 = results["4"]["summary"]["evaluation_wall_seconds"]
            t8 = results["8"]["summary"]["evaluation_wall_seconds"]
            if t8 > 0 and t4 / t8 >= 1.10:
                return 8
        return 4
    return 1 if safe(1) else None


def estimate_seconds(summary):
    return ((summary.get("model_load_seconds") or 0)
            + 1.25 * (500 / 20) * summary["evaluation_wall_seconds"]
            + 1.25 * (500 / 20) * (summary.get("reporting_wall_seconds", 0)
                                   + summary.get("offline_analysis_wall_seconds", 0)))


def stress(args):
    from dlm_project.tasks.gsm8k.experiment_1 import context, load_model, prepare
    from scripts.profile_gpu_batch_size import run_empirical_cuda_sweep

    ctx = context(stage="test", num_samples=500)
    wrapper = load_model(ctx, model_path=args.model_path, local_files_only=args.local_files_only)
    prepared = prepare(wrapper, ctx)
    lengths = [int(p["ids"].shape[1]) for p in prepared]
    maximum = max(lengths) + ctx["common"]["generation"]["gen_length"]
    # Both search width and worst-case two-budget completion width, without cache hits.
    widths = sorted({args.batch_size, 2 * args.batch_size})
    rows = run_empirical_cuda_sweep(wrapper, batch_sizes=widths, max_seq_len=maximum)
    output = {"scope": "longest_500_prompt_forward_stress_only", "max_sequence_tokens": maximum,
              "configured_question_batch_size": args.batch_size,
              "passed": all(r["status"] == "OK" and r["vram_utilization_pct"] <= 80 for r in rows),
              "rows": rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "length_stress.json", output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--stress-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4, choices=(1, 4, 8))
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    if args.stress_only:
        stress(args)
        return 0
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use an empty output directory so resumed work cannot enter the benchmark.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    model_flags = ["--model-path", args.model_path]
    if args.local_files_only:
        model_flags.append("--local-files-only")
    for size in (1, 4, 8):
        if size == 8 and not (
            results["4"].get("summary", {}).get("gpu_headroom_fraction", 0) is not None
            and (results["4"].get("summary", {}).get("gpu_headroom_fraction") or 0) >= .20
            and results["4"].get("stress_passed")
            and results["4"].get("strict_parity_passed")
        ):
            break
        directory = args.output_dir / f"batch_{size}"
        command = [sys.executable, "-u", "scripts/run_e1_e3.py", "--stage", "evaluation",
                   "--num-samples", "20", "--batch-size", str(size), "--budgets", "256", "512",
                   "--profile-resources", "--run-e3", "--no-resume", "--output-dir", str(directory), *model_flags]
        run = subprocess.run(command, cwd=PROJECT_ROOT)
        summary_path = directory / "resource_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"status": "failed"}
        results[str(size)] = {"summary": summary, "exit_code": run.returncode,
                              "strict_parity_passed": size == 1 and run.returncode == 0}
        if run.returncode:
            if size == 1:
                break
            continue
        if size != 1:
            results[str(size)].update(compare_runs(args.output_dir / "batch_1", directory))
        check = subprocess.run([sys.executable, "-u", "scripts/benchmark_carc.py", "--stress-only",
                                "--batch-size", str(size), "--output-dir", str(directory), *model_flags], cwd=PROJECT_ROOT)
        stress_path = directory / "length_stress.json"
        results[str(size)]["stress_passed"] = (
            check.returncode == 0 and stress_path.exists() and json.loads(stress_path.read_text())["passed"])
        save_json(args.output_dir / "benchmark_progress.json", results)
    chosen = choose_batch(results)
    report = {"results": results, "recommended_batch_size": chosen,
              "production_validation_passed": chosen is not None,
              "estimated_500_question_seconds": estimate_seconds(results[str(chosen)]["summary"]) if chosen else None,
              "timing_scope": "E1 20-question measured pilot; reporting allowance scaled with sample count"}
    save_json(args.output_dir / "benchmark_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    return 0 if chosen is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
