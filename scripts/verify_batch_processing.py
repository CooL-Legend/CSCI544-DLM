"""Unified verification script for MEDAL and AB-MEDAL cooperative batch processing.

Tests padding probe parity, full-run tree parity, discrete decisions, and speedups
across batch sizes (e.g. 1, 2, 4) taking reference from both base notebooks:
  - notebooks/gsm8k/MEDAL_MCTS_Demo.ipynb
  - notebooks/gsm8k/AB_MEDAL_MCTS_Colab.ipynb

Example usage:
    python scripts/verify_batch_processing.py --device cuda --methods medal ab_medal --batch-sizes 2 4
    python scripts/verify_batch_processing.py --fast --device cuda
"""

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence, Tuple

import torch

from dlm_project.decoding.ab_batch import sync
from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.paths import create_run_directory
from dlm_project.experiments.sweep_report import write_csv
from dlm_project.tasks.gsm8k import ab_development as dev
from dlm_project.tasks.gsm8k import ab_batch_study as study


def verify_method_batching(
    wrapper,
    method: str,
    prepared: List[Dict[str, Any]],
    base_ctx: Dict[str, Any],
    output_dir: Path,
    batch_sizes: Sequence[int] = (2,),
) -> Dict[str, Any]:
    """Run serial baseline, padding probes, and batched runs for a given method."""
    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)

    ctx = dev.context(
        stage=base_ctx["stage"],
        condition=base_ctx["condition"],
        indices=tuple(range(len(prepared))),
        method=method,
    )
    # Inherit termination tokens and pad token from base_ctx
    ctx["termination"] = base_ctx["termination"]
    ctx["pad_token_id"] = base_ctx["pad_token_id"]
    ctx["model"] = wrapper.identity

    print(f"\n{'='*75}")
    print(f"VERIFYING BATCH PROCESSING: {method.upper()}")
    print(f"Condition: {ctx['condition']} | Depth: {ctx['search']['init_tokens']} | Budget: {ctx['search']['max_model_calls']}")
    print(f"Questions: {len(prepared)} | Batch Sizes to Test: {list(batch_sizes)}")
    print(f"{'='*75}")

    # 1. Serial Baseline (batch_size = 1)
    print(f"\n[1/3] Running Serial Baseline (batch_size=1) for {method}...")
    t0 = perf_counter()
    serial_dir = method_dir / "serial"
    serial = dev.run_samples(wrapper, prepared, ctx, serial_dir, batch_size=1)
    serial_time = perf_counter() - t0
    serial_summary = serial["summary"]
    print(f"  ✓ Serial completed in {serial_time:.2f}s | Status: {serial_summary['status']}")
    print(f"  ✓ Search NFEs: {serial_summary['search_nfes']} | Completion NFEs: {serial_summary['completion_nfes']} | Invocations: {serial_summary['total_model_invocations']}")

    results: Dict[str, Any] = {
        "method": method,
        "serial": {
            "wall_seconds": serial_time,
            "decoding_seconds": serial_summary["decoding_seconds"],
            "search_nfes": serial_summary["search_nfes"],
            "completion_nfes": serial_summary["completion_nfes"],
            "total_nfes": serial_summary["total_nfes"],
            "invocations": serial_summary["total_model_invocations"],
            "accuracy": serial_summary.get("accuracy"),
            "status": serial_summary["status"],
        },
        "batches": {},
        "all_passed": True,
    }

    # 2. Test each requested batch size
    for b_size in batch_sizes:
        print(f"\n[2/3] Probing Padding Consistency (batch_size={b_size}) for {method}...")
        probe = study.padding_probe(wrapper, prepared, ctx, serial, b_size)
        probe_passed = bool(probe.get("passed", False))
        print(f"  Padding Probe Result: {'✓ PASSED' if probe_passed else '✗ FAILED'}")

        if not probe_passed:
            print(f"  WARNING: Padding probe failed for batch_size={b_size}. Error: {probe.get('error')}")
            results["batches"][str(b_size)] = {
                "batch_size": b_size,
                "padding_probe_passed": False,
                "full_run_passed": False,
                "error": probe.get("error") or "Padding probe failed",
            }
            results["all_passed"] = False
            continue

        print(f"\n[3/3] Running Full Batched Search & Completion (batch_size={b_size}) for {method}...")
        t_batch_start = perf_counter()
        batched_dir = method_dir / f"batch_{b_size}"
        batched = dev.run_samples(wrapper, prepared, ctx, batched_dir, batch_size=b_size)
        batch_wall_time = perf_counter() - t_batch_start
        batched_summary = batched["summary"]

        # Parity comparison with serial
        parity = study.compare_runs(serial, batched)
        parity_passed = bool(parity.get("passed", False))

        speedup = serial_time / batch_wall_time if batch_wall_time > 0 else 0.0
        nfe_ratio = serial_summary["total_model_invocations"] / batched_summary["total_model_invocations"] if batched_summary["total_model_invocations"] > 0 else 1.0

        print(f"  ✓ Batched execution finished in {batch_wall_time:.2f}s (Speedup: {speedup:.2f}x)")
        print(f"  ✓ Model Invocations reduced from {serial_summary['total_model_invocations']} down to {batched_summary['total_model_invocations']} ({nfe_ratio:.1f}x fewer forward calls)")
        print(f"  ✓ Parity Verdict: {'✓ PASSED (Bit-for-bit identical)' if parity_passed else '✗ FAILED'}")

        if not parity_passed:
            print(f"    Parity Differences ({len(parity.get('differences', []))}):")
            for diff in parity.get("differences", [])[:5]:
                print(f"      - {diff}")
            results["all_passed"] = False

        results["batches"][str(b_size)] = {
            "batch_size": b_size,
            "padding_probe_passed": probe_passed,
            "parity_passed": parity_passed,
            "wall_seconds": batch_wall_time,
            "speedup_over_serial": speedup,
            "invocation_reduction": nfe_ratio,
            "serial_invocations": serial_summary["total_model_invocations"],
            "batch_invocations": batched_summary["total_model_invocations"],
            "total_nfes": batched_summary["total_nfes"],
            "differences": parity.get("differences", []),
            "max_ig_difference": parity.get("max_ig_difference"),
            "max_q_difference": parity.get("max_q_difference"),
        }

    save_json(method_dir / "verification_results.json", results)
    return results


def format_summary_table(all_results: Dict[str, Any]) -> str:
    """Format verification outcomes as a Markdown table."""
    lines = [
        "| Method | Batch Size | Padding Parity | Full Tree Parity | Serial Wall Time | Batched Wall Time | Speedup | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for method, res in all_results.items():
        serial_time = res["serial"]["wall_seconds"]
        for b_str, b_data in res["batches"].items():
            b_size = b_data["batch_size"]
            pad_str = "PASS" if b_data.get("padding_probe_passed") else "FAIL"
            tree_str = "PASS" if b_data.get("parity_passed") else "FAIL"
            b_time = b_data.get("wall_seconds", 0.0)
            spd = b_data.get("speedup_over_serial", 0.0)
            status = "✓ VERIFIED" if b_data.get("parity_passed") else "✗ MISMATCH"
            lines.append(
                f"| **{method.upper()}** | {b_size} | {pad_str} | {tree_str} | "
                f"{serial_time:.2f}s | {b_time:.2f}s | **{spd:.2f}x** | {status} |"
            )
    return "\n".join(lines)


def run_verification(
    wrapper,
    methods: Sequence[str] = ("medal", "ab_medal"),
    batch_sizes: Sequence[int] = (2,),
    num_samples: int = 2,
    condition: str = "A",
    output_dir: Path = Path("results/batch_verification"),
) -> Tuple[Dict[str, Any], bool]:
    """Execute batch verification across both methods and compile full diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_ctx = dev.context(stage="dev", condition=condition, indices=tuple(range(num_samples)))
    prepared = dev.prepare(wrapper, base_ctx)

    print("Running initial warm-up...")
    warmup_res = dev.warmup(wrapper, prepared, base_ctx, batch_size=max(batch_sizes))
    print(f"Warm-up complete ({warmup_res['seconds']:.3f}s).")

    all_results = {}
    overall_success = True

    for method in methods:
        res = verify_method_batching(wrapper, method, prepared, base_ctx, output_dir, batch_sizes=batch_sizes)
        all_results[method] = res
        if not res["all_passed"]:
            overall_success = False

    summary_md = format_summary_table(all_results)
    print("\n" + "="*75)
    print("FINAL BATCH PROCESSING VERIFICATION SUMMARY")
    print("="*75)
    print(summary_md)
    print("="*75)

    if overall_success:
        print("\n✓ ALL BATCH PROCESSING VERIFICATIONS PASSED SUCCESSFULLY!")
        print("  Both MEDAL and AB-MEDAL cooperative batch processors produce 100% bit-for-bit")
        print("  equivalent search trees, decision traces, and completions to serial execution.")
    else:
        print("\n✗ SOME BATCH VERIFICATIONS FAILED. Check individual reports above.")

    save_json(output_dir / "summary.json", all_results)
    with (output_dir / "summary.md").open("w") as f:
        f.write("# Batch Processing Parity & Performance Summary\n\n" + summary_md + "\n")

    return all_results, overall_success


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", choices=("medal", "ab_medal"), default=["medal", "ab_medal"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[2])
    parser.add_argument("--num-samples", type=int, default=2, help="Number of frozen dev questions (default: 2)")
    parser.add_argument("--condition", choices=("A", "B", "C"), default="A", help="Condition A=(10,256), B=(10,512), C=(10,1024)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-path", help="Local model directory or HuggingFace repo ID.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/batch_verification"))
    parser.add_argument("--fast", action="store_true", help="Fast diagnostic mode with 2 samples and batch_size 2.")
    args = parser.parse_args(argv)

    if args.fast:
        args.num_samples = min(args.num_samples, 2)
        args.batch_sizes = [2]

    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    print("Initializing environment and loading model...")
    dummy_ctx = dev.context(stage="dev", condition=args.condition, indices=tuple(range(args.num_samples)))
    wrapper = dev.load_model(dummy_ctx, model_path=args.model_path, device=args.device, local_files_only=args.local_files_only)

    all_results, success = run_verification(
        wrapper,
        methods=args.methods,
        batch_sizes=args.batch_sizes,
        num_samples=args.num_samples,
        condition=args.condition,
        output_dir=args.output_dir,
    )

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
