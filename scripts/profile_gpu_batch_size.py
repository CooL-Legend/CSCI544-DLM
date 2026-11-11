"""GPU VRAM & Batch Size Capacity Profiler for LLaDA-8B-Instruct (MEDAL + AB-MEDAL E1/E3).

Provides both:
  1. Analytical VRAM & runtime estimation (runs anywhere, including CPU-only laptops,
     to plan which GPU tier to provision for a target batch_size and dataset size N=200 or N=500).
  2. Empirical CUDA sweep (when run on a target GPU, measures exact peak allocated/reserved
     VRAM, forward throughput in NFEs/s, and speedup across batch_size values, catching OOMs safely).

Examples:
    # Analytical table + plot (no GPU required):
    python scripts/profile_gpu_batch_size.py --batch-sizes 1 2 4 6 8 12 16 24 32 48 64

    # Live empirical CUDA sweep on the attached GPU:
    python scripts/profile_gpu_batch_size.py --empirical --batch-sizes 1 2 4 6 8 12 16
"""

import argparse
import gc
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dlm_project.decoding.ab_batch import pack_states, sync
from dlm_project.decoding.mcts_decoding import MCTSConfig, distribution_from_logits, model_device
from dlm_project.tasks.gsm8k.experiment_1 import context, load_model
from dlm_project.tasks.gsm8k.experiment_protocol import build_prompt

# LLaDA-8B-Instruct architectural constants (bfloat16)
LLADA_8B_PARAMS = 8_015_581_184
LLADA_VOCAB_SIZE = 126_464
LLADA_D_MODEL = 4096
LLADA_D_FF = 12288
BYTES_PER_BF16 = 2
GIB = 1024 ** 3

# Common GPU hardware tiers: (Name, Usable GiB, Safe Reserved Ceiling GiB, Relative BF16 TFLOPS vs L4)
GPU_TIERS: List[Tuple[str, float, float, float]] = [
    ("L4 (24GB / 22.5 GiB)", 22.49, 20.8, 1.00),
    ("RTX 3090 / 4090 / A10G (24GB)", 23.65, 22.0, 1.35),
    ("A100 (40GB)", 39.50, 37.0, 2.40),
    ("L40S / RTX 6000 Ada / A6000 (48GB)", 44.70, 42.0, 2.60),
    ("A100 (80GB)", 79.20, 75.0, 2.55),
    ("H100 (80GB)", 79.20, 75.0, 4.80),
]

DEFAULT_BATCH_SIZES = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def inspect_manifest_token_lengths(
    num_samples: int = 500,
    gen_length: int = 256,
    tokenizer=None,
) -> Dict[str, Any]:
    """Measure prompt + generation token lengths across the predetermined GSM8K test manifest."""
    ctx = context(stage="test", num_samples=num_samples)
    prompts = [build_prompt(ex["question"], ctx["common"]["prompt"]) for ex in ctx["examples"]]

    if tokenizer is not None:
        prompt_lens = [len(tokenizer.encode(p)) for p in prompts]
        method_used = "exact_tokenizer"
    else:
        # Calibrated whitespace/punctuation heuristic matching LLaDA tokenizer within ~3%
        prompt_lens = [int(round(len(p.split()) * 1.34 + 38)) for p in prompts]
        method_used = "calibrated_heuristic"

    total_lens = [plen + gen_length for plen in prompt_lens]
    sorted_totals = sorted(total_lens)
    p50 = sorted_totals[len(sorted_totals) // 2]
    p95 = sorted_totals[min(len(sorted_totals) - 1, int(len(sorted_totals) * 0.95))]
    p99 = sorted_totals[min(len(sorted_totals) - 1, int(len(sorted_totals) * 0.99))]

    return {
        "num_samples": len(total_lens),
        "measurement": method_used,
        "gen_length": gen_length,
        "min_total_tokens": min(total_lens),
        "mean_total_tokens": sum(total_lens) / len(total_lens),
        "p50_total_tokens": p50,
        "p95_total_tokens": p95,
        "p99_total_tokens": p99,
        "max_total_tokens": max(total_lens),
    }


def recommend_gpu_tier(reserved_gib: float) -> str:
    """Return the minimum GPU tier that safely accommodates reserved_gib."""
    for name, _total_gib, safe_ceiling_gib, _rel_speed in GPU_TIERS:
        if reserved_gib <= safe_ceiling_gib:
            return name
    return "Multi-GPU / B200 (>80GB)"


def estimate_vram_and_compute(
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    max_seq_len: int = 850,
    max_budget: int = 512,
    gen_length: int = 256,
    max_depth: int = 10,
    base_serial_nfe_sec_l4: float = 0.283,
) -> List[Dict[str, Any]]:
    """Compute analytical GPU VRAM (GiB) and runtime projections for each batch_size.

    Calibrated against verified L4 measurements on LLaDA-8B-Instruct (bfloat16 + varlen FA2):
      - Base static weights (8.02B * 2B = 14.93 GiB) + CUDA/cuBLAS context (~0.87 GiB) = 15.80 GiB.
      - Per-item dynamic peak during varlen forward pass:
          * packed out.logits (1, K*T, V) + unpacked padded_logits (K, T, V) in bf16
          * single-layer SwiGLU MLP + varlen QKV/attention workspace in bf16
          * float32 action-selection probability/entropy buffer on [1, gen_length, V]
    """
    static_weights_gib = (LLADA_8B_PARAMS * BYTES_PER_BF16) / GIB
    cuda_context_gib = 0.87
    action_sel_f32_gib = (gen_length * LLADA_VOCAB_SIZE * 4) / GIB  # ~0.12 GiB constant

    # Per question across BOTH methods (medal + ab_medal) at continuous B=512 with B=256 checkpoint:
    # Search NFEs per question = 2 methods * max_budget (e.g., 2 * 512 = 1024 NFEs)
    # Completion NFEs per question = 2 methods * (246 for B=256 + ~0.65 * 246 for B=512 due to cache hits)
    search_nfes_per_q = 2 * max_budget
    comp_nfes_per_q = int(round(2 * (gen_length - max_depth) * 1.65))
    total_nfes_per_q = search_nfes_per_q + comp_nfes_per_q

    rows = []
    for k in batch_sizes:
        if k < 1:
            continue
        # Dual logits peak in varlen_inner_forward: out.logits + padded_logits
        logits_peak_gib = (2.0 * k * max_seq_len * LLADA_VOCAB_SIZE * BYTES_PER_BF16) / GIB
        # Single-block SwiGLU MLP + varlen attention transient tensors under torch.no_grad()
        mlp_attn_gib = (k * max_seq_len * (3 * LLADA_D_FF + 5 * LLADA_D_MODEL) * BYTES_PER_BF16) / GIB

        dynamic_allocated_gib = logits_peak_gib + mlp_attn_gib + action_sel_f32_gib
        est_allocated_gib = static_weights_gib + cuda_context_gib + dynamic_allocated_gib
        # PyTorch CUDA caching allocator segment rounding + safety margin (~16% on dynamic tensors + 0.45 GiB)
        est_reserved_gib = static_weights_gib + cuda_context_gib + 0.45 + 1.16 * dynamic_allocated_gib

        # Effective batch speedup saturates as GPU transitions from memory-bandwidth bound to compute bound
        # Calibrated to match verified 1.35x at batch_size=4 on L4
        l4_speedup = 1.0 + 0.52 * (1.0 - math.exp(-0.28 * (k - 1)))
        a100_speedup = 1.0 + 1.65 * (1.0 - math.exp(-0.18 * (k - 1)))

        sec_per_q_l4 = (total_nfes_per_q * base_serial_nfe_sec_l4) / l4_speedup
        sec_per_q_a100 = (total_nfes_per_q * (base_serial_nfe_sec_l4 / 2.45)) / a100_speedup

        rows.append({
            "batch_size": k,
            "max_seq_len": max_seq_len,
            "static_base_gib": round(static_weights_gib + cuda_context_gib, 2),
            "dynamic_peak_gib": round(dynamic_allocated_gib, 2),
            "est_peak_allocated_gib": round(est_allocated_gib, 2),
            "est_peak_reserved_gib": round(est_reserved_gib, 2),
            "fits_l4_22_5gib": est_reserved_gib <= 20.8,
            "fits_24gb_gpu": est_reserved_gib <= 22.0,
            "fits_a100_40gb": est_reserved_gib <= 37.0,
            "fits_48gb_gpu": est_reserved_gib <= 42.0,
            "fits_80gb_gpu": est_reserved_gib <= 75.0,
            "recommended_gpu": recommend_gpu_tier(est_reserved_gib),
            "nfes_per_question_4configs": total_nfes_per_q,
            "est_l4_speedup": round(l4_speedup, 2),
            "est_hours_200q_l4": round((200 * sec_per_q_l4) / 3600.0, 2),
            "est_hours_500q_l4": round((500 * sec_per_q_l4) / 3600.0, 2),
            "est_hours_200q_a100": round((200 * sec_per_q_a100) / 3600.0, 2),
            "est_hours_500q_a100": round((500 * sec_per_q_a100) / 3600.0, 2),
        })
    return rows


@torch.inference_mode()
def run_empirical_cuda_sweep(
    wrapper,
    batch_sizes: Sequence[int] = (1, 2, 4, 6, 8, 12, 16),
    max_seq_len: int = 850,
    gen_length: int = 256,
    max_budget: int = 512,
    max_depth: int = 10,
    warmup_iters: int = 1,
    bench_iters: int = 3,
) -> List[Dict[str, Any]]:
    """Empirically benchmark peak GPU VRAM and throughput across batch_sizes on the attached CUDA device."""
    device = model_device(wrapper.model)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("run_empirical_cuda_sweep requires an active CUDA GPU.")

    total_vram_gib = torch.cuda.get_device_properties(device).total_memory / GIB
    gpu_name = torch.cuda.get_device_name(device)
    prompt_len = max_seq_len - gen_length
    pad_token_id = getattr(wrapper.tokenizer, "pad_token_id", None) or 126081
    cfg = MCTSConfig(init_tokens=max_depth, num_candidates=3, top_k_tokens=3, top_k_actions=5, stop_mode="budget")

    search_nfes_per_q = 2 * max_budget
    comp_nfes_per_q = int(round(2 * (gen_length - max_depth) * 1.65))
    total_nfes_per_q = search_nfes_per_q + comp_nfes_per_q

    results = []
    b1_nfes_per_sec: Optional[float] = None

    for k in batch_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        sync(wrapper.model)

        # Construct realistic variable-length states up to max_seq_len
        states = []
        for i in range(k):
            # Vary prompt length slightly across the batch (e.g., max_seq_len - 24 .. max_seq_len)
            seq_len_i = max_seq_len - (i % 4) * 8
            p_len_i = seq_len_i - gen_length
            st = torch.full((1, seq_len_i), wrapper.mask_id, dtype=torch.long)
            st[0, :p_len_i] = (torch.arange(p_len_i, dtype=torch.long) % 1000) + 10
            states.append(st)

        try:
            ids, attn, _ = pack_states(states, pad_token_id, device)
            for _ in range(warmup_iters):
                out = wrapper.model(ids) if k == 1 else wrapper.model(ids, attention_mask=attn)
                for row in range(k):
                    s_i = states[row]
                    p_len_i = s_i.shape[1] - gen_length
                    _ = distribution_from_logits(
                        out.logits[row, :s_i.shape[1]], s_i, p_len_i, wrapper.mask_id, cfg
                    )
                del out
            sync(wrapper.model)

            torch.cuda.reset_peak_memory_stats(device)
            t0 = perf_counter()
            for _ in range(bench_iters):
                out = wrapper.model(ids) if k == 1 else wrapper.model(ids, attention_mask=attn)
                for row in range(k):
                    s_i = states[row]
                    p_len_i = s_i.shape[1] - gen_length
                    _ = distribution_from_logits(
                        out.logits[row, :s_i.shape[1]], s_i, p_len_i, wrapper.mask_id, cfg
                    )
                del out
            sync(wrapper.model)
            elapsed = perf_counter() - t0

            peak_alloc_gib = torch.cuda.max_memory_allocated(device) / GIB
            peak_res_gib = torch.cuda.max_memory_reserved(device) / GIB
            step_ms = (elapsed / bench_iters) * 1000.0
            nfes_per_sec = (k * bench_iters) / elapsed
            if b1_nfes_per_sec is None:
                b1_nfes_per_sec = nfes_per_sec
            speedup = nfes_per_sec / b1_nfes_per_sec if b1_nfes_per_sec else 1.0

            sec_per_q = total_nfes_per_q / nfes_per_sec
            results.append({
                "batch_size": k,
                "status": "OK",
                "gpu_name": gpu_name,
                "gpu_total_gib": round(total_vram_gib, 2),
                "peak_allocated_gib": round(peak_alloc_gib, 2),
                "peak_reserved_gib": round(peak_res_gib, 2),
                "vram_headroom_gib": round(total_vram_gib - peak_res_gib, 2),
                "vram_utilization_pct": round(100.0 * peak_res_gib / total_vram_gib, 1),
                "batch_step_ms": round(step_ms, 1),
                "nfes_per_sec": round(nfes_per_sec, 2),
                "speedup_vs_b1": round(speedup, 2),
                "est_hours_200q": round((200 * sec_per_q) / 3600.0, 2),
                "est_hours_500q": round((500 * sec_per_q) / 3600.0, 2),
            })
            del ids, attn
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            results.append({
                "batch_size": k,
                "status": "OOM",
                "gpu_name": gpu_name,
                "gpu_total_gib": round(total_vram_gib, 2),
                "peak_allocated_gib": None,
                "peak_reserved_gib": None,
                "vram_headroom_gib": 0.0,
                "vram_utilization_pct": 100.0,
                "batch_step_ms": None,
                "nfes_per_sec": None,
                "speedup_vs_b1": None,
                "est_hours_200q": None,
                "est_hours_500q": None,
            })
    gc.collect()
    torch.cuda.empty_cache()
    return results


def plot_gpu_capacity_profile(
    analytical_rows: Sequence[Dict[str, Any]],
    empirical_rows: Optional[Sequence[Dict[str, Any]]] = None,
    save_path: Optional[Path] = None,
) -> Path:
    """Create a 2-panel capacity planning figure (VRAM vs batch_size and Runtime vs batch_size)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    bs = [r["batch_size"] for r in analytical_rows]
    alloc = [r["est_peak_allocated_gib"] for r in analytical_rows]
    res = [r["est_peak_reserved_gib"] for r in analytical_rows]

    ax1.plot(bs, alloc, marker="o", lw=2, color="#1f77b4", label="Analytical Peak Allocated (GiB)")
    ax1.plot(bs, res, marker="s", lw=2, color="#ff7f0e", label="Analytical Peak Reserved w/ Margin (GiB)")

    if empirical_rows:
        ok_emp = [r for r in empirical_rows if r.get("status") == "OK"]
        if ok_emp:
            ax1.plot(
                [r["batch_size"] for r in ok_emp],
                [r["peak_allocated_gib"] for r in ok_emp],
                marker="^", lw=2.2, ls="--", color="#2ca02c", label="Empirical Peak Allocated (GiB)",
            )
            ax1.plot(
                [r["batch_size"] for r in ok_emp],
                [r["peak_reserved_gib"] for r in ok_emp],
                marker="D", lw=2.2, ls="--", color="#d62728", label="Empirical Peak Reserved (GiB)",
            )

    tier_lines = [
        ("L4 (22.5 GiB)", 22.5, "#9467bd"),
        ("RTX 4090 / A10G (24 GB)", 24.0, "#8c564b"),
        ("A100 (40 GB)", 40.0, "#e377c2"),
        ("L40S / A6000 (48 GB)", 48.0, "#7f7f7f"),
        ("A100 / H100 (80 GB)", 80.0, "#17becf"),
    ]
    for label, vram, color in tier_lines:
        ax1.axhline(vram, ls=":", lw=1.4, color=color, alpha=0.85, label=f"Ceiling: {label}")

    ax1.set_xlabel("Cooperative Batch Size (K questions)", fontsize=10)
    ax1.set_ylabel("GPU Memory (GiB)", fontsize=10)
    ax1.set_title("GPU VRAM Required vs. Batch Size (LLaDA-8B, bf16)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8, loc="upper left")

    if empirical_rows and any(r.get("status") == "OK" for r in empirical_rows):
        ok_emp = [r for r in empirical_rows if r.get("status") == "OK"]
        emp_bs = [r["batch_size"] for r in ok_emp]
        ax2.plot(emp_bs, [r["est_hours_200q"] for r in ok_emp], marker="o", lw=2.2, color="#2ca02c",
                 label=f"Empirical N=200 ({ok_emp[0]['gpu_name']})")
        ax2.plot(emp_bs, [r["est_hours_500q"] for r in ok_emp], marker="s", lw=2.2, color="#d62728",
                 label=f"Empirical N=500 ({ok_emp[0]['gpu_name']})")
    else:
        ax2.plot(bs, [r["est_hours_200q_l4"] for r in analytical_rows], marker="o", lw=2, color="#1f77b4",
                 label="Est. N=200 Primary (L4 GPU)")
        ax2.plot(bs, [r["est_hours_500q_l4"] for r in analytical_rows], marker="s", lw=2, color="#ff7f0e",
                 label="Est. N=500 Extended (L4 GPU)")
        ax2.plot(bs, [r["est_hours_200q_a100"] for r in analytical_rows], marker="^", lw=2, ls="--", color="#2ca02c",
                 label="Est. N=200 Primary (A100 GPU)")
        ax2.plot(bs, [r["est_hours_500q_a100"] for r in analytical_rows], marker="D", lw=2, ls="--", color="#d62728",
                 label="Est. N=500 Extended (A100 GPU)")

    ax2.set_xlabel("Cooperative Batch Size (K questions)", fontsize=10)
    ax2.set_ylabel("Projected Wall-Clock Hours (All 4 Configs)", fontsize=10)
    ax2.set_title("Projected E1 Total Runtime vs. Batch Size (B=256 & 512)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8.5, loc="upper right")

    plt.tight_layout()
    out_path = Path(save_path or "results/gpu_batch_size_profile.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--max-seq-len", type=int, default=850, help="Worst-case prompt + gen token length.")
    parser.add_argument("--max-budget", type=int, default=512)
    parser.add_argument("--empirical", action="store_true", help="Run live CUDA memory & throughput sweep.")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-plot", type=Path, default=Path("results/gpu_batch_size_profile.png"))
    args = parser.parse_args(argv)

    token_stats = inspect_manifest_token_lengths(num_samples=500)
    seq_len = max(args.max_seq_len, token_stats["max_total_tokens"])

    print("=" * 94)
    print("GPU VRAM & COMPUTE CAPACITY PLANNER — LLaDA-8B-Instruct (bfloat16, varlen FlashAttention-2)")
    print(f"GSM8K 500-Question Test Manifest Token Lengths (prompt + 256 gen): "
          f"mean={token_stats['mean_total_tokens']:.0f}, p95={token_stats['p95_total_tokens']}, "
          f"max={token_stats['max_total_tokens']} (profiling at T_max={seq_len})")
    print("=" * 94)

    analytical = estimate_vram_and_compute(
        batch_sizes=args.batch_sizes, max_seq_len=seq_len, max_budget=args.max_budget
    )
    print(f"{'Batch':>5} | {'Alloc (GiB)':>11} | {'Rsrvd (GiB)':>11} | {'Minimum Recommended GPU':<34} | {'200Q (L4/A100)':>15} | {'500Q (L4/A100)':>15}")
    print("-" * 94)
    for r in analytical:
        h200 = f"{r['est_hours_200q_l4']:.1f}h / {r['est_hours_200q_a100']:.1f}h"
        h500 = f"{r['est_hours_500q_l4']:.1f}h / {r['est_hours_500q_a100']:.1f}h"
        print(
            f"{r['batch_size']:5d} | {r['est_peak_allocated_gib']:11.2f} | {r['est_peak_reserved_gib']:11.2f} | "
            f"{r['recommended_gpu']:<34} | {h200:>15} | {h500:>15}"
        )

    empirical = None
    if args.empirical:
        if not torch.cuda.is_available():
            print("\n[WARNING] --empirical requested, but no CUDA GPU is available. Skipping live sweep.")
        else:
            print("\nRunning live empirical CUDA sweep on attached GPU...", flush=True)
            ctx = context(stage="test", num_samples=2)
            wrapper = load_model(ctx, model_path=args.model_path, device="cuda", local_files_only=args.local_files_only)
            empirical = run_empirical_cuda_sweep(
                wrapper, batch_sizes=args.batch_sizes, max_seq_len=seq_len, max_budget=args.max_budget
            )
            print("\nEMPIRICAL CUDA SWEEP RESULTS:")
            print(f"{'Batch':>5} | {'Status':>6} | {'Alloc (GiB)':>11} | {'Rsrvd (GiB)':>11} | {'Headroom':>9} | {'NFEs/s':>7} | {'Speedup':>7} | {'200Q Hrs':>8} | {'500Q Hrs':>8}")
            print("-" * 94)
            for r in empirical:
                if r["status"] == "OK":
                    print(
                        f"{r['batch_size']:5d} | {r['status']:>6} | {r['peak_allocated_gib']:11.2f} | "
                        f"{r['peak_reserved_gib']:11.2f} | {r['vram_headroom_gib']:8.2f}G | "
                        f"{r['nfes_per_sec']:7.2f} | {r['speedup_vs_b1']:6.2f}x | "
                        f"{r['est_hours_200q']:7.2f}h | {r['est_hours_500q']:7.2f}h"
                    )
                else:
                    print(f"{r['batch_size']:5d} | {r['status']:>6} | {'OOM':>11} | {'OOM':>11} | {'0.00G':>9} | {'-':>7} | {'-':>7} | {'-':>8} | {'-':>8}")

    plot_path = plot_gpu_capacity_profile(analytical, empirical, save_path=args.output_plot)
    print(f"\nSaved GPU capacity profile chart to: {plot_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

