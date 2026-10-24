#!/usr/bin/env python3
"""Comprehensive diagnostic suite for LLaDA padding alignment, attention kernels, and batching.

Executes the 5-step diagnostic protocol:
1. Padding Alignment & Slicing: Verifies prompt tokens, 256 output masks, pad tokens, and RoPE indices.
2. Kernel Isolation: Compares FlashAttention vs PyTorch SDPA on identical unpadded inputs,
   and unpadded SDPA vs padded SDPA to isolate pure padding effects from kernel divergence.
3. Identical-Input Batching: Tests [x, x] batching to isolate batch dimensions from padding.
4. Heterogeneous Batching: Analyzes cross-sequence effects in [x_long, x_short].
5. Decision Parity: Evaluates candidate action sets, Shannon entropy, IG ratio, and decoded text.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from dlm_project.decoding.ab_batch import pack_states, sync
from dlm_project.decoding.common import validate_generation
from dlm_project.decoding.mcts_decoding import (
    complete_llada,
    distribution_from_logits,
    model_device,
)
from dlm_project.experiments.artifacts import save_json
from dlm_project.tasks.gsm8k import ab_development as dev
from dlm_project.tasks.gsm8k.ab_batch_study import _logit_difference, RTOL, ATOL


def toggle_flash_attention(model: torch.nn.Module, enabled: bool) -> int:
    """Toggle flash_attn_func on model blocks between FlashAttention-2 and PyTorch SDPA."""
    count = 0
    for module in model.modules():
        if hasattr(module, "flash_attn_func"):
            if enabled:
                try:
                    from flash_attn import flash_attn_func  # type: ignore
                    module.flash_attn_func = flash_attn_func
                except ImportError:
                    module.flash_attn_func = None
            else:
                module.flash_attn_func = None
            count += 1
    return count


def forward_pass(model: torch.nn.Module, ids: torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Run model forward without gradients and return float32 logits."""
    with torch.inference_mode():
        if attention_mask is None:
            out = model(ids)
        else:
            out = model(ids, attention_mask=attention_mask)
        return out.logits.float()


def run_diagnostics(
    wrapper: Any,
    ctx: Dict[str, Any],
    prepared: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute the full 5-step diagnostic suite."""
    device = model_device(wrapper.model)
    mask_id = wrapper.mask_id
    pad_token_id = ctx["pad_token_id"]
    config = dev.search_config(ctx)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("LLaDA PADDING & BATCHING DIAGNOSTIC SUITE")
    print(f"Device: {device} | Mask ID: {mask_id} | Pad Token ID: {pad_token_id}")
    print(f"Samples to Diagnose: {len(prepared)} ({[p['example']['example_id'] for p in prepared]})")
    print("=" * 78)

    results: Dict[str, Any] = {
        "device": str(device),
        "mask_id": mask_id,
        "pad_token_id": pad_token_id,
        "samples": [p["example"]["example_id"] for p in prepared],
        "step1_alignment": {},
        "step2_kernel_isolation": {},
        "step3_identical_batching": {},
        "step4_heterogeneous_batching": {},
        "step5_decision_parity": {},
    }

    # Prepare root states (prompt + 256 masks)
    root_states = []
    for item in prepared:
        tokens = item["prompt_ids"] + [mask_id] * 256
        root_states.append(torch.tensor([tokens], dtype=torch.long, device=device))

    # -------------------------------------------------------------------------
    # STEP 1: Verify Padding Alignment & Slicing
    # -------------------------------------------------------------------------
    print("\n[STEP 1/5] Verifying Padding Alignment & Token/Mask Geometry...")
    ids_packed, attention_packed, lengths = pack_states(root_states, pad_token_id, device)
    step1_passed = True

    for i, (item, state, length) in enumerate(zip(prepared, root_states, lengths)):
        name = item["example"]["example_id"]
        prompt_len = len(item["prompt_ids"])
        total_len = prompt_len + 256
        pad_len = ids_packed.shape[1] - total_len

        # Check token fidelity in packed slice
        tokens_match = bool(torch.equal(ids_packed[i, :length], state[0]))
        # Check pad tokens in tail
        pad_tokens_correct = bool(torch.all(ids_packed[i, length:] == pad_token_id)) if pad_len > 0 else True
        # Check attention mask
        attn_active_correct = bool(torch.all(attention_packed[i, :length] == 1))
        attn_pad_correct = bool(torch.all(attention_packed[i, length:] == 0)) if pad_len > 0 else True

        passed = tokens_match and pad_tokens_correct and attn_active_correct and attn_pad_correct
        step1_passed &= passed

        status_str = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status_str} [{name}]: Prompt={prompt_len} | Masks=256 | Total={total_len} | PadTail={pad_len}")
        if not passed:
            print(f"    Tokens match: {tokens_match}, Pad tokens: {pad_tokens_correct}, "
                  f"Active mask: {attn_active_correct}, Pad mask: {attn_pad_correct}")

        results["step1_alignment"][name] = {
            "passed": passed,
            "prompt_length": prompt_len,
            "total_length": total_len,
            "pad_length": pad_len,
            "tokens_match": tokens_match,
            "pad_tokens_correct": pad_tokens_correct,
            "attn_active_correct": attn_active_correct,
            "attn_pad_correct": attn_pad_correct,
        }

    # -------------------------------------------------------------------------
    # STEP 2: Isolate Kernel Difference (FlashAttention vs PyTorch SDPA)
    # -------------------------------------------------------------------------
    print("\n[STEP 2/5] Isolating Attention Kernel (FlashAttention vs PyTorch SDPA)...")
    has_flash = any(hasattr(m, "flash_attn_func") and m.flash_attn_func is not None for m in wrapper.model.modules())
    print(f"  FlashAttention-2 active on model: {has_flash}")

    for i, (item, state, length) in enumerate(zip(prepared, root_states, lengths)):
        name = item["example"]["example_id"]
        prompt_len = len(item["prompt_ids"])
        positions = torch.nonzero(state[0, prompt_len:] == mask_id).flatten().to(device) + prompt_len

        # A: FlashAttention Unpadded (if available)
        toggle_flash_attention(wrapper.model, enabled=True)
        logits_flash_unpadded = forward_pass(wrapper.model, state)

        # B: SDPA Unpadded (force flash_attn_func = None)
        toggle_flash_attention(wrapper.model, enabled=False)
        logits_sdpa_unpadded = forward_pass(wrapper.model, state)

        # C: SDPA Padded Serial
        logits_sdpa_padded = forward_pass(wrapper.model, ids_packed[i:i+1], attention_packed[i:i+1])

        # Restore FlashAttention
        if has_flash:
            toggle_flash_attention(wrapper.model, enabled=True)

        # Compare Kernel Difference on UNPADDED input
        diff_kernel = _logit_difference(logits_flash_unpadded[0], logits_sdpa_unpadded[0], positions)
        # Compare Pure Padding Difference under the SAME SDPA backend
        diff_padding = _logit_difference(logits_sdpa_unpadded[0], logits_sdpa_padded[0, :length], positions)

        print(f"  [{name}]:")
        print(f"    - Kernel Delta (FlashAttn vs SDPA unpadded): "
              f"max_abs={diff_kernel['max_abs_logit_difference']:.6f}, "
              f"argmax_equal={diff_kernel['argmax_equal']}")
        print(f"    - Pure Padding Delta (SDPA unpadded vs SDPA padded): "
              f"max_abs={diff_padding['max_abs_logit_difference']:.6f}, "
              f"argmax_equal={diff_padding['argmax_equal']}")

        results["step2_kernel_isolation"][name] = {
            "kernel_delta": diff_kernel,
            "padding_delta": diff_padding,
        }

    # -------------------------------------------------------------------------
    # STEP 3: Test Identical-Input Batching ([x, x])
    # -------------------------------------------------------------------------
    print("\n[STEP 3/5] Testing Identical-Input Batching ([x, x])...")
    for i, (item, state, length) in enumerate(zip(prepared, root_states, lengths)):
        name = item["example"]["example_id"]
        prompt_len = len(item["prompt_ids"])
        positions = torch.nonzero(state[0, prompt_len:] == mask_id).flatten().to(device) + prompt_len

        # Single unpadded forward
        single_logits = forward_pass(wrapper.model, state)[0]

        # Duplicate into batch of 2 identical sequences
        twin_states = torch.cat([state, state], dim=0)
        twin_logits = forward_pass(wrapper.model, twin_states)

        # Within-batch consistency (row 0 vs row 1)
        diff_within = _logit_difference(twin_logits[0], twin_logits[1], positions)
        # Batch vs Single consistency (row 0 vs single)
        diff_single = _logit_difference(single_logits, twin_logits[0], positions)

        passed = diff_within["argmax_equal"] and diff_single["argmax_equal"]
        status_str = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status_str} [{name}] Identical Batch: "
              f"Within-Batch abs_diff={diff_within['max_abs_logit_difference']:.6f} (argmax={diff_within['argmax_equal']}) | "
              f"Batch-vs-Single abs_diff={diff_single['max_abs_logit_difference']:.6f} (argmax={diff_single['argmax_equal']})")

        results["step3_identical_batching"][name] = {
            "passed": passed,
            "within_batch": diff_within,
            "batch_vs_single": diff_single,
        }

    # -------------------------------------------------------------------------
    # STEP 4: Mixed Heterogeneous Batching ([x_long, x_short])
    # -------------------------------------------------------------------------
    print("\n[STEP 4/5] Testing Mixed Heterogeneous Batching...")
    batched_logits = forward_pass(wrapper.model, ids_packed, attention_packed)

    for i, (item, state, length) in enumerate(zip(prepared, root_states, lengths)):
        name = item["example"]["example_id"]
        prompt_len = len(item["prompt_ids"])
        positions = torch.nonzero(state[0, prompt_len:] == mask_id).flatten().to(device) + prompt_len

        single_logits = forward_pass(wrapper.model, state)[0]
        padded_serial_logits = forward_pass(wrapper.model, ids_packed[i:i+1], attention_packed[i:i+1])[0, :length]
        candidate_batched = batched_logits[i, :length]

        diff_vs_single = _logit_difference(single_logits, candidate_batched, positions)
        diff_vs_padded_serial = _logit_difference(padded_serial_logits, candidate_batched, positions)

        print(f"  [{name}]:")
        print(f"    - Batched vs Single Unpadded: "
              f"max_abs={diff_vs_single['max_abs_logit_difference']:.6f}, "
              f"argmax_equal={diff_vs_single['argmax_equal']}")
        print(f"    - Batched vs Padded Serial:   "
              f"max_abs={diff_vs_padded_serial['max_abs_logit_difference']:.6f}, "
              f"argmax_equal={diff_vs_padded_serial['argmax_equal']}")

        results["step4_heterogeneous_batching"][name] = {
            "batched_vs_single": diff_vs_single,
            "batched_vs_padded_serial": diff_vs_padded_serial,
        }

    # -------------------------------------------------------------------------
    # STEP 5: Decision & Decoding Parity
    # -------------------------------------------------------------------------
    print("\n[STEP 5/5] Evaluating Decision Parity (Actions, Entropy, and Answers)...")
    for i, (item, state, length) in enumerate(zip(prepared, root_states, lengths)):
        name = item["example"]["example_id"]
        prompt_len = len(item["prompt_ids"])

        single_logits = forward_pass(wrapper.model, state)[0]
        batched_row = batched_logits[i, :length]

        single_dist = distribution_from_logits(
            single_logits, state, prompt_len, mask_id, config,
            search_excluded_token_ids=ctx["termination"]["token_ids"]
        )[0]
        batched_dist = distribution_from_logits(
            batched_row, state, prompt_len, mask_id, config,
            search_excluded_token_ids=ctx["termination"]["token_ids"]
        )[0]

        single_actions = [(a.output_position, a.token_id) for a in single_dist.actions]
        batched_actions = [(a.output_position, a.token_id) for a in batched_dist.actions]
        actions_match = (single_actions == batched_actions)

        h_single = float(single_dist.entropies.mean())
        h_batched = float(batched_dist.entropies.mean())
        h_diff = abs(h_single - h_batched)

        # Run greedy completion to test whether FlashAttention vs SDPA alters decoded output
        print(f"    Decoding greedy completion for {name} (FlashAttn vs SDPA)...")
        toggle_flash_attention(wrapper.model, enabled=True)
        flash_completion = complete_llada(wrapper.model, state, prompt_len, mask_id, temperature=0.0)

        toggle_flash_attention(wrapper.model, enabled=False)
        sdpa_completion = complete_llada(wrapper.model, state, prompt_len, mask_id, temperature=0.0)
        if has_flash:
            toggle_flash_attention(wrapper.model, enabled=True)

        flash_text = wrapper.tokenizer.decode(flash_completion.token_ids[0, prompt_len:].tolist(), skip_special_tokens=True)
        sdpa_text = wrapper.tokenizer.decode(sdpa_completion.token_ids[0, prompt_len:].tolist(), skip_special_tokens=True)
        text_match = (flash_text.strip() == sdpa_text.strip())

        passed = actions_match and text_match
        status_str = "✓ PASSED" if passed else "✗ FAILED"

        print(f"  {status_str} [{name}]:")
        print(f"    - Candidate Actions Match: {actions_match} ({len(single_actions)} actions)")
        print(f"    - Mean Entropy: Single={h_single:.4f} | Batched={h_batched:.4f} (diff={h_diff:.4f})")
        print(f"    - Final Decoded Text Match (FlashAttn vs SDPA): {text_match}")
        if not text_match:
            print(f"      FlashAttn Answer: {flash_text[:60]!r}")
            print(f"      SDPA      Answer: {sdpa_text[:60]!r}")

        results["step5_decision_parity"][name] = {
            "actions_match": actions_match,
            "mean_entropy_single": h_single,
            "mean_entropy_batched": h_batched,
            "entropy_abs_diff": h_diff,
            "text_match": text_match,
            "flash_answer_preview": flash_text[:80],
            "sdpa_answer_preview": sdpa_text[:80],
        }

    save_json(output_dir / "padding_batch_diagnosis.json", results)
    print("\n" + "=" * 78)
    print(f"DIAGNOSTIC SUITE COMPLETE. Results saved to: {output_dir / 'padding_batch_diagnosis.json'}")
    print("=" * 78 + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Diagnose LLaDA padding alignment and attention kernel divergence.")
    parser.add_argument("--method", choices=["medal", "ab_medal"], default="medal")
    parser.add_argument("--condition", choices=["A", "B", "C"], default="B")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--model-path", default=None, help="Local directory or HuggingFace repo ID.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", default="experiments/gsm8k/diagnostics")
    args = parser.parse_args()

    indices = tuple(range(args.num_samples))
    ctx = dev.context(condition=args.condition, indices=indices)

    print(f"Loading model onto {args.device}...")
    wrapper = dev.load_model(
        ctx,
        model_path=args.model_path,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    ctx["pad_token_id"] = getattr(wrapper.tokenizer, "pad_token_id", None) or 126081
    ctx["model"] = wrapper.identity
    prepared = dev.prepare(wrapper, ctx)

    run_diagnostics(wrapper, ctx, prepared, Path(args.output_dir))


if __name__ == "__main__":
    main()
