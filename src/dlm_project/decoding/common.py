"""Shared fixed-length sampling for the four GSM8K decoding methods."""

import math

import torch

from dlm_project.decoding.baseline import add_gumbel_noise, get_num_transfer_tokens
from dlm_project.decoding.decoder_result import DecodeResult


def validate_generation(*, gen_length, steps, block_length, temperature, seed=1):
    for name, value in (("gen_length", gen_length), ("steps", steps),
                        ("block_length", block_length)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if gen_length % block_length or steps > gen_length or steps % (gen_length // block_length):
        raise ValueError("gen_length must divide into blocks; steps must be <= gen_length "
                         "and divisible by the number of blocks.")
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature < 0):
        raise ValueError("temperature must be finite and nonnegative.")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63).")


@torch.inference_mode()
def decode_masked(model, prompt_ids, mask_id, *, selection_rule, gen_length=256,
                  steps=256, block_length=128, temperature=1.0, seed=1):
    """Sample tokens, then commit the scheduled positions in the active block.

    Scores use float32 probabilities over legal tokens. Gumbel sampling uses
    the original LLaDA helper, with a constant logit shift to avoid overflow.
    Equal position scores prefer the earlier output position.
    """
    validate_generation(gen_length=gen_length, steps=steps, block_length=block_length,
                        temperature=temperature, seed=seed)
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.dtype != torch.long:
        raise ValueError("prompt_ids must be int64 with shape [1, P].")
    if type(mask_id) is not int or mask_id < 0:
        raise ValueError("mask_id must be a nonnegative integer.")
    if selection_rule not in {"random_position", "top_two_margin", "token_probability"}:
        raise ValueError(f"Unknown position selection rule: {selection_rule}")
    device = torch.device(getattr(model, "device", prompt_ids.device))
    torch.manual_seed(seed)
    prompt_len = prompt_ids.shape[1]
    state = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
    state[:, :prompt_len] = prompt_ids.to(device)
    calls, trace, selected_margins = 0, [], []
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    for block in range(num_blocks):
        start, end = prompt_len + block * block_length, prompt_len + (block + 1) * block_length
        schedule = get_num_transfer_tokens(state[:, start:end] == mask_id, steps_per_block)
        for step in range(steps_per_block):
            positions = torch.nonzero(state[0, start:end] == mask_id, as_tuple=False).flatten() + start
            count = int(schedule[0, step].item())
            prediction = model(state)
            calls += 1
            logits = prediction.logits[0, positions].float().clone()
            del prediction
            if logits.shape[-1] < 2 or mask_id >= logits.shape[-1]:
                raise ValueError("Model vocabulary must contain mask_id and a legal token.")
            logits[:, mask_id] = -torch.inf
            probabilities = logits.softmax(dim=-1)
            if not torch.isfinite(probabilities).all():
                raise ValueError("Non-finite model probabilities in masked decoding.")
            shifted = logits - logits.max(dim=-1, keepdim=True).values
            proposed = add_gumbel_noise(shifted, temperature).argmax(dim=-1)
            proposed_probabilities = probabilities.gather(-1, proposed[:, None]).squeeze(-1)
            if selection_rule == "random_position":
                scores = torch.rand(positions.numel(), device=device)
            elif selection_rule == "token_probability":
                scores = proposed_probabilities
            else:
                top_two = probabilities.topk(2, dim=-1).values
                scores = top_two[:, 0] - top_two[:, 1]
            chosen = torch.argsort(scores, descending=True, stable=True)[:count]
            state[0, positions[chosen]] = proposed[chosen]
            actions = [{"output_position": int(positions[i].item()) - prompt_len,
                        "token_id": int(proposed[i].item()),
                        "selection_score": float(scores[i].item()),
                        "token_probability": float(proposed_probabilities[i].item())}
                       for i in chosen.tolist()]
            if selection_rule == "top_two_margin":
                selected_margins.extend(action["selection_score"] for action in actions)
            trace.append({"stage": "commit", "block": block, "step": step,
                          "generation_nfes": calls, "actions": actions,
                          "remaining_masks": int((state[0, prompt_len:] == mask_id).sum().item())})
    remaining = int((state[0, prompt_len:] == mask_id).sum().item())
    if remaining:
        raise RuntimeError("The transfer schedule left masked output positions.")
    details = {"seed": seed, "selection_rule": selection_rule, "remaining_masks": remaining}
    if selected_margins:
        details["mean_selected_margin"] = sum(selected_margins) / len(selected_margins)
    return DecodeResult(state, calls, details=details, trace=trace)
