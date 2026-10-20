"""Global Entropy-Bounded sampling (arXiv:2505.24857, Section 4 / Algorithm 1).

Entropy and position ranking use unscaled probabilities over non-mask tokens.
Temperature controls token sampling only. No fixed transfer schedule is used.
"""

import math

import torch

from dlm_project.decoding.baseline import add_gumbel_noise
from dlm_project.decoding.common import initialize_state, validate_generation, validate_prompt
from dlm_project.decoding.decoder_result import DecodeResult
from dlm_project.decoding.signals import masked_state_signals


ERROR_PROXIES = ("confidence", "entropy", "margin")


def _validate_gamma(gamma):
    if (isinstance(gamma, bool) or not isinstance(gamma, (int, float))
            or math.isnan(gamma) or gamma < 0):
        raise ValueError("gamma must be nonnegative, finite or positive infinity.")


def _validate_error_proxy(error_proxy):
    if not isinstance(error_proxy, str) or error_proxy not in ERROR_PROXIES:
        raise ValueError(f"error_proxy must be one of {ERROR_PROXIES}.")


def resolve_config(settings=None):
    """Validate EB settings, representing infinite gamma as JSON-safe 'inf'."""
    settings = {} if settings is None else settings
    if not isinstance(settings, dict) or set(settings) - {"gamma", "error_proxy"}:
        raise ValueError("decoding.eb_sampler must contain only gamma and error_proxy.")
    gamma = settings.get("gamma", 0.1)
    if isinstance(gamma, str) and gamma == "inf":
        gamma = math.inf
    _validate_gamma(gamma)
    error_proxy = settings.get("error_proxy", "confidence")
    _validate_error_proxy(error_proxy)
    return {"gamma": "inf" if math.isinf(gamma) else float(gamma),
            "error_proxy": error_proxy}


def select_entropy_bounded_prefix(entropies, errors, gamma):
    """Return local indices in increasing error order under the entropy bound.

    Inputs describe one sample's eligible positions in output-position order.
    Stable sorting resolves equal errors by earlier position. The empty input
    returns no indices; every nonempty input selects at least one position.
    """
    _validate_gamma(gamma)
    if (not isinstance(entropies, torch.Tensor) or not isinstance(errors, torch.Tensor)
            or entropies.ndim != 1 or errors.shape != entropies.shape
            or entropies.device != errors.device):
        raise ValueError("entropies and errors must be equal-length 1-D tensors on one device.")
    if (not torch.isfinite(entropies).all() or (entropies < 0).any()
            or not torch.isfinite(errors).all()):
        raise ValueError("Entropies must be finite and nonnegative; errors must be finite.")
    order = torch.argsort(errors, stable=True)
    if not order.numel():
        return order
    sorted_entropy = entropies[order].double()
    bounds = sorted_entropy.cumsum(0) - sorted_entropy.cummax(0).values
    # For nonnegative entropy these bounds are monotone, and the first is zero.
    count = max(1, int((bounds <= gamma).sum().item()))
    return order[:count]


def _attention_mask(prompt_ids, attention_mask, gen_length, device):
    if attention_mask is None:
        return None
    if (not isinstance(attention_mask, torch.Tensor)
            or attention_mask.shape != prompt_ids.shape
            or not ((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError("attention_mask must have shape [B, P] and contain only 0 or 1.")
    mask = attention_mask.to(device)
    return torch.cat((mask, torch.ones((mask.shape[0], gen_length),
                                      dtype=mask.dtype, device=device)), dim=1)


@torch.inference_mode()
def decode(model, prompt_ids, mask_id, *, gen_length=256, steps=None,
           block_length=None, temperature=0.0, seed=1, gamma=0.1,
           error_proxy="confidence", initial_state=None, attention_mask=None):
    """Unmask all output targets globally with one full-batch forward per step.

    ``steps`` and ``block_length`` accept positive compatibility values but do
    not restrict EB's adaptive iterations or global position selection. Native
    model uncertainty is retained even when temperature=0 makes sampling greedy.
    Per-sample count histories have shape [B, iterations], padded with zeros
    after completion; per-sample averages use only that row's active iterations.
    """
    validate_generation(gen_length=gen_length, steps=gen_length,
                        block_length=gen_length, temperature=temperature, seed=seed)
    for name, value in (("steps", steps), ("block_length", block_length)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer or None.")
    _validate_gamma(gamma)
    _validate_error_proxy(error_proxy)
    validate_prompt(prompt_ids, mask_id, allow_batch=True)
    device = torch.device(getattr(model, "device", prompt_ids.device))
    state = initialize_state(prompt_ids, mask_id, gen_length, device, initial_state,
                             allow_batch=True)
    forward_mask = _attention_mask(prompt_ids, attention_mask, gen_length, device)
    torch.manual_seed(seed)
    batch_size, prompt_len = prompt_ids.shape
    calls, trace, counts = 0, [], []
    per_sample_counts = [[] for _ in range(batch_size)]
    logged_gamma = "inf" if math.isinf(gamma) else float(gamma)
    while (state[:, prompt_len:] == mask_id).any():
        rows, output_positions = torch.nonzero(state[:, prompt_len:] == mask_id, as_tuple=True)
        positions = output_positions + prompt_len
        prediction = model(state) if forward_mask is None else model(state, attention_mask=forward_mask)
        calls += 1
        if (prediction.logits.ndim != 3 or prediction.logits.shape[:2] != state.shape
                or prediction.logits.shape[-1] < 2 or mask_id >= prediction.logits.shape[-1]):
            raise ValueError("Model logits must have shape [B, P + gen_length, V] with mask_id and a legal token.")
        # Retain the existing native full-vocabulary diagnostic signal semantics.
        signals = [masked_state_signals(prediction.logits[row], positions[rows == row])
                   for row in range(batch_size)]
        logits = prediction.logits[rows, positions].float().clone()
        del prediction
        logits[:, mask_id] = -torch.inf
        probabilities = logits.softmax(-1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("Non-finite model probabilities in EB sampling.")
        entropies = -torch.special.xlogy(probabilities, probabilities).sum(-1)
        top_two = probabilities.topk(2, dim=-1).values
        if error_proxy == "confidence":
            errors = -top_two[:, 0]
        elif error_proxy == "entropy":
            errors = entropies
        else:
            errors = -(top_two[:, 0] - top_two[:, 1])
        selected, iteration_counts, entropy_bounds = [], [], []
        for row in range(batch_size):
            local = torch.nonzero(rows == row, as_tuple=False).flatten()
            chosen = local[select_entropy_bounded_prefix(entropies[local], errors[local], gamma)]
            selected.append(chosen)
            count = chosen.numel()
            iteration_counts.append(count)
            per_sample_counts[row].append(count)
            h = entropies[chosen].double()
            entropy_bounds.append(float((h.sum() - h.max()).item()) if count else 0.0)
        chosen = torch.cat(selected)
        selected_logits = logits[chosen]
        shifted = selected_logits - selected_logits.max(-1, keepdim=True).values
        proposed = add_gumbel_noise(shifted, temperature).argmax(-1)
        state[rows[chosen], positions[chosen]] = proposed
        token_probabilities = probabilities[chosen].gather(-1, proposed[:, None]).squeeze(-1)
        actions = [dict(batch_index=row, output_position=position, token_id=token,
                        selection_score=error, token_probability=probability, entropy=entropy)
                   for row, position, token, error, probability, entropy in zip(
                       rows[chosen].tolist(), output_positions[chosen].tolist(), proposed.tolist(),
                       errors[chosen].tolist(), token_probabilities.tolist(), entropies[chosen].tolist())]
        remaining = (state[:, prompt_len:] == mask_id).sum(-1).tolist()
        masked_count = rows.numel()
        counts.append(sum(iteration_counts))
        trace.append({"stage": "commit", "step": calls - 1, "generation_nfes": calls,
                      "actions": actions, "gamma": logged_gamma, "error_proxy": error_proxy,
                      "tokens_unmasked": counts[-1], "tokens_unmasked_per_sample": iteration_counts,
                      "entropy_bounds_per_sample": entropy_bounds,
                      "remaining_masks": sum(remaining), "remaining_masks_per_sample": remaining,
                      "signal_nfe": calls, "signal_extra_nfes": 0, "masked_count": masked_count,
                      "mean_token_entropy": sum(s["mean_token_entropy"] * s["masked_count"]
                                                for s in signals if s["masked_count"]) / masked_count,
                      "mean_confidence_margin": sum(s["mean_confidence_margin"] * s["masked_count"]
                                                    for s in signals if s["masked_count"]) / masked_count,
                      "signals_per_sample": signals})
        del logits, probabilities, selected_logits, shifted
    active_steps = [sum(count > 0 for count in history) for history in per_sample_counts]
    details = {"seed": seed, "selection_rule": "entropy_bounded", "selection_scope": "global",
               "gamma": logged_gamma, "error_proxy": error_proxy, "temperature": temperature,
               "remaining_masks": 0, "stop_reason": "completed", "decoding_steps": calls,
               "selected_output_steps": calls, "decoding_steps_per_sample": active_steps,
               "batch_size": batch_size, "state_evaluations": batch_size * calls,
               "tokens_unmasked_per_iteration": counts,
               "average_tokens_unmasked_per_iteration": sum(counts) / calls if calls else 0.0,
               "tokens_unmasked_per_iteration_per_sample": per_sample_counts,
               "average_tokens_unmasked_per_iteration_per_sample": [
                   sum(history) / active if active else 0.0
                   for history, active in zip(per_sample_counts, active_steps)]}
    return DecodeResult(state, calls, details=details, trace=trace)
