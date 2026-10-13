"""Compact uncertainty measurements from logits already produced by a forward."""

import torch


@torch.inference_mode()
def masked_state_signals(logits, positions, *, epsilon=1e-8, chunk_size=32):
    """Mean entropy (nats) and top-two gap over all unresolved output slots.

    logits has shape [sequence, vocabulary], positions is a 1-D index tensor.
    Use the native full-vocabulary distribution, before excluding the mask ID.
    Chunking bounds temporary GPU memory; no logits are retained in the trace.
    This function performs no model forwards.
    """
    count = positions.numel()
    if not count:
        return {"masked_count": 0, "mean_token_entropy": None, "mean_confidence_margin": None}
    totals = torch.zeros(2, device=logits.device, dtype=torch.float64)
    for indices in positions.to(logits.device).split(chunk_size):
        probabilities = logits[indices].float().softmax(-1)
        entropy = -(probabilities * (probabilities + epsilon).log()).sum(-1)
        top_two = probabilities.topk(2, dim=-1).values
        totals[0] += entropy.double().sum()
        totals[1] += (top_two[:, 0] - top_two[:, 1]).double().sum()
    entropy, margin = (totals / count).cpu().tolist()
    if not all(torch.isfinite(totals)):
        raise ValueError("Non-finite uncertainty signal.")
    return {"masked_count": count, "mean_token_entropy": entropy,
            "mean_confidence_margin": margin}
