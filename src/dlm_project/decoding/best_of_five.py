"""Five independent random MDM candidates; choose by masked-block self-score."""

from time import perf_counter

import torch

from dlm_project.decoding.decoder_result import DecodeResult
from dlm_project.decoding.random_mdm import decode as random_decode


def _sync(model):
    device = torch.device(getattr(model, "device", "cpu"))
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def masked_block_score(model, candidate, prompt_len, mask_id, block_size):
    """Mean token log p when each output block is masked in turn.

    No token sees itself: the evaluated block is replaced by masks. Returns
    (mean log probability, actual scoring forwards). References are not inputs.
    """
    length = candidate.shape[1] - prompt_len
    if ((candidate[:, prompt_len:] == mask_id).any() or length < 1):
        return -1e30, 0
    total, calls = 0.0, 0
    for start in range(0, length, block_size):
        stop = min(start + block_size, length)
        masked = candidate.clone()
        masked[:, prompt_len + start:prompt_len + stop] = mask_id
        prediction = model(masked)
        calls += 1
        rows = prediction.logits[0, prompt_len + start:prompt_len + stop].float()
        targets = candidate[0, prompt_len + start:prompt_len + stop].to(rows.device)
        selected = rows.log_softmax(dim=-1).gather(-1, targets[:, None]).squeeze(-1)
        if not torch.isfinite(selected).all():
            raise ValueError("Non-finite masked-block candidate score.")
        total += selected.sum().item()
        del prediction, masked, rows, selected
    return total / length, calls


@torch.inference_mode()
def decode(model, prompt_ids, mask_id, *, gen_length, steps, block_length,
           temperature=0.0, seed=1, score_block_size=32):
    """Generate five random paths; rank by a reference-free denoising proxy.

    All five generation NFEs plus every masked-block scoring NFE are charged.
    Stable ties select the earliest candidate. Candidate seeds are seed+[0..4].
    """
    if type(score_block_size) is not int or not 1 <= score_block_size <= gen_length:
        raise ValueError("score_block_size must be an integer in [1, gen_length].")
    candidates, scores, events = [], [], []
    generation_nfes = scoring_nfes = 0
    _sync(model)
    started = perf_counter()
    for index in range(5):
        result = random_decode(model, prompt_ids, mask_id, gen_length=gen_length,
                               steps=steps, block_length=block_length,
                               temperature=temperature, seed=seed + index)
        candidates.append(result.token_ids)
        generation_nfes += result.generation_nfes
        events.append({"stage": "candidate", "index": index, "seed": seed + index,
                       "generation_nfes": result.generation_nfes,
                       "remaining_masks": result.details["remaining_masks"]})
    _sync(model)
    generation_seconds = perf_counter() - started
    started = perf_counter()
    for index, candidate in enumerate(candidates):
        score, calls = masked_block_score(model, candidate, prompt_ids.shape[1],
                                         mask_id, score_block_size)
        scores.append(score)
        scoring_nfes += calls
        events.append({"stage": "score", "index": index, "mean_masked_log_probability": score,
                       "scoring_nfes": calls})
    _sync(model)
    scoring_seconds = perf_counter() - started
    winner = max(range(5), key=lambda index: (scores[index], -index))
    output_text_keys = [tuple(candidate[0, prompt_ids.shape[1]:].cpu().tolist()) for candidate in candidates]
    events.append({"stage": "selection", "selected_index": winner,
                   "selection_score": scores[winner],
                   "total_nfes": generation_nfes + scoring_nfes})
    return DecodeResult(candidates[winner], generation_nfes, scoring_nfes,
                        details={"candidate_count": 5, "candidate_seeds": [seed + i for i in range(5)],
                                 "candidate_scores": scores, "selected_candidate_index": winner,
                                 "selection_score": scores[winner],
                                 "unique_candidate_count": len(set(output_text_keys)),
                                 "score_block_size": score_block_size,
                                 "generation_seconds": generation_seconds,
                                 "scoring_seconds": scoring_seconds,
                                 "remaining_masks": int((candidates[winner][:, prompt_ids.shape[1]:]
                                                         == mask_id).sum().item())},
                        trace=events)
