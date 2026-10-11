"""Greedy argmax tokens, choosing K masked positions by top-two margin."""

import torch

from dlm_project.decoding.decoder_result import DecodeResult


@torch.inference_mode()
def decode(model, prompt_ids, mask_id, *, gen_length, steps,
           top_k_positions=1):
    """Commit up to K positions per forward by p(top1)-p(top2).

    Probabilities and the margin include the mask token, as in MEDAL Eq. 5;
    commitments use the highest-probability legal (non-mask) token. Ties prefer
    lower output positions. Prompt IDs [1,P] remain unchanged in [1,P+L].
    """
    if type(top_k_positions) is not int or not 1 <= top_k_positions <= gen_length:
        raise ValueError("top_k_positions must be an integer in [1, gen_length].")
    if type(steps) is not int or steps < 1 or steps * top_k_positions < gen_length:
        raise ValueError("steps * top_k_positions must cover gen_length.")
    device = torch.device(getattr(model, "device", prompt_ids.device))
    state = torch.full((1, prompt_ids.shape[1] + gen_length), mask_id,
                       dtype=torch.long, device=device)
    state[:, :prompt_ids.shape[1]] = prompt_ids.to(device)
    calls, events, margins = 0, [], []
    while True:
        masked = torch.nonzero(state[0, prompt_ids.shape[1]:] == mask_id, as_tuple=False).flatten()
        if not masked.numel():
            break
        if calls == steps:
            raise RuntimeError("Top-K Margin reached its step limit with masked output positions.")
        prediction = model(state)
        calls += 1
        logits = prediction.logits[0, (masked + prompt_ids.shape[1]).to(device)].float()
        del prediction
        probabilities = torch.softmax(logits, dim=-1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("Non-finite model probabilities in Top-K Margin.")
        top_two = probabilities.topk(2, dim=-1).values
        gaps = (top_two[:, 0] - top_two[:, 1]).cpu().tolist()
        legal = probabilities.clone()
        legal[:, mask_id] = -1
        tokens = legal.argmax(dim=-1).cpu().tolist()
        positions = masked.cpu().tolist()
        chosen = sorted(range(len(positions)), key=lambda index: (-gaps[index], positions[index]))[:top_k_positions]
        actions = []
        for index in chosen:
            position, token = positions[index], tokens[index]
            state[0, prompt_ids.shape[1] + position] = token
            margins.append(gaps[index])
            actions.append({"output_position": position, "token_id": token,
                            "top_two_margin": gaps[index],
                            "token_probability": float(probabilities[index, token].item())})
        events.append({"stage": "commit", "generation_nfes": calls,
                       "remaining_masks": len(positions) - len(chosen), "actions": actions})
    return DecodeResult(state, calls, details={"top_k_positions": top_k_positions,
                                               "mean_selected_margin": sum(margins) / len(margins),
                                               "remaining_masks": 0}, trace=events)
