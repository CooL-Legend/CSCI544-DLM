"""Search-free greedy completion using MEDAL's confidence-adjusted action score."""

import torch

from dlm_project.decoding.decoder_result import DecodeResult
from dlm_project.decoding.mcts_decoding import MCTSConfig, complete_confidence


@torch.inference_mode()
def decode(model, prompt_ids, mask_id, *, gen_length, margin_gamma=10.0):
    """Fill one output mask per forward using p*exp(-H)*sigmoid(gamma*margin).

    Reuses the existing MCTS completion action scorer, without constructing a
    search tree. Prompt IDs [1,P] are copied into [1,P+L] and remain fixed.
    """
    config = MCTSConfig(top_k_tokens=1, top_k_actions=1, margin_gamma=margin_gamma)
    state = torch.full((1, prompt_ids.shape[1] + gen_length), mask_id, dtype=torch.long)
    state[:, :prompt_ids.shape[1]] = prompt_ids.detach().cpu()
    completion = complete_confidence(model, state, prompt_ids.shape[1], mask_id, config)
    return DecodeResult(completion.token_ids, completion.model_calls,
                        details={"margin_gamma": margin_gamma, "remaining_masks": 0},
                        trace=completion.trace)
