"""MEDAL's unchanged downstream sampler, starting without search initialization."""

from time import perf_counter

import torch

from dlm_project.decoding.common import initialize_state, validate_generation, validate_prompt
from dlm_project.decoding.decoder_result import DecodeResult
from dlm_project.decoding.mcts_confidence import _sync, resolve_mcts_config
from dlm_project.decoding.mcts_decoding import complete_medal, derive_seed, model_device


@torch.inference_mode()
def decode(model, prompt_ids, mask_id, *, gen_length=256, steps=256,
           block_length=256, temperature=1.0, seed=1, mcts=None, tokenizer=None,
           on_event=None):
    """Use the same fresh-forward completion and seed as searched MEDAL."""
    validate_generation(gen_length=gen_length, steps=steps, block_length=block_length,
                        temperature=temperature, seed=seed)
    validate_prompt(prompt_ids, mask_id)
    if steps != gen_length or block_length != gen_length or temperature <= 0:
        raise ValueError("MEDAL control requires steps = block_length = gen_length and positive temperature.")
    config = resolve_mcts_config(mcts, gen_length=gen_length)
    device = model_device(model)
    state = initialize_state(prompt_ids, mask_id, gen_length, device)
    completion_seed = derive_seed(seed, "final_completion")
    events = []

    def emit(event):
        event = {**event, "phase": "completion", "generation_nfes": event.get("model_calls", 0)}
        events.append(event)
        if on_event:
            on_event(event)

    _sync(device)
    started = perf_counter()
    result = complete_medal(model, state, prompt_ids.shape[1], mask_id, config,
                            seed=completion_seed, temperature=temperature,
                            tokenizer=tokenizer, on_event=emit)
    _sync(device)
    return DecodeResult(result.token_ids, result.model_calls, details={
        "search_model_calls": 0, "completion_model_calls": result.model_calls,
        "search_latency_seconds": 0.0, "completion_latency_seconds": perf_counter() - started,
        "selected_depth": 0, "seed": seed, "completion_seed": completion_seed,
        "remaining_masks": 0, "selection_source": "no_search", "fallback_decode_used": False,
    }, trace=events)
