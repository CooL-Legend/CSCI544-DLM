"""Entropy-based MCTS initialization followed by token-probability sampling.

Search keeps the existing one-step entropy evaluator. Completion recomputes
probabilities for the selected state and uses the same sampler as confidence.py.
"""

from dataclasses import asdict
from time import perf_counter

import torch

from dlm_project.decoding.common import validate_generation, validate_prompt
from dlm_project.decoding.confidence import decode as confidence_decode
from dlm_project.decoding.decoder_result import DecodeResult
from dlm_project.decoding.mcts_decoding import MCTSConfig, MCTSSearch, model_device, state_record


DEFAULT_MCTS_CONFIG = MCTSConfig(init_tokens=20, num_candidates=3,
                                top_k_tokens=3, top_k_actions=5,
                                max_model_calls=512)


def resolve_mcts_config(settings=None, *, gen_length):
    """Apply this method's defaults without changing legacy MCTS defaults."""
    if isinstance(settings, MCTSConfig):
        config = settings
    else:
        if settings is not None and not isinstance(settings, dict):
            raise ValueError("decoding.mcts must be a mapping or MCTSConfig.")
        try:
            config = MCTSConfig(**{**asdict(DEFAULT_MCTS_CONFIG), **(settings or {})})
        except TypeError as error:
            raise ValueError(f"Invalid decoding.mcts settings: {error}") from error
    config.validate_length(gen_length)
    return config


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def decode(model, prompt_ids, mask_id, *, gen_length=256, steps=256,
           block_length=128, temperature=1.0, seed=1, mcts=None, tokenizer=None):
    """Search once, then complete the selected state using the same loaded model.

    The cap includes root and all child evaluations, and excludes completion.
    Budget/frontier fallbacks are completed and retain their actual depth.
    """
    validate_generation(gen_length=gen_length, steps=steps, block_length=block_length,
                        temperature=temperature, seed=seed)
    validate_prompt(prompt_ids, mask_id)
    config = resolve_mcts_config(mcts, gen_length=gen_length)
    device = model_device(model)
    search = MCTSSearch(model, prompt_ids, mask_id, gen_length, config, tokenizer=tokenizer)
    _sync(device)
    started = perf_counter()
    result = search.run()
    _sync(device)
    search_seconds = perf_counter() - started

    started = perf_counter()
    completion = confidence_decode(
        model, prompt_ids, mask_id, gen_length=gen_length, steps=steps,
        block_length=block_length, temperature=temperature, seed=seed,
        initial_state=result.selected.state,
    )
    _sync(device)
    completion_seconds = perf_counter() - started
    search_calls = result.stats["search_model_calls"]
    details = {**completion.details, **result.stats, "mcts": asdict(config),
               "completion_model_calls": completion.total_nfes,
               "search_latency_seconds": search_seconds,
               "completion_latency_seconds": completion_seconds,
               "model_device": str(device), "search_state_device": str(result.selected.state.device),
               "selected_initialization": state_record(result.selected.state, prompt_ids.shape[1],
                                                       mask_id, tokenizer)}
    trace = [{**event, "phase": "search"} for event in result.trace]
    trace.extend({**event, "phase": "completion",
                  "completion_model_calls": event["generation_nfes"],
                  "generation_nfes": search_calls + event["generation_nfes"]}
                 for event in completion.trace)
    return DecodeResult(completion.token_ids, search_calls + completion.generation_nfes,
                        scoring_nfes=completion.scoring_nfes, details=details, trace=trace)
