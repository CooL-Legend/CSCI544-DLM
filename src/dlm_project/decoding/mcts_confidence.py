"""MEDAL reconstruction: complete MCTS rollouts, then adjusted-score sampling.

The historical method name is retained; implementation_version distinguishes
new runs from the earlier one-step/token-probability method.
"""

from dataclasses import asdict
from time import perf_counter
import torch

from dlm_project.decoding.common import validate_generation, validate_prompt
from dlm_project.decoding.decoder_result import DecodeResult
from dlm_project.decoding.mcts_decoding import (
    MCTSConfig, MCTSSearch, complete_medal, derive_seed, model_device, state_record,
)

DEFAULT_MCTS_CONFIG = MCTSConfig()


def resolve_mcts_config(settings=None, *, gen_length):
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
           block_length=256, temperature=1.0, seed=1, mcts=None, tokenizer=None,
           on_event=None):
    """Accept prepared prompt IDs; task runners own two-shot prompt augmentation."""
    validate_generation(gen_length=gen_length, steps=steps, block_length=block_length,
                        temperature=temperature, seed=seed)
    validate_prompt(prompt_ids, mask_id)
    if steps != gen_length or block_length != gen_length or temperature <= 0:
        raise ValueError("MEDAL requires steps = block_length = gen_length and positive temperature.")
    config = resolve_mcts_config(mcts, gen_length=gen_length)
    device = model_device(model)
    search = MCTSSearch(model, prompt_ids, mask_id, gen_length, config,
                        tokenizer=tokenizer, on_event=on_event, seed=seed)
    _sync(device)
    started = perf_counter()
    result = search.run()
    _sync(device)
    search_seconds = perf_counter() - started
    completion_seed = derive_seed(seed, "final_completion")
    events = []

    def emit(event):
        event = {**event, "phase": "completion",
                 "generation_nfes": result.stats["search_model_calls"] + event.get("model_calls", 0)}
        events.append(event)
        if on_event:
            on_event(event)

    started = perf_counter()
    completion = complete_medal(model, result.selected.state, search.prompt_len, mask_id,
                                config, seed=completion_seed, temperature=temperature,
                                tokenizer=tokenizer, on_event=emit)
    _sync(device)
    completion_seconds = perf_counter() - started
    details = {**result.stats, "mcts_config": asdict(config), "seed": seed,
               "completion_seed": completion_seed, "completion_model_calls": completion.model_calls,
               "search_latency_seconds": search_seconds, "completion_latency_seconds": completion_seconds,
               "remaining_masks": 0, "model_device": str(device), "search_state_device": "cpu",
               "selected_initialization": state_record(result.selected.state, search.prompt_len,
                                                        mask_id, tokenizer)}
    completion_steps = sum(event["stage"] == "rollout_step" for event in completion.trace)
    details.update(decoding_steps=completion_steps, selected_output_steps=result.selected.depth + completion_steps,
                   batch_size=1, state_evaluations=search.model_calls + completion.model_calls)
    return DecodeResult(completion.token_ids, search.model_calls + completion.model_calls,
                        details=details, trace=result.trace + events)
