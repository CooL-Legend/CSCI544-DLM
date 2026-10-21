"""Normal masked diffusion decoding with random position selection."""

from dlm_project.decoding.common import decode_masked


def decode(model, prompt_ids, mask_id, *, gen_length=256, steps=256,
           block_length=128, temperature=1.0, seed=1):
    """Sample tokens and randomly fill the scheduled masked output positions."""
    return decode_masked(model, prompt_ids, mask_id, selection_rule="random_position",
                         gen_length=gen_length, steps=steps, block_length=block_length,
                         temperature=temperature, seed=seed)
