"""LLaDA token-probability confidence decoding, without search."""

from dlm_project.decoding.common import decode_masked


def decode(model, prompt_ids, mask_id, *, gen_length=256, steps=256,
           block_length=128, temperature=1.0, seed=1):
    """Fill positions with the highest probabilities of their sampled tokens."""
    return decode_masked(model, prompt_ids, mask_id, selection_rule="token_probability",
                         gen_length=gen_length, steps=steps, block_length=block_length,
                         temperature=temperature, seed=seed)
