"""Rank masked output positions by their top-two token probability margin."""

from dlm_project.decoding.common import decode_masked


def decode(model, prompt_ids, mask_id, *, gen_length=256, steps=256,
           block_length=128, temperature=1.0, seed=1):
    """Fill scheduled K positions with largest p(top1)-p(top2); sample tokens."""
    return decode_masked(model, prompt_ids, mask_id, selection_rule="top_two_margin",
                         gen_length=gen_length, steps=steps, block_length=block_length,
                         temperature=temperature, seed=seed)
