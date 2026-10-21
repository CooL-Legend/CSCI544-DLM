"""Lowest-predictive-entropy position decoding, without search."""

from dlm_project.decoding.common import decode_masked


def decode(model, prompt_ids, mask_id, *, gen_length=256, steps=256,
           block_length=256, temperature=0.0, seed=1, initial_state=None):
    """Rank non-mask-token distributions by entropy; T=0 commits argmax tokens."""
    return decode_masked(model, prompt_ids, mask_id, selection_rule="lowest_entropy",
                         gen_length=gen_length, steps=steps, block_length=block_length,
                         temperature=temperature, seed=seed, initial_state=initial_state)
