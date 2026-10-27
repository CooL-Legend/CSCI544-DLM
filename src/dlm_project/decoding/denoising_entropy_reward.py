"""Trajectory-level Denoising Entropy reward for masked-diffusion search.

This module uses entropy statistics already cached by MEDAL, so computing this
reward requires no additional model forward evaluations.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DenoisingEntropyResult:
    """Denoising Entropy statistics for one root-to-child trajectory."""

    state_entropies: tuple[float, ...]
    path_entropy: float
    root_entropy: float
    reward: float


def state_entropy(analysis, state, prompt_len, mask_id):
    """Return mean marginal entropy over the state's remaining masked tokens."""
    if analysis is None:
        raise ValueError("State analysis is required.")
    if state.ndim != 2 or state.shape[0] != 1:
        raise ValueError("State must have shape [1, sequence_length].")
    if not 0 <= prompt_len <= state.shape[1]:
        raise ValueError("Invalid prompt length.")

    remaining_masks = int(
        (state[0, prompt_len:] == mask_id).sum().item()
    )

    if remaining_masks == 0:
        return 0.0

    value = analysis.total_entropy / remaining_masks
    if not math.isfinite(value):
        raise ValueError("State entropy must be finite.")

    return float(value)


def compute_denoising_entropy_reward(
    parent,
    child_state,
    child_analysis,
    prompt_len,
    mask_id,
    epsilon=1e-8,
):
    """Compute normalized trajectory-level Denoising Entropy reward.

    The trajectory contains every cached state from the root through the parent,
    followed by the newly evaluated child state.

    Lower Path Denoising Entropy is better. MEDAL maximizes rewards, so this is
    converted to a root-normalized utility:

        reward = (root_state_entropy - path_entropy)
                 / max(root_state_entropy, epsilon)

    For a fixed root, maximizing this reward is equivalent to minimizing the
    trajectory's Path Denoising Entropy.
    """
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive.")

    ancestors = []
    node = parent

    while node is not None:
        if node.analysis is None:
            raise ValueError("Every trajectory node must have cached analysis.")
        ancestors.append(node)
        node = node.parent

    ancestors.reverse()

    values = [
        state_entropy(
            node.analysis,
            node.state,
            prompt_len,
            mask_id,
        )
        for node in ancestors
    ]

    values.append(
        state_entropy(
            child_analysis,
            child_state,
            prompt_len,
            mask_id,
        )
    )

    if not values:
        raise ValueError("The trajectory must contain at least one state.")

    path_entropy = sum(values) / len(values)
    root_entropy = values[0]
    reward = (
        (root_entropy - path_entropy)
        / max(root_entropy, epsilon)
    )

    if not math.isfinite(path_entropy) or not math.isfinite(reward):
        raise ValueError("Denoising Entropy result must be finite.")

    return DenoisingEntropyResult(
        state_entropies=tuple(values),
        path_entropy=float(path_entropy),
        root_entropy=float(root_entropy),
        reward=float(reward),
    )
