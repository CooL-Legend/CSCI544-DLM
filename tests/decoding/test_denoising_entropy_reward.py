from types import SimpleNamespace

import pytest
import torch

from dlm_project.decoding.denoising_entropy_reward import (
    compute_denoising_entropy_reward,
    state_entropy,
)


MASK = 99
PROMPT_LEN = 1


def analysis(entropies):
    tensor = torch.tensor(entropies, dtype=torch.float32)
    return SimpleNamespace(
        entropies=tensor,
        total_entropy=tensor.double().sum().item(),
    )


def node(state, entropies, parent=None):
    return SimpleNamespace(
        state=torch.tensor([state], dtype=torch.long),
        analysis=analysis(entropies),
        parent=parent,
    )


def test_state_entropy_averages_only_over_remaining_masks():
    current = node(
        [7, 10, MASK, MASK],
        [0.0, 2.0, 4.0],
    )

    result = state_entropy(
        current.analysis,
        current.state,
        PROMPT_LEN,
        MASK,
    )

    assert result == pytest.approx(3.0)


def test_state_entropy_counts_zero_entropy_masked_positions():
    current = node(
        [7, 10, MASK, MASK],
        [0.0, 0.0, 0.0],
    )

    result = state_entropy(
        current.analysis,
        current.state,
        PROMPT_LEN,
        MASK,
    )

    assert result == 0.0


def test_trajectory_reward_uses_all_root_to_child_states():
    root = node(
        [7, MASK, MASK, MASK],
        [3.0, 3.0, 3.0],
    )
    parent = node(
        [7, 10, MASK, MASK],
        [0.0, 2.0, 2.0],
        parent=root,
    )

    child_state = torch.tensor(
        [[7, 10, 11, MASK]],
        dtype=torch.long,
    )
    child_analysis = analysis([0.0, 0.0, 1.0])

    result = compute_denoising_entropy_reward(
        parent,
        child_state,
        child_analysis,
        PROMPT_LEN,
        MASK,
    )

    assert result.state_entropies == pytest.approx((3.0, 2.0, 1.0))
    assert result.path_entropy == pytest.approx(2.0)
    assert result.root_entropy == pytest.approx(3.0)
    assert result.reward == pytest.approx(1.0 / 3.0)


def test_complete_child_has_zero_state_entropy():
    root = node(
        [7, MASK],
        [2.0],
    )

    child_state = torch.tensor(
        [[7, 10]],
        dtype=torch.long,
    )
    child_analysis = analysis([0.0])

    result = compute_denoising_entropy_reward(
        root,
        child_state,
        child_analysis,
        PROMPT_LEN,
        MASK,
    )

    assert result.state_entropies == pytest.approx((2.0, 0.0))
    assert result.path_entropy == pytest.approx(1.0)
    assert result.reward == pytest.approx(0.5)



def test_mcts_config_rejects_unknown_reward_type():
    from dlm_project.decoding import mcts_decoding as medal

    with pytest.raises(ValueError, match="reward_type"):
        medal.MCTSConfig(reward_type="unknown")


def _apply_reward(reward_type):
    from dlm_project.decoding import mcts_decoding as medal

    root = medal.Node(
        node_id=0,
        state=torch.tensor(
            [[7, MASK, MASK, MASK]],
            dtype=torch.long,
        ),
    )
    root.analysis = medal.StateAnalysis(
        entropies=torch.tensor([3.0, 3.0, 3.0]),
        actions=(),
        state_key=(MASK, MASK, MASK),
    )

    action = medal.Action(
        output_position=0,
        token_id=10,
        confidence_score=1.0,
    )
    child = medal.expand_node(
        root,
        action,
        node_id=1,
        prompt_len=PROMPT_LEN,
        mask_id=MASK,
        init_tokens=2,
    )

    child_analysis = medal.StateAnalysis(
        entropies=torch.tensor([0.0, 2.0, 2.0]),
        actions=(),
        state_key=(10, MASK, MASK),
    )

    config = medal.MCTSConfig(
        init_tokens=2,
        reward_type=reward_type,
    )

    return medal.apply_child_analysis(
        root,
        child,
        child_analysis,
        PROMPT_LEN,
        MASK,
        config,
    )


def test_apply_child_analysis_can_use_information_gain():
    result = _apply_reward("information_gain")

    assert result["reward_type"] == "information_gain"
    assert result["immediate_gain"] == pytest.approx(5.0 / 9.0)
    assert result["path_denoising_entropy"] == pytest.approx(2.5)
    assert result["denoising_entropy_reward"] == pytest.approx(1.0 / 6.0)
    assert result["reward"] == pytest.approx(5.0 / 9.0)


def test_apply_child_analysis_can_use_denoising_entropy():
    result = _apply_reward("denoising_entropy")

    assert result["reward_type"] == "denoising_entropy"
    assert result["immediate_gain"] == pytest.approx(5.0 / 9.0)
    assert result["trajectory_state_entropies"] == pytest.approx([3.0, 2.0])
    assert result["path_denoising_entropy"] == pytest.approx(2.5)
    assert result["denoising_entropy_reward"] == pytest.approx(1.0 / 6.0)
    assert result["reward"] == pytest.approx(1.0 / 6.0)
