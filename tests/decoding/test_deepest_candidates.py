"""Final selection is depth-first, distinct-state, backed-up-mean ranking."""

import torch

from dlm_project.decoding.mcts_decoding import Node, StateAnalysis, select_deepest_candidates


def node(identifier, ids, depth, *, total, visits, immediate, parent):
    state = torch.tensor([ids], dtype=torch.long)
    return Node(identifier, state, parent=parent, depth=depth, visits=visits, reward_sum=total,
                immediate_reward=immediate, analysis=StateAnalysis(torch.zeros(len(ids)), (), tuple(ids)))


def test_means_override_immediate_rewards_and_first_duplicate_wins():
    root = node(0, [0, 0, 0], 0, total=0, visits=0, immediate=None, parent=None)
    first = node(1, [1, 0, 0], 1, total=-4, visits=4, immediate=50, parent=root)
    better = node(2, [2, 0, 0], 1, total=-1, visits=2, immediate=-50, parent=root)
    duplicate = node(3, [1, 0, 0], 1, total=100, visits=1, immediate=100, parent=root)
    tied = node(4, [3, 0, 0], 1, total=-2, visits=4, immediate=-60, parent=root)
    last = node(5, [4, 0, 0], 1, total=-2, visits=1, immediate=-2, parent=root)
    candidates, shortlist = select_deepest_candidates([duplicate, last, root, tied, better, first])
    assert [n.node_id for n in candidates] == [1, 2, 4, 5]
    assert [n.node_id for n in shortlist] == [2, 4, 1]
    assert first.visits == 4 and first.reward_sum == -4


def test_deeper_evaluated_states_win_and_unfinished_expansion_is_ineligible():
    root = node(0, [0, 0, 0], 0, total=0, visits=0, immediate=None, parent=None)
    shallow = node(1, [1, 0, 0], 1, total=100, visits=1, immediate=100, parent=root)
    deep = node(2, [1, 2, 0], 2, total=-10, visits=1, immediate=-10, parent=shallow)
    unfinished = Node(3, torch.tensor([[1, 2, 3]]), parent=deep, depth=3)
    candidates, shortlist = select_deepest_candidates([root, shallow, deep, unfinished])
    assert candidates == shortlist == [deep]
    assert select_deepest_candidates([root]) == ([root], [root])
