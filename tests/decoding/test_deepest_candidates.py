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


def test_depth_with_at_least_c_distinct_nodes_is_chosen():
    root = node(0, [0, 0, 0], 0, total=0, visits=0, immediate=None, parent=None)
    # Depth 1 has 3 distinct states (>= C=3)
    d1_a = node(1, [1, 0, 0], 1, total=10, visits=1, immediate=10, parent=root)
    d1_b = node(2, [2, 0, 0], 1, total=20, visits=1, immediate=20, parent=root)
    d1_c = node(3, [3, 0, 0], 1, total=30, visits=1, immediate=30, parent=root)
    # Depth 2 only has 2 distinct states (< C=3)
    d2_a = node(4, [1, 1, 0], 2, total=50, visits=1, immediate=50, parent=d1_a)
    d2_b = node(5, [1, 2, 0], 2, total=60, visits=1, immediate=60, parent=d1_a)
    candidates, shortlist = select_deepest_candidates([root, d1_a, d1_b, d1_c, d2_a, d2_b], limit=3)
    # Even though depth 2 is deeper, depth 1 has >= 3 distinct nodes and is chosen!
    assert [n.node_id for n in candidates] == [1, 2, 3]
    assert [n.node_id for n in shortlist] == [3, 2, 1]


def test_deepest_among_depths_with_at_least_c_distinct_nodes_is_chosen():
    root = node(0, [0, 0, 0], 0, total=0, visits=0, immediate=None, parent=None)
    # Depth 1 has 3 distinct states
    d1_a = node(1, [1, 0, 0], 1, total=10, visits=1, immediate=10, parent=root)
    d1_b = node(2, [2, 0, 0], 1, total=20, visits=1, immediate=20, parent=root)
    d1_c = node(3, [3, 0, 0], 1, total=30, visits=1, immediate=30, parent=root)
    # Depth 2 also has 3 distinct states
    d2_a = node(4, [1, 1, 0], 2, total=5, visits=1, immediate=5, parent=d1_a)
    d2_b = node(5, [1, 2, 0], 2, total=15, visits=1, immediate=15, parent=d1_a)
    d2_c = node(6, [2, 1, 0], 2, total=25, visits=1, immediate=25, parent=d1_b)
    # Depth 3 has only 1 distinct state
    d3_a = node(7, [1, 1, 1], 3, total=100, visits=1, immediate=100, parent=d2_a)

    candidates, shortlist = select_deepest_candidates([root, d1_a, d1_b, d1_c, d2_a, d2_b, d2_c, d3_a], limit=3)
    # Depth 2 is the deepest with >= 3 distinct states
    assert [n.node_id for n in candidates] == [4, 5, 6]
    assert [n.node_id for n in shortlist] == [6, 5, 4]


def test_most_distinct_fallback_when_no_depth_has_c_distinct():
    root = node(0, [0, 0, 0], 0, total=0, visits=0, immediate=None, parent=None)
    # Depth 1 has 2 distinct states
    d1_a = node(1, [1, 0, 0], 1, total=10, visits=1, immediate=10, parent=root)
    d1_b = node(2, [2, 0, 0], 1, total=20, visits=1, immediate=20, parent=root)
    # Depth 2 has 1 distinct state
    d2_a = node(3, [1, 1, 0], 2, total=50, visits=1, immediate=50, parent=d1_a)

    # Limit is 3, neither has 3 distinct states. Depth 1 has 2 states vs depth 2 with 1.
    candidates, shortlist = select_deepest_candidates([root, d1_a, d1_b, d2_a], limit=3)
    assert [n.node_id for n in candidates] == [1, 2]

    # If tied on count (both have 2), deeper depth wins the tie
    d2_b = node(4, [1, 2, 0], 2, total=40, visits=1, immediate=40, parent=d1_a)
    candidates_tied, _ = select_deepest_candidates([root, d1_a, d1_b, d2_a, d2_b], limit=3)
    assert [n.node_id for n in candidates_tied] == [3, 4]

