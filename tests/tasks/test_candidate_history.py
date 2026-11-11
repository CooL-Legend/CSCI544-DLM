"""Tests for candidate history extraction and question score comparison."""

import json
from pathlib import Path
import pytest

from dlm_project.tasks.gsm8k.candidate_history import (
    compute_negative_ig_allocation,
    extract_distinct_candidates,
    extract_candidate_progression,
    load_question_comparison,
)


def _sample_tree_data():
    return {
        "nodes": [
            {"node_id": 0, "parent_id": None, "depth": 0, "evaluation_order": 0, "visits": 4, "mean_reward": 0.1, "state_hash": "h_root"},
            {"node_id": 1, "parent_id": 0, "depth": 1, "evaluation_order": 1, "visits": 2, "mean_reward": 0.3, "state_hash": "h_d1_a"},
            {"node_id": 2, "parent_id": 0, "depth": 1, "evaluation_order": 2, "visits": 1, "mean_reward": 0.2, "state_hash": "h_d1_b"},
            {"node_id": 3, "parent_id": 0, "depth": 1, "evaluation_order": 3, "visits": 1, "mean_reward": 0.4, "state_hash": "h_d1_c"},
            {"node_id": 4, "parent_id": 1, "depth": 2, "evaluation_order": 4, "visits": 1, "mean_reward": 0.5, "state_hash": "h_d2_a"},
            {"node_id": 5, "parent_id": 1, "depth": 2, "evaluation_order": 5, "visits": 1, "mean_reward": 0.6, "state_hash": "h_d2_b"},
            # Node 6 has duplicate state with Node 5
            {"node_id": 6, "parent_id": 2, "depth": 2, "evaluation_order": 6, "visits": 1, "mean_reward": 0.1, "state_hash": "h_d2_b"},
        ],
        "selection": {
            "selected_node_id": 3,
            "selected_depth": 1,
            "candidate_ids": [1, 2, 3],
            "shortlist_ids": [3, 1, 2],
        }
    }


def test_extract_distinct_candidates():
    data = _sample_tree_data()
    # At min_c = 3:
    # Depth 0: 1 distinct
    # Depth 1: 3 distinct (h_d1_a, h_d1_b, h_d1_c) -> qualifies!
    # Depth 2: 2 distinct (h_d2_a, h_d2_b) -> only 2 distinct, does not qualify for min_c=3
    # Therefore d* should be 1!
    res = extract_distinct_candidates(data, min_c=3)
    assert res["candidate_depth"] == 1
    assert res["candidate_count"] == 3
    assert res["distinct_candidates_by_depth"] == {0: 1, 1: 3, 2: 2}
    assert res["qualifying_depths"] == [1]
    assert len(res["candidates"]) == 3

    # Sorted by mean_reward descending: Node 3 (0.4), Node 1 (0.3), Node 2 (0.2)
    assert [c["node_id"] for c in res["candidates"]] == [3, 1, 2]
    assert res["candidates"][0]["is_selected"] is True
    assert res["candidates"][0]["is_shortlist"] is True


def test_extract_distinct_candidates_fallback():
    data = _sample_tree_data()
    # At min_c = 5: no depth has 5 distinct states.
    # Fallback picks depth with most states (ties broken by deeper depth) -> depth 1 (has 3).
    res = extract_distinct_candidates(data, min_c=5)
    assert res["candidate_depth"] == 1
    assert res["candidate_count"] == 3
    assert res["qualifying_depths"] == []


def test_extract_candidate_progression():
    data = _sample_tree_data()
    prog = extract_candidate_progression(data)
    assert len(prog) == 7
    # Node 6 has duplicate state hash at depth 2, so is_new_state should be False
    assert prog[5]["is_new_state"] is True
    assert prog[6]["is_new_state"] is False
    assert prog[6]["cumulative_distinct_total"] == 6


def test_load_question_comparison(tmp_path):
    budget = 512
    stage = "dev"
    for method in ("medal", "ab_medal"):
        for q_id, is_match in [("q1", True), ("q2", method == "ab_medal")]:
            chk_dir = tmp_path / stage / method / q_id / "checkpoints" / str(budget)
            chk_dir.mkdir(parents=True, exist_ok=True)
            met = {
                "example_id": q_id,
                "method": method,
                "search_budget": budget,
                "question": f"Question {q_id} text",
                "reference_answer": "42",
                "predicted_answer": "42" if is_match else "0",
                "exact_match": is_match,
                "selected_depth": 5,
                "selected_q": 0.8 if is_match else 0.1,
                "total_nfes": 700,
            }
            (chk_dir / "metrics.json").write_text(json.dumps(met))

    q_rows, agg = load_question_comparison(tmp_path, stage=stage, budget=budget)
    assert len(q_rows) == 2
    assert q_rows[0]["example_id"] == "q1"
    assert q_rows[0]["outcome"] == "Both Correct (Tie)"
    assert q_rows[1]["example_id"] == "q2"
    assert q_rows[1]["outcome"] == "AB-MEDAL Win"

    assert agg["head_to_head"]["ab_medal_wins"] == 1
    assert agg["head_to_head"]["medal_wins"] == 0
    assert agg["head_to_head"]["both_correct_ties"] == 1
    assert agg["methods"]["medal"]["accuracy"] == 0.5
    assert agg["methods"]["ab_medal"]["accuracy"] == 1.0


def test_compute_negative_ig_allocation():
    data = {
        "nodes": [
            {"node_id": 0, "parent_id": None, "depth": 0, "visits": 5, "immediate_reward": None, "mean_reward": 0.0},
            # Child 1: positive IG (+0.4)
            {"node_id": 1, "parent_id": 0, "depth": 1, "visits": 3, "immediate_reward": 0.4, "mean_reward": 0.5},
            # Child 2: negative IG (-0.3)
            {"node_id": 2, "parent_id": 0, "depth": 1, "visits": 1, "immediate_reward": -0.3, "mean_reward": -0.2},
            # Grandchild 3: under child 1 (positive branch)
            {"node_id": 3, "parent_id": 1, "depth": 2, "visits": 1, "immediate_reward": 0.2, "mean_reward": 0.3},
            # Grandchild 4: under child 1 (positive branch)
            {"node_id": 4, "parent_id": 1, "depth": 2, "visits": 1, "immediate_reward": 0.1, "mean_reward": 0.2},
        ]
    }
    alloc = compute_negative_ig_allocation(data)
    assert alloc["total_nodes"] == 5
    assert alloc["total_child_nfes"] == 4
    assert alloc["neg_direct_nfes"] == 1  # Node 2
    assert alloc["pos_direct_nfes"] == 3  # Nodes 1, 3, 4
    assert alloc["neg_direct_pct"] == 25.0
    assert alloc["pos_direct_pct"] == 75.0
    assert alloc["neg_expansions"] == 0   # Node 2 has 0 children (search stopped exploring it)
    assert alloc["pos_expansions"] == 1   # Node 1 was expanded (had 2 children)
    assert alloc["neg_downstream_nfes"] == 0  # 0 NFEs wasted below node 2
    assert alloc["pos_downstream_nfes"] == 2  # Nodes 3 and 4 were evaluated below node 1

    # Nested negative nodes test to ensure NO double counting:
    # Root -> Node 1 (neg) -> Node 2 (neg) -> Node 3 (pos)
    # Both 1 and 2 are negative. Descendants of 1 are {2, 3}. Descendants of 2 are {3}.
    # Without deduplication: 2 + 1 = 3 (even though total non-root nodes = 3).
    # With deduplication: unique descendants = {2, 3} -> exactly 2.
    nested_data = {
        "nodes": [
            {"node_id": 0, "parent_id": None, "depth": 0, "immediate_reward": None},
            {"node_id": 1, "parent_id": 0, "depth": 1, "immediate_reward": -0.4},
            {"node_id": 2, "parent_id": 1, "depth": 2, "immediate_reward": -0.2},
            {"node_id": 3, "parent_id": 2, "depth": 3, "immediate_reward": 0.5},
        ]
    }
    nested_alloc = compute_negative_ig_allocation(nested_data)
    assert nested_alloc["total_child_nfes"] == 3
    assert nested_alloc["neg_direct_nfes"] == 2  # Nodes 1 and 2
    assert nested_alloc["neg_downstream_nfes"] == 2  # Unique descendants {2, 3} - NOT double counted
    assert nested_alloc["neg_branch_total_nfes"] == 3  # All 3 nodes lie in negative branches
    assert nested_alloc["pos_branch_total_nfes"] == 0  # 0 pure positive nodes


