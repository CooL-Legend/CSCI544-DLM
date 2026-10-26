"""Offline extraction and analysis of candidate history and scores for MEDAL and AB-MEDAL.

Extracts distinct candidate states at qualifying depths (candidates >= C),
reconstructs candidate discovery progression over search iterations, and compiles
per-question and aggregate comparisons between MEDAL and AB-MEDAL.
"""

from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from dlm_project.decoding.mcts_decoding import select_candidate_depth
from dlm_project.experiments.sweep_report import wilson_interval


def _state_key(node_state: Union[List[int], Tuple[int, ...]]) -> Tuple[int, ...]:
    if isinstance(node_state, tuple):
        return node_state
    if isinstance(node_state, list):
        if node_state and isinstance(node_state[0], list):
            return tuple(node_state[0])
        return tuple(node_state)
    return tuple(node_state)


def _compute_state_hash(state: Union[List[int], Tuple[int, ...]]) -> str:
    key = _state_key(state)
    return hashlib.sha256(repr(key).encode()).hexdigest()


def extract_distinct_candidates(
    tree_data: Dict[str, Any],
    min_c: int = 3,
    selection_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract distinct candidates and depth statistics from saved tree data.

    Args:
        tree_data: Parsed `tree.json` dictionary containing a "nodes" list.
        min_c: Minimum number of distinct states required to qualify a depth (default: 3).
        selection_record: Optional parsed `selection.json` record with candidate/shortlist IDs.

    Returns:
        Dict containing:
        - candidate_depth (d*): Deepest depth with >= min_c distinct states (or fallback).
        - candidate_count: Number of distinct states at d*.
        - min_c: The threshold C used.
        - qualifying_depths: List of depths that have >= min_c distinct states.
        - distinct_candidates_by_depth: {depth: count} mapping for all depths.
        - max_depth_reached: Maximum depth reached in the tree.
        - candidates: List of candidate records at depth d*.
        - selected_node_id: ID of the chosen state (if known).
    """
    nodes = tree_data.get("nodes", [])
    if not nodes:
        return {
            "candidate_depth": 0,
            "candidate_count": 0,
            "min_c": min_c,
            "qualifying_depths": [],
            "distinct_candidates_by_depth": {},
            "max_depth_reached": 0,
            "candidates": [],
            "selected_node_id": None,
        }

    # Extract selection metadata if present
    selection = selection_record or tree_data.get("selection") or {}
    shortlist_ids = set(selection.get("shortlist_ids") or [])
    candidate_ids = set(selection.get("candidate_ids") or [])
    selected_node_id = selection.get("selected_node_id")

    # Group evaluated nodes by depth and deduplicate by state identity
    by_depth: Dict[int, Dict[Any, Dict[str, Any]]] = {}
    for node in sorted(nodes, key=lambda n: n.get("node_id", 0)):
        depth = node.get("depth", 0)
        s_hash = node.get("state_hash")
        if not s_hash and "state" in node:
            s_hash = _compute_state_hash(node["state"])
        key = s_hash or node.get("node_id")

        if depth not in by_depth:
            by_depth[depth] = {}
        if key not in by_depth[depth]:
            by_depth[depth][key] = node

    distinct_counts = {d: len(states) for d, states in sorted(by_depth.items())}
    max_depth = max(distinct_counts.keys(), default=0)

    # Determine candidate depth d*
    target_depth = select_candidate_depth(distinct_counts, min_c)
    candidates_dict = by_depth.get(target_depth, {})
    qualifying_depths = [d for d, count in distinct_counts.items() if count >= min_c]

    # Format detailed candidate state list
    candidate_list = []
    for key, node in candidates_dict.items():
        nid = node.get("node_id")
        action = node.get("action")
        mean_rew = node.get("mean_reward")
        if mean_rew is None:
            mean_rew = node.get("q_value", 0.0)

        candidate_list.append({
            "node_id": nid,
            "depth": target_depth,
            "state_hash": key if isinstance(key, str) else node.get("state_hash"),
            "visits": node.get("visits", 0),
            "mean_reward": mean_rew,
            "immediate_reward": node.get("immediate_reward"),
            "evaluation_order": node.get("evaluation_order"),
            "action": action,
            "is_shortlist": nid in shortlist_ids,
            "is_candidate": (nid in candidate_ids) or node.get("candidate", False),
            "is_selected": (nid == selected_node_id),
            "state": node.get("state"),
        })

    # Sort candidates by backed-up Q-value descending
    candidate_list.sort(key=lambda c: (-c["mean_reward"], c["node_id"]))

    return {
        "candidate_depth": target_depth,
        "candidate_count": len(candidate_list),
        "min_c": min_c,
        "qualifying_depths": qualifying_depths,
        "distinct_candidates_by_depth": distinct_counts,
        "max_depth_reached": max_depth,
        "candidates": candidate_list,
        "selected_node_id": selected_node_id,
    }


def extract_candidate_progression(tree_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reconstruct cumulative distinct candidates over simulation iterations t = 0..B-1.

    Args:
        tree_data: Parsed `tree.json` dictionary containing a "nodes" list.

    Returns:
        List of progression steps sorted by evaluation_order, each step containing:
        - evaluation_order: Simulation index t.
        - node_id: ID of evaluated node.
        - depth: Depth of evaluated node.
        - is_new_state: Whether this node represents a newly discovered distinct state.
        - cumulative_distinct_total: Total unique states discovered so far.
        - cumulative_distinct_by_depth: {depth: count} mapping at step t.
    """
    nodes = tree_data.get("nodes", [])
    if not nodes:
        return []

    sorted_nodes = sorted(nodes, key=lambda n: n.get("evaluation_order", n.get("node_id", 0)))
    seen_states = set()
    depth_counts: Dict[int, set] = defaultdict(set)
    progression = []

    for node in sorted_nodes:
        order = node.get("evaluation_order", 0)
        nid = node.get("node_id")
        depth = node.get("depth", 0)
        s_hash = node.get("state_hash")
        if not s_hash and "state" in node:
            s_hash = _compute_state_hash(node["state"])
        key = (depth, s_hash or nid)

        is_new = key not in seen_states
        seen_states.add(key)
        depth_counts[depth].add(s_hash or nid)

        progression.append({
            "evaluation_order": order,
            "node_id": nid,
            "depth": depth,
            "is_new_state": is_new,
            "cumulative_distinct_total": len(seen_states),
            "cumulative_distinct_by_depth": {d: len(s) for d, s in sorted(depth_counts.items())},
        })

    return progression


def compute_negative_ig_allocation(tree_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute NFE/forward compute allocation to negative IG states vs non-negative states.

    In the MCTS search tree:
    - Root (depth 0) has no incoming edge/IG and counts as 1 root analysis forward.
    - Each evaluated child node (parent_id is not None) corresponds to 1 forward evaluation
      with an immediate reward (information gain, IG).
    - An evaluated child can have negative IG (immediate_reward < 0) or non-negative IG (immediate_reward >= 0).
    - If a state has negative IG, does the search continue expanding below it?
      We calculate the subtrees rooted at negative IG nodes to determine the downstream
      forward compute (children NFEs) expended exploring negative paths.

    Returns:
        Dict containing:
        - total_nodes: Total evaluated nodes in the tree (B).
        - total_child_nfes: Total evaluated non-root nodes (B - 1).
        - neg_direct_nfes: Number of evaluated nodes with immediate_reward < 0.
        - pos_direct_nfes: Number of evaluated nodes with immediate_reward >= 0.
        - neg_direct_pct: Percentage of child NFEs with negative IG.
        - pos_direct_pct: Percentage of child NFEs with non-negative IG.
        - neg_expansions: Number of negative IG nodes that were subsequently expanded (had children).
        - pos_expansions: Number of non-negative IG nodes that were subsequently expanded.
        - neg_subtree_nfes: Total NFEs expended inside subtrees originating from negative IG states.
        - pos_subtree_nfes: Total NFEs expended inside subtrees originating from non-negative IG states.
        - neg_subtree_pct: Percentage of search compute allocated to negative IG branches.
        - neg_avg_visits: Mean MCTS visit count of negative IG states.
        - pos_avg_visits: Mean MCTS visit count of non-negative IG states.
        - neg_avg_q: Mean backed-up Q-value of negative IG states.
        - pos_avg_q: Mean backed-up Q-value of non-negative IG states.
    """
    nodes = tree_data.get("nodes", [])
    if not nodes:
        return {
            "total_nodes": 0, "total_child_nfes": 0,
            "neg_direct_nfes": 0, "pos_direct_nfes": 0,
            "neg_direct_pct": 0.0, "pos_direct_pct": 0.0,
            "neg_expansions": 0, "pos_expansions": 0,
            "neg_subtree_nfes": 0, "pos_subtree_nfes": 0, "neg_subtree_pct": 0.0,
            "neg_avg_visits": 0.0, "pos_avg_visits": 0.0,
            "neg_avg_q": 0.0, "pos_avg_q": 0.0,
        }

    children = [n for n in nodes if n.get("parent_id") is not None]
    total_child = len(children)
    if total_child == 0:
        return {
            "total_nodes": len(nodes), "total_child_nfes": 0,
            "neg_direct_nfes": 0, "pos_direct_nfes": 0,
            "neg_direct_pct": 0.0, "pos_direct_pct": 0.0,
            "neg_expansions": 0, "pos_expansions": 0,
            "neg_subtree_nfes": 0, "pos_subtree_nfes": 0, "neg_subtree_pct": 0.0,
            "neg_avg_visits": 0.0, "pos_avg_visits": 0.0,
            "neg_avg_q": 0.0, "pos_avg_q": 0.0,
        }

    node_by_id = {n["node_id"]: n for n in nodes}
    child_map: Dict[int, List[int]] = defaultdict(list)
    for c in children:
        child_map[c["parent_id"]].append(c["node_id"])

    neg_nodes = [c for c in children if c.get("immediate_reward") is not None and c["immediate_reward"] < 0]
    pos_nodes = [c for c in children if c.get("immediate_reward") is not None and c["immediate_reward"] >= 0]

    neg_direct = len(neg_nodes)
    pos_direct = len(pos_nodes)

    neg_expanded = sum(1 for n in neg_nodes if len(child_map.get(n["node_id"], [])) > 0)
    pos_expanded = sum(1 for n in pos_nodes if len(child_map.get(n["node_id"], [])) > 0)

    # Calculate unique downstream compute (descendants) spawned below negative states without double counting
    neg_descendant_ids = set()
    for n in neg_nodes:
        queue = list(child_map.get(n["node_id"], []))
        while queue:
            curr = queue.pop(0)
            if curr not in neg_descendant_ids:
                neg_descendant_ids.add(curr)
                queue.extend(child_map.get(curr, []))

    pos_descendant_ids = set()
    for n in pos_nodes:
        queue = list(child_map.get(n["node_id"], []))
        while queue:
            curr = queue.pop(0)
            if curr not in pos_descendant_ids:
                pos_descendant_ids.add(curr)
                queue.extend(child_map.get(curr, []))

    # Number of unique NFEs evaluated in branches below at least one negative IG node
    neg_downstream = len(neg_descendant_ids)
    pos_downstream = len(pos_descendant_ids)

    # Unique nodes impacted by negative IG (either negative IG themselves OR evaluated under a negative state)
    neg_impacted_ids = set(n["node_id"] for n in neg_nodes) | neg_descendant_ids
    pos_pure_ids = set(n["node_id"] for n in children) - neg_impacted_ids

    neg_visits = sum(n.get("visits", 0) for n in neg_nodes) / neg_direct if neg_direct else 0.0
    pos_visits = sum(n.get("visits", 0) for n in pos_nodes) / pos_direct if pos_direct else 0.0

    neg_q = sum(n.get("mean_reward", n.get("q_value", 0.0)) or 0.0 for n in neg_nodes) / neg_direct if neg_direct else 0.0
    pos_q = sum(n.get("mean_reward", n.get("q_value", 0.0)) or 0.0 for n in pos_nodes) / pos_direct if pos_direct else 0.0

    return {
        "total_nodes": len(nodes),
        "total_child_nfes": total_child,
        "neg_direct_nfes": neg_direct,
        "pos_direct_nfes": pos_direct,
        "neg_direct_pct": (neg_direct / total_child) * 100.0,
        "pos_direct_pct": (pos_direct / total_child) * 100.0,
        "neg_expansions": neg_expanded,
        "pos_expansions": pos_expanded,
        "neg_downstream_nfes": neg_downstream,
        "pos_downstream_nfes": pos_downstream,
        "neg_branch_total_nfes": len(neg_impacted_ids),
        "pos_branch_total_nfes": len(pos_pure_ids),
        "neg_avg_visits": neg_visits,
        "pos_avg_visits": pos_visits,
        "neg_avg_q": neg_q,
        "pos_avg_q": pos_q,
    }


def load_question_comparison(
    results_dir: Union[str, Path],
    stage: str = "dev",
    budget: int = 512,
    methods: Sequence[str] = ("medal", "ab_medal"),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compile per-question and aggregate comparison table between MEDAL and AB-MEDAL.

    Args:
        results_dir: Results root directory containing stage or method subdirectories.
        stage: Experiment stage ('dev', 'evaluation', 'test').
        budget: Checkpoint budget (e.g. 256, 512, 1024).
        methods: Methods to compare (default: ('medal', 'ab_medal')).

    Returns:
        Tuple of (per_question_rows, aggregate_summary):
        - per_question_rows: List of dicts comparing each question.
        - aggregate_summary: Dict with overall accuracy, NFEs, and Win/Tie/Loss statistics.
    """
    results_path = Path(results_dir)
    stage_path = results_path / stage if (results_path / stage).is_dir() else results_path

    # Discover all question IDs present under any method
    question_ids = set()
    for m in methods:
        method_dir = stage_path / m
        if method_dir.is_dir():
            for q_dir in method_dir.iterdir():
                if q_dir.is_dir() and (q_dir / "checkpoints" / str(budget) / "metrics.json").is_file():
                    question_ids.add(q_dir.name)

    sorted_q_ids = sorted(question_ids)
    per_question_rows = []

    for q_id in sorted_q_ids:
        row: Dict[str, Any] = {
            "example_id": q_id,
            "search_budget": budget,
        }

        records: Dict[str, Dict[str, Any]] = {}
        for m in methods:
            chk_dir = stage_path / m / q_id / "checkpoints" / str(budget)
            met_file = chk_dir / "metrics.json"
            if met_file.is_file():
                try:
                    records[m] = json.loads(met_file.read_text())
                except Exception:
                    pass

        # Populate baseline question information
        first_met = next(iter(records.values()), {})
        row["question"] = first_met.get("question", "")
        row["reference_solution"] = first_met.get("reference_solution", "")
        row["reference_answer"] = first_met.get("reference_answer", "")

        for m in methods:
            met = records.get(m, {})
            row[f"{m}_predicted"] = met.get("predicted_answer")
            row[f"{m}_exact_match"] = met.get("exact_match")
            row[f"{m}_score"] = 1.0 if met.get("exact_match") else (0.0 if met.get("exact_match") is False else None)
            row[f"{m}_selected_depth"] = met.get("selected_depth")
            row[f"{m}_selected_q"] = met.get("selected_q")
            row[f"{m}_total_nfes"] = met.get("total_nfes")
            row[f"{m}_search_nfes"] = met.get("search_nfes")
            row[f"{m}_completion_nfes"] = met.get("completion_nfes")
            row[f"{m}_total_wall_seconds"] = met.get("total_wall_seconds")

        # Head-to-head outcome category if both methods are present
        if "medal" in records and "ab_medal" in records:
            m_ok = bool(records["medal"].get("exact_match"))
            ab_ok = bool(records["ab_medal"].get("exact_match"))
            if m_ok and ab_ok:
                row["outcome"] = "Both Correct (Tie)"
            elif ab_ok and not m_ok:
                row["outcome"] = "AB-MEDAL Win"
            elif m_ok and not ab_ok:
                row["outcome"] = "MEDAL Win"
            else:
                row["outcome"] = "Both Incorrect (Tie)"
        else:
            row["outcome"] = "Incomplete"

        per_question_rows.append(row)

    # Compute aggregate metrics
    agg_by_method = {}
    for m in methods:
        valid_records = [r for r in per_question_rows if r.get(f"{m}_exact_match") is not None]
        n = len(valid_records)
        if n > 0:
            correct = sum(1 for r in valid_records if r.get(f"{m}_exact_match") is True)
            ci_low, ci_high = wilson_interval(correct, n)
            agg_by_method[m] = {
                "sample_count": n,
                "correct_count": correct,
                "accuracy": correct / n,
                "accuracy_percent": 100.0 * correct / n,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "mean_total_nfes": sum(r.get(f"{m}_total_nfes", 0) or 0 for r in valid_records) / n,
                "mean_selected_depth": sum(r.get(f"{m}_selected_depth", 0) or 0 for r in valid_records) / n,
                "mean_selected_q": sum(r.get(f"{m}_selected_q", 0) or 0 for r in valid_records) / n,
                "mean_wall_seconds": sum(r.get(f"{m}_total_wall_seconds", 0) or 0 for r in valid_records) / n,
            }

    # Win / Tie / Loss counts
    both_evaluated = [r for r in per_question_rows if r.get("medal_exact_match") is not None and r.get("ab_medal_exact_match") is not None]
    outcomes_count = Counter(r.get("outcome") for r in both_evaluated)

    aggregate_summary = {
        "stage": stage,
        "search_budget": budget,
        "total_questions": len(sorted_q_ids),
        "methods": agg_by_method,
        "head_to_head": {
            "compared_count": len(both_evaluated),
            "ab_medal_wins": outcomes_count.get("AB-MEDAL Win", 0),
            "medal_wins": outcomes_count.get("MEDAL Win", 0),
            "both_correct_ties": outcomes_count.get("Both Correct (Tie)", 0),
            "both_incorrect_ties": outcomes_count.get("Both Incorrect (Tie)", 0),
        },
    }

    return per_question_rows, aggregate_summary
