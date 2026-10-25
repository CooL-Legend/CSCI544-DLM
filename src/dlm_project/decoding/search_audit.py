"""Model-free reconstruction and compute-allocation metrics for MEDAL and AB.

Only standard-library imports: saved prompts and traces are sufficient offline.
State identities match mcts_decoding.candidate_value_record exactly.
"""

from collections import Counter
import copy
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

TRACE_SCHEMA_VERSION = "search_trace_v1"


def state_hash(ids):
    return hashlib.sha256(repr(tuple(ids)).encode()).hexdigest()


def _mean(values):
    return sum(values) / len(values) if values else None


def reconstruct(audit, prompt_ids=None):
    """Validate links/order/backups and, when available, exact state hashes."""
    rows = audit["nodes"]
    if not rows:
        return {}
    by_id, states, rewards = {}, {}, {}
    for order, row in enumerate(rows):
        node_id, parent_id = row["node_id"], row["parent_id"]
        if node_id in by_id or row["evaluation_order"] != order:
            raise ValueError("Duplicate node or non-contiguous evaluation order.")
        if order == 0:
            if node_id != 0 or parent_id is not None or row["depth"] != 0 or row["immediate_reward"] is not None:
                raise ValueError("Invalid analyzed root.")
            if prompt_ids is not None:
                states[0] = list(prompt_ids) + [audit["mask_id"]] * audit["gen_length"]
        else:
            if parent_id not in by_id or row["depth"] != by_id[parent_id]["depth"] + 1:
                raise ValueError("Invalid parent or evaluated-node depth.")
            reward = row["immediate_reward"]
            if reward is None or not math.isfinite(reward):
                raise ValueError("Evaluated children need finite immediate rewards.")
            action = row["action"]
            position, token = action["output_position"], action["token_id"]
            if not 0 <= position < audit["gen_length"] or token == audit["mask_id"]:
                raise ValueError("Invalid committed action.")
            if prompt_ids is not None:
                state = states[parent_id].copy()
                index = len(prompt_ids) + position
                if state[index] != audit["mask_id"]:
                    raise ValueError("Action overwrote a commitment.")
                state[index] = token
                states[node_id] = state
        if prompt_ids is not None and row.get("state_hash") is not None:
            if state_hash(states[node_id]) != row["state_hash"]:
                raise ValueError("Evaluated state hash mismatch.")
        by_id[node_id] = row
        rewards[node_id] = [0, 0.0]
        if parent_id is not None:
            current = node_id
            while current is not None:
                rewards[current][0] += 1
                rewards[current][1] += row["immediate_reward"]
                current = by_id[current]["parent_id"]
    for node_id, (visits, total) in rewards.items():
        row = by_id[node_id]
        q = total / visits if visits else 0.0
        if row["visits"] != visits or not math.isclose(row["reward_sum"], total, abs_tol=1e-12, rel_tol=1e-12) or not math.isclose(row["q_value"], q, abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError("Search statistics do not match reward backups.")
    selection = audit.get("selection")
    if selection and audit.get("stop_mode", "budget") == "budget":
        by_depth = {}
        for row in sorted(rows, key=lambda n: n["node_id"]):
            key = tuple(states[row["node_id"]]) if states else row.get("state_hash")
            if key is None:
                break
            by_depth.setdefault(row["depth"], {}).setdefault(key, row)
        else:
            limit = audit.get("shortlist_limit", 3)
            version = selection.get("candidate_selection_version")
            if version == "deepest_with_c_distinct_mean_q_v2":
                distinct_counts = {d: len(s) for d, s in by_depth.items()}
                qualified = [d for d, c in distinct_counts.items() if c >= limit]
                selected_depth = max(qualified) if qualified else max(distinct_counts, key=lambda d: (distinct_counts[d], d))
                distinct = by_depth[selected_depth]
            else:
                deepest = max(row["depth"] for row in rows)
                selected_depth = deepest
                distinct = by_depth.get(deepest, {})
            shortlist = sorted(distinct.values(), key=lambda r: (-r["q_value"], r["node_id"]))[:limit]
            if (selection["candidate_ids"] != [r["node_id"] for r in distinct.values()]
                    or selection["shortlist_ids"] != [r["node_id"] for r in shortlist]
                    or selection["selected_node_id"] != shortlist[0]["node_id"]
                    or selection["selected_depth"] != selected_depth
                    or selection["fallback"] != (selected_depth < audit["requested_depth"])):
                raise ValueError("Saved selection differs from empirical-Q ranking.")
    return states


def from_events(events, *, prompt_ids=None, mask_id, gen_length, requested_depth,
                requested_budget, search_nfes, adaptive=False, stop_mode="budget",
                shortlist_limit=3, allow_legacy=False):
    """Rebuild final node statistics without model/tokenizer access.

    New traces are strict. Historical traces can recover topology/rewards while
    missing state hashes remain null and are identified in availability metadata.
    """
    nodes = [copy.deepcopy(e) for e in events if e.get("stage") == "node_evaluated"]
    modern = bool(nodes)
    if not nodes and allow_legacy:
        root = next((e for e in events if e.get("stage") == "initialization"), None)
        if root is not None:
            nodes = [dict(node_id=0, parent_id=None, depth=0, evaluation_order=0,
                          state_hash=None, immediate_reward=None, action=None)]
            expansions = {e["child_id"]: e for e in events if e.get("stage") == "expansion"}
            for event in (e for e in events if e.get("stage") == "evaluation"):
                expanded = expansions[event["child_id"]]
                nodes.append(dict(node_id=event["child_id"], parent_id=expanded["parent_id"],
                                  depth=expanded["depth"], evaluation_order=len(nodes), state_hash=None,
                                  immediate_reward=event["reward"], action=expanded["action"]))
    nodes = [{key: row.get(key) for key in ("node_id", "parent_id", "depth", "evaluation_order",
              "state_hash", "immediate_reward", "action")} for row in nodes]
    by_id = {row["node_id"]: row for row in nodes}
    for row in nodes:
        row.update(visits=0, reward_sum=0.0, q_value=0.0)
    for row in nodes[1:]:
        current = row
        seen = set()
        while current is not None:
            if current["node_id"] in seen:
                raise ValueError("Cyclic parent links.")
            seen.add(current["node_id"])
            current["visits"] += 1
            current["reward_sum"] += row["immediate_reward"]
            parent = current["parent_id"]
            current = by_id[parent] if parent is not None else None
    for row in nodes:
        row["q_value"] = row["reward_sum"] / row["visits"] if row["visits"] else 0.0
    final = next((e for e in reversed(events) if e.get("stage") == "search_selection"), None)
    if final:
        for expected in final["node_statistics"]:
            actual = by_id[expected["node_id"]]
            if any(actual[key] != expected[key] for key in ("visits", "reward_sum", "q_value")):
                raise ValueError("Saved final statistics differ from reconstructed backups.")
    selection = None if final is None else {key: final[key] for key in (
        "candidate_ids", "shortlist_ids", "selected_node_id", "selected_depth", "stop_reason", "fallback",
        "candidate_selection_version") if key in final}
    root_event = next((e for e in events if e.get("stage") == "initialization"), {})
    audit = dict(trace_schema_version=TRACE_SCHEMA_VERSION if modern else "legacy_recovered",
                 mask_id=mask_id, gen_length=gen_length, requested_depth=requested_depth,
                 requested_budget=requested_budget, stop_mode=stop_mode, shortlist_limit=shortlist_limit,
                 search_nfes=search_nfes, nodes=nodes, selection=selection,
                 root_action_count=final["root_action_count"] if final else len(root_event.get("actions", [])),
                 decisions=[copy.deepcopy(e) for e in events if e.get("stage") == "adaptive_decision"] if adaptive else None,
                 forwards=[copy.deepcopy(e) for e in events if e.get("stage") == "forward" and e.get("phase") in ("tree_analysis", "simulation", "candidate_evaluation")],
                 availability={"explicit_state_hashes": modern, "explicit_evaluation_order": modern,
                               "final_selection": final is not None, "adaptive_decisions": adaptive})
    reconstruct(audit, prompt_ids)
    if final is not None and len(nodes) != search_nfes:
        raise ValueError("Successful search NFEs differ from evaluated root plus children.")
    return audit


def metrics(audit):
    """Calculate common allocation/structure/reward metrics from compact records."""
    reconstruct(audit)
    nodes, selection = audit["nodes"], audit["selection"]
    by_id = {row["node_id"]: row for row in nodes}
    children = [row for row in nodes if row["parent_id"] is not None]
    branching = Counter(row["parent_id"] for row in children)
    root_child, paths = {}, {0: []}
    for row in children:
        node_id, parent_id = row["node_id"], row["parent_id"]
        root_child[node_id] = node_id if parent_id == 0 else root_child[parent_id]
        action = row["action"]
        paths[node_id] = paths[parent_id] + [(action["output_position"], action["token_id"])]
    depths = Counter(row["depth"] for row in nodes)
    subtrees = Counter(root_child[row["node_id"]] for row in children)
    hashes_available = bool(nodes) and all(row["state_hash"] is not None for row in nodes)
    # Actions also reconstruct exact output identities without prompt tokens.
    identities = {tuple(sorted(paths[row["node_id"]])) for row in nodes}
    unique = len(identities)
    duplicates = len(nodes) - unique
    rewards = [row["immediate_reward"] for row in children]
    negative = sum(r < 0 for r in rewards)
    deepest = max(depths, default=0)
    target = {tuple(sorted(paths[r["node_id"]])) for r in nodes if r["depth"] == audit["requested_depth"]}
    candidate_depth = selection["selected_depth"] if selection else deepest
    candidate_states = {tuple(sorted(paths[r["node_id"]])) for r in nodes if r["depth"] == candidate_depth}
    candidate_representatives = {}
    for row in sorted(nodes, key=lambda n: n['node_id']):
        if row['depth'] == candidate_depth:
            candidate_representatives.setdefault(tuple(sorted(paths[row['node_id']])), row['node_id'])
    decisions = audit["decisions"]
    counts = Counter(d["decision_type"] for d in decisions) if decisions is not None else None
    forced = Counter(d["decision_type"] for d in decisions if d["forced"]) if decisions is not None else None
    total_decisions = len(decisions) if decisions is not None else None
    shortlist = selection["shortlist_ids"] if selection else []
    pairs = []
    for a, b in combinations(shortlist, 2):
        left, right = paths[a], paths[b]
        prefix = 0
        for x, y in zip(left, right):
            if x != y:
                break
            prefix += 1
        union = set(left) | set(right)
        pairs.append(dict(candidate_ids=[a, b], shared_prefix_length=prefix,
                          jaccard_distance=1 - len(set(left) & set(right)) / len(union) if union else 0.0))
    budget, nfe = audit["requested_budget"], audit["search_nfes"]
    result = dict(
        max_depth=deepest, selected_depth=selection["selected_depth"] if selection else None,
        average_search_depth=_mean([r["depth"] for r in children]) or 0.0,
        evaluated_children=len(children), expanded_parents=len(branching),
        branching_factor_min=min(branching.values(), default=0),
        branching_factor_mean=_mean(list(branching.values())) or 0.0,
        branching_factor_max=max(branching.values(), default=0),
        candidate_count=len(candidate_states), target_depth_candidate_count=len(target), shortlist_size=len(shortlist),
        unique_evaluated_states=unique, duplicate_state_count=duplicates,
        duplicate_state_percentage=100 * duplicates / len(children) if children else None,
        evaluated_root_subtrees=len(subtrees), retained_root_actions=audit["root_action_count"],
        root_subtree_coverage=len(subtrees) / audit["root_action_count"] if audit["root_action_count"] else None,
        deepest_candidate_root_subtrees=sorted({root_child[i] for i in candidate_representatives.values() if i != 0}),
        shortlist_root_subtrees=sorted({root_child[i] for i in shortlist if i != 0}),
        negative_ig_count=negative, negative_ig_percentage=100 * negative / len(rewards) if rewards else None,
        ig_min=min(rewards) if rewards else None, ig_mean=_mean(rewards), ig_max=max(rewards) if rewards else None,
        selected_q=by_id[selection["selected_node_id"]]["q_value"] if selection else None,
        nfes_per_depth={str(k): v for k, v in sorted(depths.items())},
        nfes_per_root_subtree={str(k): v for k, v in sorted(subtrees.items())},
        root_analysis_nfes=depths.get(0, 0), uncompleted_evaluation_nfes=nfe - len(nodes),
        requested_depth_success=bool(target), fallback=selection["fallback"] if selection else None,
        fallback_rate=float(selection["fallback"]) if selection else None,
        stop_reason=selection["stop_reason"] if selection else None,
        budget_utilization=nfe / budget if budget else None, search_nfes=nfe,
        decision_counts={k: counts[k] for k in ("GEN", "CONT")} if counts is not None else None,
        decision_percentages={k: 100 * counts[k] / total_decisions if total_decisions else None for k in ("GEN", "CONT")} if counts is not None else None,
        forced_decision_counts={k: forced[k] for k in ("GEN", "CONT")} if forced is not None else None,
        forced_decision_percentages={k: 100 * forced[k] / total_decisions if total_decisions else None
                                     for k in ("GEN", "CONT")} if forced is not None else None,
        forced_decisions=sum(forced.values()) if forced is not None else None,
        forced_decision_percentage=100 * sum(forced.values()) / total_decisions if total_decisions else None,
        candidate_diversity=dict(pairs=pairs, mean_jaccard_distance=_mean([p["jaccard_distance"] for p in pairs]),
                                shared_prefix_length=min((p["shared_prefix_length"] for p in pairs), default=None)),
        state_hashes_available=hashes_available,
    )
    if selection and (sum(depths.values()) != nfe or depths.get(0, 0) + sum(subtrees.values()) != nfe):
        raise ValueError("Depth/subtree NFE allocation does not reconcile.")
    return result


def load(directory):
    """Reconstruct a development artifact without importing torch or transformers."""
    directory = Path(directory)
    audit = json.loads((directory / "search_audit.json").read_text())
    prompt = json.loads((directory / "prompt.json").read_text())
    reconstruct(audit, prompt["prompt_ids"])
    return metrics(audit)
