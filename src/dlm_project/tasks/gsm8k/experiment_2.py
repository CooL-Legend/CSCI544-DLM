"""Unified runner for Experiment 2 (E2: Two-Step Lookahead FIG Reward).

Compares MEDAL vs. AB-MEDAL under a two-step lookahead Fractional Information Gain (FIG)
reward function across matched search budgets (e.g. 256, 512).
Supports cooperative multi-question batching, verified checkpointing, completion caching,
and flexible dataset evaluation across the canonical 20 samples, extended +50 samples, or combined 70.
"""

import argparse
from collections import defaultdict
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from dlm_project.decoding import mcts_decoding as medal
from dlm_project.decoding.ab_batch import pack_states, sync
from dlm_project.decoding.ab_medal import ABMEDALConfig
from dlm_project.decoding.common import commit_from_logits
from dlm_project.decoding.e2_search import E2ABMEDALSearch, E2MEDALSearch, compute_fig_reward
from dlm_project.decoding.mcts_decoding import MCTSConfig, select_deepest_candidates
from dlm_project.decoding.medal_demo import DecodingResult, run_completion
from dlm_project.decoding import search_audit
from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.manifests import digest
from dlm_project.experiments.paths import create_run_directory
from dlm_project.experiments.sweep_report import wilson_interval, write_csv
from dlm_project.tasks.gsm8k.e0_tokens import decode_output, resolve_termination_tokens
from dlm_project.tasks.gsm8k.evaluation import score_answer
from dlm_project.tasks.gsm8k.experiment_protocol import (
    E2_EVALUATION_COUNT,
    E2_STAGE_COUNTS,
    FIXED_DEPTH,
    STAGE_COUNTS,
    build_prompt,
    example_seed,
    get_e2_manifest,
    get_manifest,
    load_shared_protocol,
    select_examples,
)

PROTOCOL_VERSION = "e2_two_step_lookahead_v1"
CHECKPOINT_BUDGETS = (256, 512)
MAX_SEARCH_DEPTH = 10
MIN_CANDIDATES = 3


def state_hash(state_tensor):
    return hashlib.sha256(repr(medal._state_key(state_tensor)).encode()).hexdigest()


def context(
    *,
    stage="dev",
    methods=("medal", "ab_medal"),
    budgets=None,
    max_depth=MAX_SEARCH_DEPTH,
    num_candidates=MIN_CANDIDATES,
    num_samples=None,
    sample_slice: Optional[Tuple[int, int]] = None,
):
    """Create and validate execution context for E2."""
    common = load_shared_protocol()
    if budgets is None:
        budgets = CHECKPOINT_BUDGETS

    manifest = get_e2_manifest(common=common)
    norm_stage = "evaluation" if stage in ("eval", "evaluation") else stage
    examples = select_examples(manifest, norm_stage, expected_count=E2_STAGE_COUNTS.get(norm_stage, STAGE_COUNTS[norm_stage]))

    if sample_slice is not None:
        start, end = sample_slice
        examples = examples[start:end]
    elif num_samples is not None:
        examples = examples[:num_samples]

    if not examples:
        raise ValueError(f"No examples found for stage {stage} (slice={sample_slice}, num_samples={num_samples}).")

    for m in methods:
        if m not in ("medal", "ab_medal"):
            raise ValueError(f"Unknown method {m}. Must be 'medal' or 'ab_medal'.")

    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": stage,
        "norm_stage": norm_stage,
        "methods": list(methods),
        "budgets": list(budgets),
        "max_depth": max_depth,
        "num_candidates": num_candidates,
        "common": common,
        "manifest": manifest["dataset"],
        "manifest_hash": digest(manifest),
        "prompt_template_hash": digest(common["prompt"]["template"]),
        "examples": examples,
        "setup": {},
    }


def prepare(wrapper, ctx):
    """Tokenize and format prompts for all examples in the stage."""
    termination = resolve_termination_tokens(wrapper.tokenizer, wrapper.model, wrapper.mask_id)
    pad = getattr(wrapper.tokenizer, "pad_token_id", None) or termination["eos_token_id"]
    prepared = []
    for example in ctx["examples"]:
        prompt = build_prompt(example["question"], ctx["common"]["prompt"])
        ids = wrapper.encode_prompt(prompt)
        prepared.append({
            "example": example,
            "text": prompt,
            "ids": ids,
            "prompt_ids": ids[0].cpu().tolist(),
            "prompt_hash": digest(prompt),
            "prompt_ids_hash": digest(ids.cpu().tolist()),
            "seed": example_seed(example["row_id"]),
            "prompt_template_hash": ctx["prompt_template_hash"],
        })
    ctx.update(termination=termination, pad_token_id=pad, model=wrapper.identity)
    return prepared


def load_model(ctx=None, model_path=None, device="cuda", local_files_only=False, **kwargs):
    """Load LLaDA wrapper for E2 experiments."""
    from dlm_project.models.llada import LLaDA
    dtype = ctx["common"]["model"]["dtype"] if ctx and "common" in ctx else "bfloat16"
    return LLaDA(device=device, model_path=model_path, dtype=dtype, local_files_only=local_files_only)


@torch.inference_mode()
def execute_search_batch_checkpointed_e2(
    wrapper,
    prepared_group,
    ctx,
    method: str,
    budgets: Tuple[int, ...] = CHECKPOINT_BUDGETS,
    max_depth: int = MAX_SEARCH_DEPTH,
    num_candidates: int = MIN_CANDIDATES,
):
    """Execute cooperative batched 2-step lookahead search up to max(budgets)."""
    max_budget = max(budgets)
    cfg_kwargs = dict(
        init_tokens=max_depth,
        num_candidates=num_candidates,
        top_k_tokens=ctx["common"]["decoding"]["mcts"]["top_k_tokens"],
        top_k_actions=ctx["common"]["decoding"]["mcts"]["top_k_actions"],
        stop_mode="budget",
        max_model_calls=max_budget,
    )

    if method == "ab_medal":
        search_cfg = ABMEDALConfig(**cfg_kwargs)
        search_cls = E2ABMEDALSearch
    else:
        search_cfg = MCTSConfig(**cfg_kwargs)
        search_cls = E2MEDALSearch

    trees = [
        search_cls(
            wrapper.model,
            item["ids"],
            wrapper.mask_id,
            ctx["common"]["generation"]["gen_length"],
            config=search_cfg,
            seed=item["seed"],
            tokenizer=wrapper.tokenizer,
            search_excluded_token_ids=ctx["termination"]["token_ids"],
            checkpoint_budgets=tuple(sorted(budgets)),
        )
        for item in prepared_group
    ]

    device = medal.model_device(wrapper.model)
    pad_token_id = ctx["pad_token_id"]
    sync(wrapper.model)
    t0 = perf_counter()
    search_latencies = [None] * len(trees)
    invocation_id = 0

    while True:
        active, requests = [], []
        for index, tree in enumerate(trees):
            request = tree.next_evaluation()
            if request is not None:
                active.append(index)
                requests.append(request)
            elif search_latencies[index] is None:
                sync(wrapper.model)
                search_latencies[index] = perf_counter() - t0
        if not active:
            break
        invocation_id += 1
        current_states = [r.node.state for r in requests]
        ids, attention, _ = pack_states(current_states, pad_token_id, device)
        for index, request in zip(active, requests):
            trees[index].record_external_forward(
                request, invocation_id=invocation_id, batch_size=len(active)
            )
        prediction = wrapper.model(ids) if len(active) == 1 else wrapper.model(ids, attention_mask=attention)
        for row, (index, request) in enumerate(zip(active, requests)):
            trees[index].submit_evaluation(
                request, logits=prediction.logits[row, :request.node.state.shape[1]]
            )
            if trees[index].done and search_latencies[index] is None:
                sync(wrapper.model)
                search_latencies[index] = perf_counter() - t0
        del prediction

    sync(wrapper.model)
    total_search_time = perf_counter() - t0
    for index, tree in enumerate(trees):
        if search_latencies[index] is None:
            search_latencies[index] = total_search_time
        for b in budgets:
            if b not in tree.checkpoints:
                tree.checkpoints[b] = tree._snapshot_checkpoint(b, stop_reason=tree.stop_reason)

    return list(zip(trees, search_latencies))


@torch.inference_mode()
def complete_selected_states_batch(wrapper, completion_requests, ctx, completion_cache=None):
    """Complete a batch of (prepared_item, selected_state, budget) requests with caching."""
    results_map = {}
    unique_uncached = []
    seen_in_batch = {}

    for item, selected_state, budget in completion_requests:
        ex_id = item["example"]["example_id"]
        s_hash = state_hash(selected_state)
        cache_key = (ex_id, s_hash)
        req_key = (ex_id, s_hash, budget)
        if completion_cache is not None and cache_key in completion_cache:
            cached = completion_cache[cache_key]
            results_map[req_key] = (cached["completion"], cached["decoded"], cached["score"], 0.0, True)
        elif cache_key in seen_in_batch:
            seen_in_batch[cache_key]["duplicate_req_keys"].append(req_key)
        else:
            entry = {
                "cache_key": cache_key,
                "primary_req_key": req_key,
                "duplicate_req_keys": [],
                "item": item,
                "prompt_len": item["ids"].shape[1],
                "state": selected_state.detach().to(medal.model_device(wrapper.model)).clone(),
                "calls": 0,
                "trace": [],
                "comp_time": None,
            }
            seen_in_batch[cache_key] = entry
            unique_uncached.append(entry)

    if unique_uncached:
        device = medal.model_device(wrapper.model)
        pad_token_id = ctx["pad_token_id"]
        mask_id = wrapper.mask_id
        sync(wrapper.model)
        t0 = perf_counter()
        invocation_id = 0

        while True:
            active = [
                i for i, entry in enumerate(unique_uncached)
                if (entry["state"][0, entry["prompt_len"]:] == mask_id).any()
            ]
            if not active:
                break
            invocation_id += 1
            current_states = [unique_uncached[i]["state"] for i in active]
            ids, attention, _ = pack_states(current_states, pad_token_id, device)
            prediction = wrapper.model(ids) if len(active) == 1 else wrapper.model(ids, attention_mask=attention)

            for row, index in enumerate(active):
                entry = unique_uncached[index]
                state = entry["state"]
                prompt_len = entry["prompt_len"]
                entry["calls"] += 1
                logits = prediction.logits[row, :state.shape[1]]
                positions = torch.nonzero(state[0, prompt_len:] == mask_id, as_tuple=False).flatten() + prompt_len
                actions, signals = commit_from_logits(
                    state, logits, prompt_len, mask_id, positions, 1,
                    selection_rule="token_probability", temperature=0.0
                )
                entry["trace"].append({
                    "stage": "commit", "block": 0, "step": entry["calls"] - 1,
                    "model_calls": entry["calls"], "actions": actions, **signals,
                    "remaining_masks": int((state[0, prompt_len:] == mask_id).sum().item()),
                })
                if not (state[0, prompt_len:] == mask_id).any() and entry["comp_time"] is None:
                    sync(wrapper.model)
                    entry["comp_time"] = perf_counter() - t0

            del prediction

        sync(wrapper.model)
        total_comp_time = perf_counter() - t0

        for entry in unique_uncached:
            comp_time = entry["comp_time"] if entry["comp_time"] is not None else total_comp_time
            completion = medal.CompletionResult(entry["state"].detach().cpu(), entry["calls"], entry["trace"])
            decoded = decode_output(
                wrapper.tokenizer,
                completion.token_ids[0, entry["prompt_len"]:].tolist(),
                ctx["termination"]["token_ids"],
            )
            score = score_answer(decoded["scoring_text"], entry["item"]["example"]["reference_solution"])

            record = {"completion": completion, "decoded": decoded, "score": score}
            if completion_cache is not None:
                completion_cache[entry["cache_key"]] = record

            out_tuple = (completion, decoded, score, comp_time, False)
            results_map[entry["primary_req_key"]] = out_tuple
            for dup_k in entry["duplicate_req_keys"]:
                results_map[dup_k] = (completion, decoded, score, 0.0, True)

    return results_map


def _jsonl(path: Path, rows: Sequence[Dict[str, Any]]):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def export_question_e2_artifacts(
    wrapper,
    search,
    search_time: float,
    prepared_item,
    ctx,
    method: str,
    base_dir: Path,
    budgets: Tuple[int, ...] = CHECKPOINT_BUDGETS,
    max_depth: int = MAX_SEARCH_DEPTH,
    num_candidates: int = MIN_CANDIDATES,
    completion_cache=None,
    precomputed_completions=None,
) -> List[Dict[str, Any]]:
    """Export traces, checkpoint trees, completions, and metrics for one completed question."""
    ex_id = prepared_item["example"]["example_id"]
    q_dir = base_dir / method / ex_id
    q_dir.mkdir(parents=True, exist_ok=True)

    _jsonl(q_dir / "trace.jsonl", search.trace)

    evaluated_nodes = [n for n in search.nodes if n.evaluation_order is not None]
    node_rows = []
    for n in evaluated_nodes:
        node_rows.append({
            "node_id": n.node_id,
            "parent_id": n.parent.node_id if n.parent else None,
            "depth": n.depth,
            "visits": n.visits,
            "reward_sum": n.reward_sum,
            "mean_reward": n.mean_reward,
            "immediate_reward": n.immediate_reward,
            "state_hash": medal.candidate_value_record(n)["state_hash"],
            "state": n.state.tolist()[0],
            "action": None if n.action is None else asdict(n.action),
            "child_ids": [c.node_id for c in n.children],
            "evaluation_order": n.evaluation_order,
        })
    _jsonl(q_dir / "nodes.jsonl", node_rows)

    if hasattr(search, "decision_history"):
        dec_rows = [
            {"node_id": nid, **dec}
            for nid, history in search.decision_history.items()
            for dec in history
        ]
        _jsonl(q_dir / "decisions.jsonl", dec_rows)
    else:
        _jsonl(q_dir / "decisions.jsonl", [])

    prompt_len = prepared_item["ids"].shape[1]
    gen_len = ctx["common"]["generation"]["gen_length"]
    audit = search_audit.from_events(
        search.trace,
        prompt_ids=prepared_item["prompt_ids"],
        mask_id=wrapper.mask_id,
        gen_length=gen_len,
        requested_depth=max_depth,
        requested_budget=max(budgets),
        search_nfes=search.model_calls,
        adaptive=(method == "ab_medal"),
        shortlist_limit=num_candidates,
        allow_legacy=True,
    )
    save_json(q_dir / "search_audit.json", audit)

    checkpoint_results = []
    for budget in sorted(budgets):
        chk = search.checkpoints[budget]
        chk_dir = q_dir / "checkpoints" / str(budget)
        chk_dir.mkdir(parents=True, exist_ok=True)

        selected_state = chk["selected"].state
        s_hash = state_hash(selected_state)
        req_key = (ex_id, s_hash, budget)

        if precomputed_completions is not None and req_key in precomputed_completions:
            completion, decoded, score, comp_time, was_cached = precomputed_completions[req_key]
        else:
            completion, decoded, score, comp_time, was_cached = complete_selected_states_batch(
                wrapper, [(prepared_item, selected_state, budget)], ctx, completion_cache=completion_cache
            )[req_key]

        comp_nfes = completion.model_calls
        total_nfes = budget + comp_nfes

        selection_record = chk["selection"]
        chk_metrics = {
            "search_budget": budget,
            "search_nfes": budget,
            "completion_nfes": comp_nfes,
            "total_nfes": total_nfes,
            "selected_node_id": chk["selected_node_id"],
            "selected_depth": chk["selected_depth"],
            "selected_q": chk["selected_q"],
            "max_depth_reached": chk["max_depth_reached"],
            "selection_depth_gap": chk["selection_depth_gap"],
            "distinct_candidates_by_depth": chk["distinct_candidates_by_depth"],
        }
        save_json(chk_dir / "selection.json", selection_record)
        save_json(chk_dir / "completion.json", {
            **decoded, "score": score, "token_ids": completion.token_ids.tolist(),
            "completion_nfes": comp_nfes, "completion_wall_seconds": comp_time,
            "was_cached": was_cached
        })
        save_json(chk_dir / "metrics.json", chk_metrics)

        tree_nodes = []
        for n in chk["nodes"]:
            tree_nodes.append({
                "node_id": n.node_id,
                "parent_id": n.parent.node_id if n.parent else None,
                "depth": n.depth,
                "visits": n.visits,
                "reward_sum": n.reward_sum,
                "mean_reward": n.mean_reward,
                "immediate_reward": n.immediate_reward,
                "candidate": n.node_id in chk["selection"]["candidate_ids"],
                "state_hash": medal.candidate_value_record(n)["state_hash"],
                "state": n.state.tolist()[0],
                "action": None if n.action is None else asdict(n.action),
                "child_ids": [c.node_id for c in n.children],
                "evaluation_order": n.evaluation_order,
            })

        save_json(chk_dir / "tree.json", {
            "method": method,
            "budget": budget,
            "example_id": prepared_item["example"]["example_id"],
            "seed": prepared_item["seed"],
            "nodes": tree_nodes,
            "statistics": chk_metrics,
        })

        if "posteriors" in chk and chk["posteriors"]:
            save_json(chk_dir / "posteriors.json", {
                str(node_id): {k: asdict(p) if hasattr(p, "__dataclass_fields__") else p for k, p in models.items()}
                for node_id, models in chk["posteriors"].items()
            })

        checkpoint_row = {
            "method": method,
            "search_budget": budget,
            "example_id": prepared_item["example"]["example_id"],
            "row_id": prepared_item["example"]["row_id"],
            "question": prepared_item["example"]["question"],
            "reference_solution": prepared_item["example"]["reference_solution"],
            "scoring_text": decoded["scoring_text"],
            "predicted_answer": score["predicted_answer"],
            "reference_answer": score["reference_answer"],
            "exact_match": bool(score["exact_match"]),
            "parse_success": bool(score["parse_success"]),
            "search_nfes": budget,
            "completion_nfes": comp_nfes,
            "total_nfes": total_nfes,
            "search_wall_seconds": search_time * (budget / max(budgets)),
            "completion_wall_seconds": comp_time,
            "total_wall_seconds": (search_time * (budget / max(budgets))) + comp_time,
            "max_depth_reached": chk["max_depth_reached"],
            "selected_depth": chk["selected_depth"],
            "selection_depth_gap": chk["selection_depth_gap"],
            "selected_depth_candidate_count": chk["selected_depth_candidate_count"],
            "distinct_candidates_by_depth": chk["distinct_candidates_by_depth"],
            "shortlist_node_ids": chk["shortlist_node_ids"],
            "selected_node_id": chk["selected_node_id"],
            "selected_q": chk["selected_q"],
            "selected_state_hash": chk["selected_state_hash"],
            "reached_target_depth": chk["reached_target_depth"],
            "selection_below_target": chk["selection_below_target"],
            "fallback_used": chk["fallback_used"],
            "stop_reason": chk["stop_reason"],
            "was_cached": was_cached,
            "artifact_directory": str(chk_dir),
        }
        save_json(chk_dir / "metrics.json", checkpoint_row)
        checkpoint_results.append(checkpoint_row)

        match_sym = "✓" if score["exact_match"] else "✗"
        total_time = (search_time * (budget / max(budgets))) + comp_time
        print(
            f"  [{match_sym} Budget {budget:4d}] Q: {prepared_item['example']['example_id']:<10} | "
            f"Model: {method.upper():8s} | Pred: {score['predicted_answer']!r:<10} | "
            f"Ref: {score['reference_answer']!r:<10} | Match: {str(score['exact_match']):<5} | Time: {total_time:.1f}s",
            flush=True,
        )

    return checkpoint_results


def run_questions_batch_e2(
    wrapper,
    prepared_group,
    ctx,
    method: str,
    base_dir: Path,
    budgets: Tuple[int, ...] = CHECKPOINT_BUDGETS,
    max_depth: int = MAX_SEARCH_DEPTH,
    num_candidates: int = MIN_CANDIDATES,
    completion_cache=None,
) -> List[Dict[str, Any]]:
    """Execute cooperative batched search across group, batch completion, and export artifacts."""
    search_pairs = execute_search_batch_checkpointed_e2(
        wrapper, prepared_group, ctx, method,
        budgets=budgets, max_depth=max_depth, num_candidates=num_candidates
    )

    completion_requests = []
    for budget in sorted(budgets):
        for item, (search, _) in zip(prepared_group, search_pairs):
            selected_state = search.checkpoints[budget]["selected"].state
            completion_requests.append((item, selected_state, budget))

    precomputed_completions = complete_selected_states_batch(
        wrapper, completion_requests, ctx, completion_cache=completion_cache
    )

    batch_rows = []
    for item, (search, search_time) in zip(prepared_group, search_pairs):
        rows = export_question_e2_artifacts(
            wrapper, search, search_time, item, ctx, method, base_dir,
            budgets=budgets, max_depth=max_depth, num_candidates=num_candidates,
            completion_cache=completion_cache,
            precomputed_completions=precomputed_completions,
        )
        batch_rows.extend(rows)

    return batch_rows


def is_question_complete(stage_dir: Path, method: str, ex_id: str, budgets: Sequence[int]) -> bool:
    q_dir = stage_dir / method / ex_id
    if not (q_dir / "trace.jsonl").is_file():
        return False
    for b in budgets:
        met_file = q_dir / "checkpoints" / str(b) / "metrics.json"
        if not met_file.is_file():
            return False
        try:
            row = json.loads(met_file.read_text())
            if not isinstance(row, dict) or "exact_match" not in row or "predicted_answer" not in row:
                return False
        except Exception:
            return False
    return True


def load_completed_question_rows(stage_dir: Path, method: str, ex_id: str, budgets: Sequence[int]) -> List[Dict[str, Any]]:
    rows = []
    q_dir = stage_dir / method / ex_id
    for b in sorted(budgets):
        met_file = q_dir / "checkpoints" / str(b) / "metrics.json"
        rows.append(json.loads(met_file.read_text()))
    return rows


def compute_e2_aggregate_rows(all_checkpoint_rows: List[Dict[str, Any]], stage: str,
                              methods: Sequence[str], budgets: Sequence[int]) -> List[Dict[str, Any]]:
    agg_rows = []
    for budget in sorted(budgets):
        for method in methods:
            subset = [r for r in all_checkpoint_rows if r["method"] == method and r["search_budget"] == budget]
            n = len(subset)
            if not n:
                continue
            correct_cnt = sum(bool(r["exact_match"]) for r in subset)
            parsed_cnt = sum(bool(r["parse_success"]) for r in subset)
            ci_low, ci_high = wilson_interval(correct_cnt, n)

            agg_rows.append({
                "stage": stage,
                "method": method,
                "search_budget": budget,
                "sample_count": n,
                "accuracy": correct_cnt / n,
                "accuracy_percent": 100.0 * correct_cnt / n,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "parse_rate": parsed_cnt / n,
                "mean_search_nfe": sum(r["search_nfes"] for r in subset) / n,
                "mean_completion_nfe": sum(r["completion_nfes"] for r in subset) / n,
                "mean_total_nfe": sum(r["total_nfes"] for r in subset) / n,
                "mean_search_wall_seconds": sum(r["search_wall_seconds"] for r in subset) / n,
                "mean_completion_wall_seconds": sum(r["completion_wall_seconds"] for r in subset) / n,
                "mean_total_wall_seconds": sum(r["total_wall_seconds"] for r in subset) / n,
                "mean_selected_depth": sum(r["selected_depth"] for r in subset) / n,
                "mean_selected_q": sum(r["selected_q"] for r in subset) / n,
                "fallback_rate": sum(r["fallback_used"] for r in subset) / n,
            })
    return agg_rows


def save_e2_progress(output_dir: Path, summaries_dir: Path, stage: str,
                     all_checkpoint_rows: List[Dict[str, Any]],
                     methods: Sequence[str], budgets: Sequence[int],
                     manifest_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    summaries_dir.mkdir(parents=True, exist_ok=True)
    agg_rows = []
    if all_checkpoint_rows:
        per_q_path = summaries_dir / f"e2_{stage}_per_question.csv"
        write_csv(per_q_path, all_checkpoint_rows)
        write_csv(summaries_dir / "e2_per_question.csv", all_checkpoint_rows)

        agg_rows = compute_e2_aggregate_rows(all_checkpoint_rows, stage, methods, budgets)
        agg_path = summaries_dir / f"e2_{stage}_aggregate.csv"
        write_csv(agg_path, agg_rows)
        write_csv(summaries_dir / "e2_aggregate.csv", agg_rows)

    if manifest_data is not None:
        save_json(output_dir / "run_manifest.json", manifest_data)
        save_json(summaries_dir / f"run_manifest_{stage}.json", manifest_data)

    try:
        os.sync()
    except (AttributeError, OSError):
        pass

    return agg_rows


def run_e2_stage(
    wrapper,
    ctx,
    output_dir: Path,
    methods=("medal", "ab_medal"),
    budgets=CHECKPOINT_BUDGETS,
    max_depth=MAX_SEARCH_DEPTH,
    num_candidates=MIN_CANDIDATES,
    resume: bool = True,
    batch_size: int = 1,
):
    """Execute complete E2 stage with cooperative batching and checkpoint resumption."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    output_dir = Path(output_dir)
    stage_dir = output_dir / ctx["stage"]
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare(wrapper, ctx)
    all_checkpoint_rows = []
    completion_cache = {}

    manifest_data = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": ctx["stage"],
        "methods": list(methods),
        "budgets": list(budgets),
        "max_depth": max_depth,
        "num_candidates": num_candidates,
        "batch_size": batch_size,
        "sample_count": len(prepared),
        "examples": [p["example"]["example_id"] for p in prepared],
        "setup": ctx.get("setup", {}),
        "model": ctx.get("model", {}),
        "status": "in_progress",
    }
    save_json(output_dir / "run_manifest.json", manifest_data)
    save_json(summaries_dir / f"run_manifest_{ctx['stage']}.json", manifest_data)

    total_q = len(prepared)
    rows_by_ex_method: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for start_idx in range(0, total_q, batch_size):
        group_items = prepared[start_idx : start_idx + batch_size]
        end_idx = start_idx + len(group_items)
        group_ex_ids = [item["example"]["example_id"] for item in group_items]

        for method in methods:
            if len(group_items) == 1:
                print(
                    f"\n[Progress {end_idx}/{total_q}] Question: {group_ex_ids[0]} | Model: {method.upper()}",
                    flush=True,
                )
            else:
                print(
                    f"\n[Progress {start_idx + 1}-{end_idx}/{total_q} | batch_size={len(group_items)}] "
                    f"Questions: {', '.join(group_ex_ids)} | Model: {method.upper()}",
                    flush=True,
                )

            cached_by_ex = {}
            pending_items = []
            for item in group_items:
                ex_id = item["example"]["example_id"]
                if resume and is_question_complete(stage_dir, method, ex_id, budgets):
                    cached_rows = load_completed_question_rows(stage_dir, method, ex_id, budgets)
                    for r in cached_rows:
                        match_sym = "✓" if r["exact_match"] else "✗"
                        print(
                            f"  [{match_sym} Cached Budget {r['search_budget']:4d}] Q: {ex_id:<10} | "
                            f"Model: {method.upper():8s} | Pred: {r['predicted_answer']!r:<10} | "
                            f"Ref: {r['reference_answer']!r:<10} | Match: {str(r['exact_match']):<5} | (loaded from disk)",
                            flush=True,
                        )
                    cached_by_ex[ex_id] = cached_rows
                else:
                    pending_items.append(item)

            computed_by_ex = defaultdict(list)
            if pending_items:
                rows = run_questions_batch_e2(
                    wrapper, pending_items, ctx, method, stage_dir,
                    budgets=budgets, max_depth=max_depth,
                    num_candidates=num_candidates, completion_cache=completion_cache,
                )
                for r in rows:
                    computed_by_ex[r["example_id"]].append(r)

            for item in group_items:
                ex_id = item["example"]["example_id"]
                rows_by_ex_method[(ex_id, method)] = (
                    cached_by_ex[ex_id] if ex_id in cached_by_ex else computed_by_ex[ex_id]
                )

            all_checkpoint_rows = [
                row
                for item in prepared[:end_idx]
                for m in methods
                for row in rows_by_ex_method.get((item["example"]["example_id"], m), [])
            ]
            save_e2_progress(output_dir, summaries_dir, ctx["stage"], all_checkpoint_rows, methods, budgets)

    manifest_data["status"] = "completed"
    manifest_data["completion_timestamp"] = perf_counter()
    agg_rows = save_e2_progress(output_dir, summaries_dir, ctx["stage"], all_checkpoint_rows, methods, budgets, manifest_data=manifest_data)

    return all_checkpoint_rows, agg_rows
