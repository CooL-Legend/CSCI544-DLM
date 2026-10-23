"""Unified runner for Experiment 1 (E1: Accuracy and Inference Efficiency).

Runs MEDAL and AB-MEDAL with verified prefix checkpointing at B in {256, 512, 1024},
shared minimum-C candidate selection (v2), completion caching, and comprehensive logging.
"""

import argparse
from collections import defaultdict
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import torch

from dlm_project.decoding import mcts_decoding as medal
from dlm_project.decoding.ab_medal import ABMEDALConfig, ABMEDALSearch
from dlm_project.decoding.mcts_decoding import MCTSConfig, MCTSSearch, select_deepest_candidates
from dlm_project.decoding.medal_demo import run_completion
from dlm_project.decoding import search_audit
from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.manifests import digest
from dlm_project.experiments.paths import create_run_directory
from dlm_project.experiments.sweep_report import wilson_interval, write_csv
from dlm_project.tasks.gsm8k.e0_tokens import resolve_termination_tokens, decode_output
from dlm_project.tasks.gsm8k.evaluation import score_answer
from dlm_project.tasks.gsm8k.experiment_protocol import (
    load_shared_protocol, get_manifest, select_examples, build_prompt, example_seed
)

PROTOCOL_VERSION = "e1_unified_fixed_depth_v1"
CHECKPOINT_BUDGETS = (256, 512, 1024)
MAX_SEARCH_DEPTH = 10
MIN_CANDIDATES = 3


def state_hash(state_tensor):
    return hashlib.sha256(repr(medal._state_key(state_tensor)).encode()).hexdigest()


def context(*, stage="dev", methods=("medal", "ab_medal"), budgets=CHECKPOINT_BUDGETS,
            max_depth=MAX_SEARCH_DEPTH, num_candidates=MIN_CANDIDATES, num_samples=None):
    """Create and validate execution context for E1."""
    common = load_shared_protocol()
    manifest = get_manifest(common=common)
    examples = select_examples(manifest, stage)
    if num_samples is not None:
        examples = examples[:num_samples]
    if not examples:
        raise ValueError(f"No examples found for stage {stage}.")

    for m in methods:
        if m not in ("medal", "ab_medal"):
            raise ValueError(f"Unknown method {m}. Must be 'medal' or 'ab_medal'.")

    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": stage,
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


def _jsonl(path: Path, rows: Sequence[Dict[str, Any]]):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + "\n")


def execute_search_checkpointed(wrapper, prepared_item, ctx, method: str,
                                 budgets: Tuple[int, ...] = CHECKPOINT_BUDGETS,
                                 max_depth: int = MAX_SEARCH_DEPTH,
                                 num_candidates: int = MIN_CANDIDATES):
    """Execute continuous search up to max(budgets) capturing checkpoints."""
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
        search_cls = ABMEDALSearch
    else:
        search_cfg = MCTSConfig(**cfg_kwargs)
        search_cls = MCTSSearch

    search = search_cls(
        wrapper.model,
        prepared_item["ids"],
        wrapper.mask_id,
        ctx["common"]["generation"]["gen_length"],
        config=search_cfg,
        seed=prepared_item["seed"],
        tokenizer=wrapper.tokenizer,
        search_excluded_token_ids=ctx["termination"]["token_ids"],
        checkpoint_budgets=tuple(sorted(budgets)),
    )

    t0 = perf_counter()
    search.run()
    search_time = perf_counter() - t0

    # Ensure all requested checkpoints are populated (including early finish)
    for b in budgets:
        if b not in search.checkpoints:
            search.checkpoints[b] = search._snapshot_checkpoint(b, stop_reason=search.stop_reason)

    return search, search_time


def complete_selected_state(wrapper, selected_state, prepared_item, ctx, completion_cache=None):
    """Complete a selected state with caching."""
    s_hash = state_hash(selected_state)
    cache_key = (prepared_item["example"]["example_id"], s_hash)

    if completion_cache is not None and cache_key in completion_cache:
        cached = completion_cache[cache_key]
        return cached["completion"], cached["decoded"], cached["score"], 0.0, True

    t0 = perf_counter()
    prompt_len = prepared_item["ids"].shape[1]
    search_stub = SimpleNamespace(
        model=wrapper.model,
        prompt_len=prompt_len,
        mask_id=wrapper.mask_id,
        config=MCTSConfig(),
        tokenizer=wrapper.tokenizer,
    )
    init_stub = SimpleNamespace(search=search_stub, state=selected_state.clone())

    completion = run_completion(
        init_stub,
        method="llada",
        temperature=0.0,
        seed=prepared_item["seed"],
        tokenizer=wrapper.tokenizer,
        termination_token_ids=ctx["termination"]["token_ids"],
    )
    comp_time = perf_counter() - t0

    decoded = decode_output(
        wrapper.tokenizer,
        completion.token_ids[0, prompt_len:].tolist(),
        ctx["termination"]["token_ids"],
    )
    score = score_answer(decoded["scoring_text"], prepared_item["example"]["reference_solution"])

    record = {"completion": completion, "decoded": decoded, "score": score}
    if completion_cache is not None:
        completion_cache[cache_key] = record

    return completion, decoded, score, comp_time, False


def run_question_e1(wrapper, prepared_item, ctx, method: str,
                    base_dir: Path,
                    budgets: Tuple[int, ...] = CHECKPOINT_BUDGETS,
                    max_depth: int = MAX_SEARCH_DEPTH,
                    num_candidates: int = MIN_CANDIDATES,
                    completion_cache=None):
    """Run one question under a method and export all checkpoint and overall artifacts."""
    q_dir = base_dir / method / prepared_item["example"]["example_id"]
    q_dir.mkdir(parents=True, exist_ok=True)

    search, search_time = execute_search_checkpointed(
        wrapper, prepared_item, ctx, method, budgets=budgets,
        max_depth=max_depth, num_candidates=num_candidates
    )

    # 1. Overall question trace files
    events = [_compact(e) for e in search.trace]
    _jsonl(q_dir / "trace.jsonl", events)
    _jsonl(q_dir / "backups.jsonl", [e for e in events if e.get("stage") == "backpropagation"])
    _jsonl(q_dir / "forwards.jsonl", [e for e in events if e.get("stage") == "forward"])

    decisions = [e for e in events if e.get("stage") == "adaptive_decision"]
    _jsonl(q_dir / "decisions.jsonl", decisions)

    # Overall audit
    audit = search_audit.from_events(
        events,
        prompt_ids=prepared_item["prompt_ids"],
        mask_id=wrapper.mask_id,
        gen_length=ctx["common"]["generation"]["gen_length"],
        requested_depth=max_depth,
        requested_budget=max(budgets),
        search_nfes=search.model_calls,
        adaptive=(method == "ab_medal"),
        shortlist_limit=num_candidates,
    )
    save_json(q_dir / "search_audit.json", audit)
    _jsonl(q_dir / "nodes.jsonl", audit["nodes"])

    prompt_record = {k: v for k, v in prepared_item.items() if k != "ids"}
    prompt_record.update(mask_id=wrapper.mask_id, gen_length=ctx["common"]["generation"]["gen_length"])
    save_json(q_dir / "prompt.json", prompt_record)
    save_json(q_dir / "config.json", ctx)

    # 2. Checkpoint results
    checkpoint_results = []
    checkpoints_dir = q_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for budget in sorted(budgets):
        chk = search.checkpoints[budget]
        chk_dir = checkpoints_dir / str(budget)
        chk_dir.mkdir(parents=True, exist_ok=True)

        selected_node = chk["selected"]
        completion, decoded, score, comp_time, was_cached = complete_selected_state(
            wrapper, selected_node.state, prepared_item, ctx, completion_cache=completion_cache
        )

        comp_nfes = 256 - selected_node.depth
        total_nfes = budget + comp_nfes

        # Build checkpoint search_audit
        chk_trace = chk["trace"] + [chk["selection"]]
        chk_audit = search_audit.from_events(
            chk_trace,
            prompt_ids=prepared_item["prompt_ids"],
            mask_id=wrapper.mask_id,
            gen_length=ctx["common"]["generation"]["gen_length"],
            requested_depth=max_depth,
            requested_budget=budget,
            search_nfes=budget,
            adaptive=(method == "ab_medal"),
            shortlist_limit=num_candidates,
        )
        chk_metrics = search_audit.metrics(chk_audit)

        # Save checkpoint artifacts
        save_json(chk_dir / "selection.json", chk["selection"])
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
                "state_hash": candidate_value_record(n)["state_hash"],
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
                str(node_id): {k: asdict(p) if hasattr(p, '__dataclass_fields__') else p for k, p in models.items()}
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

    return checkpoint_results


def run_e1_stage(wrapper, ctx, output_dir: Path, methods=("medal", "ab_medal"),
                 budgets=CHECKPOINT_BUDGETS, max_depth=MAX_SEARCH_DEPTH,
                 num_candidates=MIN_CANDIDATES):
    """Execute complete E1 stage for all methods and generate aggregate and per-question CSVs."""
    output_dir = Path(output_dir)
    stage_dir = output_dir / ctx["stage"]
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare(wrapper, ctx)
    all_checkpoint_rows = []
    completion_cache = {}

    for item in prepared:
        ex_id = item["example"]["example_id"]
        for method in methods:
            rows = run_question_e1(
                wrapper, item, ctx, method, stage_dir,
                budgets=budgets, max_depth=max_depth,
                num_candidates=num_candidates, completion_cache=completion_cache
            )
            all_checkpoint_rows.extend(rows)

    # Export e1_per_question.csv
    per_q_path = summaries_dir / f"e1_{ctx['stage']}_per_question.csv"
    write_csv(per_q_path, all_checkpoint_rows)
    write_csv(summaries_dir / "e1_per_question.csv", all_checkpoint_rows)

    # Compute e1_aggregate.csv
    agg_rows = []
    for budget in sorted(budgets):
        for method in methods:
            subset = [r for r in all_checkpoint_rows if r["method"] == method and r["search_budget"] == budget]
            n = len(subset)
            if not n:
                continue
            correct_cnt = sum(r["exact_match"] for r in subset)
            parsed_cnt = sum(r["parse_success"] for r in subset)
            ci_low, ci_high = wilson_interval(correct_cnt, n)

            agg_rows.append({
                "stage": ctx["stage"],
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

    agg_path = summaries_dir / f"e1_{ctx['stage']}_aggregate.csv"
    write_csv(agg_path, agg_rows)
    write_csv(summaries_dir / "e1_aggregate.csv", agg_rows)

    # Save run manifest
    manifest_data = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": ctx["stage"],
        "methods": list(methods),
        "budgets": list(budgets),
        "max_depth": max_depth,
        "num_candidates": num_candidates,
        "sample_count": len(prepared),
        "examples": [p["example"]["example_id"] for p in prepared],
        "setup": ctx.get("setup", {}),
        "model": ctx.get("model", {}),
    }
    save_json(output_dir / "run_manifest.json", manifest_data)
    save_json(summaries_dir / f"run_manifest_{ctx['stage']}.json", manifest_data)

    return all_checkpoint_rows, agg_rows


def _compact(value):
    omitted = {"entropies", "entropies_before", "entropies_after", "state", "child_state",
               "selected_state", "candidate_evaluations"}
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items() if k not in omitted}
    if isinstance(value, list):
        return [_compact(v) for v in value]
    return value


def candidate_value_record(node):
    return {
        "node_id": node.node_id,
        "depth": node.depth,
        "visits": node.visits,
        "reward_sum": node.reward_sum,
        "q_value": node.mean_reward,
        "reward": node.mean_reward,
        "immediate_gain": node.immediate_reward,
        "state_hash": hashlib.sha256(repr(medal._state_key(node.state)).encode()).hexdigest(),
    }


def load_model(ctx=None, *, model_path=None, device="cuda", local_files_only=False):
    """Load LLaDA wrapper for E1 experiments."""
    from dlm_project.models.llada import LLaDA
    dtype = ctx["common"]["model"]["dtype"] if ctx and "common" in ctx else "bfloat16"
    return LLaDA(device=device, model_path=model_path, dtype=dtype, local_files_only=local_files_only)

