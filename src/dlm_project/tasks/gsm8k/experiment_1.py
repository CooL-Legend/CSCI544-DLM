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
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import torch

from dlm_project.decoding import mcts_decoding as medal
from dlm_project.decoding.ab_batch import pack_states, sync
from dlm_project.decoding.ab_medal import ABMEDALConfig, ABMEDALSearch
from dlm_project.decoding.common import commit_from_logits
from dlm_project.decoding.mcts_decoding import MCTSConfig, MCTSSearch, select_deepest_candidates
from dlm_project.decoding.medal_batch import CooperativeMEDALSearch
from dlm_project.decoding.medal_demo import DecodingResult, run_completion
from dlm_project.decoding import search_audit
from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.resources import (
    profiled_stage, resource_phase, watch_checkpoints, record_prepared,
    record_batch, record_completed, synchronize,
)
from dlm_project.experiments.resume import (
    resume_signature, validate_resume, question_complete, begin_question, finish_question,
)
from dlm_project.experiments.manifests import digest
from dlm_project.experiments.paths import create_run_directory
from dlm_project.experiments.sweep_report import wilson_interval, write_csv
from dlm_project.tasks.gsm8k.e0_tokens import resolve_termination_tokens, decode_output
from dlm_project.tasks.gsm8k.evaluation import score_answer
from dlm_project.tasks.gsm8k.experiment_protocol import (
    EXTENDED_TEST_COUNT, STAGE_COUNTS,
    load_shared_protocol, get_extended_manifest, get_manifest, select_examples, build_prompt, example_seed
)

PROTOCOL_VERSION = "e1_unified_fixed_depth_v1"


def _load_default_budgets() -> Tuple[int, ...]:
    try:
        common = load_shared_protocol()
        return tuple(c["budget"] for c in common["search"]["conditions"])
    except Exception:
        return (256, 512)


CHECKPOINT_BUDGETS = _load_default_budgets()
MAX_SEARCH_DEPTH = 10
MIN_CANDIDATES = 3


def state_hash(state_tensor):
    return hashlib.sha256(repr(medal._state_key(state_tensor)).encode()).hexdigest()


def context(*, stage="dev", methods=("medal", "ab_medal"), budgets=None,
            max_depth=MAX_SEARCH_DEPTH, num_candidates=MIN_CANDIDATES, num_samples=None):
    """Create and validate execution context for E1."""
    common = load_shared_protocol()
    if budgets is None:
        budgets = tuple(c["budget"] for c in common["search"]["conditions"])
    if stage == "test" and num_samples is not None and num_samples > STAGE_COUNTS["test"]:
        manifest = get_extended_manifest(common=common, test_count=EXTENDED_TEST_COUNT)
        examples = select_examples(manifest, stage, expected_count=EXTENDED_TEST_COUNT)
    else:
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


@resource_phase("search")
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

    sync(wrapper.model)
    t0 = perf_counter()
    watch_checkpoints(search, wrapper.model, t0)
    search.run()
    sync(wrapper.model)
    search_time = perf_counter() - t0

    # Ensure all requested checkpoints are populated (including early finish)
    for b in budgets:
        if b not in search.checkpoints:
            search.checkpoints[b] = search._snapshot_checkpoint(b, stop_reason=search.stop_reason)

    return search, search_time


@resource_phase("completion")
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


@torch.inference_mode()
@resource_phase("search")
def execute_search_batch_checkpointed(wrapper, prepared_group, ctx, method: str,
                                      budgets: Tuple[int, ...] = CHECKPOINT_BUDGETS,
                                      max_depth: int = MAX_SEARCH_DEPTH,
                                      num_candidates: int = MIN_CANDIDATES):
    """Execute cooperative batched search up to max(budgets) across a group of questions."""
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
        search_cls = CooperativeMEDALSearch

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
    for tree in trees:
        watch_checkpoints(tree, wrapper.model, t0)
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
@resource_phase("completion")
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
            for i in active:
                unique_uncached[i]["calls"] += 1
                unique_uncached[i]["trace"].append({
                    "stage": "forward",
                    "phase": "completion",
                    "model_calls": unique_uncached[i]["calls"],
                    "invocation_id": invocation_id,
                    "batch_size": len(active),
                })
            prediction = wrapper.model(ids) if len(active) == 1 else wrapper.model(ids, attention_mask=attention)
            for row, i in enumerate(active):
                entry = unique_uncached[i]
                state = entry["state"]
                prompt_len = entry["prompt_len"]
                positions = torch.nonzero(state[0, prompt_len:] == mask_id, as_tuple=False).flatten() + prompt_len
                actions, signals = commit_from_logits(
                    state,
                    prediction.logits[row, :state.shape[1]],
                    prompt_len,
                    mask_id,
                    positions,
                    1,
                    selection_rule="token_probability",
                    temperature=0.0,
                )
                entry["trace"].append({
                    "stage": "commit",
                    "block": 0,
                    "step": entry["calls"] - 1,
                    "model_calls": entry["calls"],
                    "actions": actions,
                    **signals,
                    "remaining_masks": int((state[0, prompt_len:] == mask_id).sum().item()),
                })
                if not (state[0, prompt_len:] == mask_id).any():
                    sync(wrapper.model)
                    entry["comp_time"] = perf_counter() - t0
            del prediction

        sync(wrapper.model)
        total_comp_time = perf_counter() - t0
        for entry in unique_uncached:
            comp_time = entry["comp_time"] if entry["comp_time"] is not None else total_comp_time
            item = entry["item"]
            prompt_len = entry["prompt_len"]
            token_ids = entry["state"].detach().cpu()
            completion = DecodingResult(
                token_ids=token_ids,
                model_calls=entry["calls"],
                wall_seconds=comp_time,
                method="llada",
                trace=entry["trace"],
            )
            decoded = decode_output(
                wrapper.tokenizer,
                completion.token_ids[0, prompt_len:].tolist(),
                ctx["termination"]["token_ids"],
            )
            score = score_answer(decoded["scoring_text"], item["example"]["reference_solution"])
            record = {"completion": completion, "decoded": decoded, "score": score}
            if completion_cache is not None:
                completion_cache[entry["cache_key"]] = record
            results_map[entry["primary_req_key"]] = (completion, decoded, score, comp_time, False)
            for dup_key in entry["duplicate_req_keys"]:
                results_map[dup_key] = (completion, decoded, score, 0.0, True)

    return results_map


@resource_phase("artifacts")
def export_question_e1_artifacts(wrapper, search, search_time: float, prepared_item, ctx, method: str,
                                 base_dir: Path,
                                 budgets: Tuple[int, ...] = CHECKPOINT_BUDGETS,
                                 max_depth: int = MAX_SEARCH_DEPTH,
                                 num_candidates: int = MIN_CANDIDATES,
                                 completion_cache=None,
                                 precomputed_completions=None):
    """Export all per-question and per-checkpoint E1 artifacts from a completed search."""
    q_dir = base_dir / method / prepared_item["example"]["example_id"]
    q_dir.mkdir(parents=True, exist_ok=True)
    begin_question(q_dir)

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
        req_key = (prepared_item["example"]["example_id"], state_hash(selected_node.state), budget)
        if precomputed_completions is not None and req_key in precomputed_completions:
            completion, decoded, score, comp_time, was_cached = precomputed_completions[req_key]
        else:
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
        selection_record = {
            **chk["selection"],
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
            "search_wall_seconds": chk["search_elapsed_seconds"],
            "completion_wall_seconds": comp_time,
            "total_wall_seconds": (chk["search_elapsed_seconds"]) + comp_time,
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
        total_time = (chk["search_elapsed_seconds"]) + comp_time
        print(
            f"  [{match_sym} Budget {budget:4d}] Q: {prepared_item['example']['example_id']:<10} | "
            f"Model: {method.upper():8s} | Pred: {score['predicted_answer']!r:<10} | "
            f"Ref: {score['reference_answer']!r:<10} | Match: {str(score['exact_match']):<5} | Time: {total_time:.1f}s",
            flush=True,
        )

    finish_question(q_dir, method, prepared_item["example"]["example_id"], budgets)
    return checkpoint_results


def run_question_e1(wrapper, prepared_item, ctx, method: str,
                    base_dir: Path,
                    budgets: Tuple[int, ...] = CHECKPOINT_BUDGETS,
                    max_depth: int = MAX_SEARCH_DEPTH,
                    num_candidates: int = MIN_CANDIDATES,
                    completion_cache=None):
    """Run one question under a method and export all checkpoint and overall artifacts."""
    search, search_time = execute_search_checkpointed(
        wrapper, prepared_item, ctx, method, budgets=budgets,
        max_depth=max_depth, num_candidates=num_candidates
    )
    return export_question_e1_artifacts(
        wrapper, search, search_time, prepared_item, ctx, method, base_dir,
        budgets=budgets, max_depth=max_depth, num_candidates=num_candidates,
        completion_cache=completion_cache,
    )


def run_questions_batch_e1(wrapper, prepared_group, ctx, method: str,
                           base_dir: Path,
                           budgets: Tuple[int, ...] = CHECKPOINT_BUDGETS,
                           max_depth: int = MAX_SEARCH_DEPTH,
                           num_candidates: int = MIN_CANDIDATES,
                           completion_cache=None):
    """Run a batch of questions cooperatively under a method and export all E1 artifacts."""
    if len(prepared_group) == 1:
        return run_question_e1(
            wrapper, prepared_group[0], ctx, method, base_dir,
            budgets=budgets, max_depth=max_depth, num_candidates=num_candidates,
            completion_cache=completion_cache,
        )

    search_pairs = execute_search_batch_checkpointed(
        wrapper, prepared_group, ctx, method, budgets=budgets,
        max_depth=max_depth, num_candidates=num_candidates,
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
        rows = export_question_e1_artifacts(
            wrapper, search, search_time, item, ctx, method, base_dir,
            budgets=budgets, max_depth=max_depth, num_candidates=num_candidates,
            completion_cache=completion_cache,
            precomputed_completions=precomputed_completions,
        )
        batch_rows.extend(rows)

    return batch_rows


def is_question_complete(stage_dir: Path, method: str, ex_id: str, budgets: Sequence[int]) -> bool:
    return question_complete(stage_dir, method, ex_id, budgets)


def load_completed_question_rows(stage_dir: Path, method: str, ex_id: str, budgets: Sequence[int]) -> List[Dict[str, Any]]:
    """Load cached checkpoint rows directly from metrics.json on disk."""
    rows = []
    q_dir = stage_dir / method / ex_id
    for b in sorted(budgets):
        met_file = q_dir / "checkpoints" / str(b) / "metrics.json"
        row = json.loads(met_file.read_text())
        rows.append(row)
    return rows


def compute_e1_aggregate_rows(all_checkpoint_rows: List[Dict[str, Any]], stage: str,
                              methods: Sequence[str], budgets: Sequence[int]) -> List[Dict[str, Any]]:
    """Compute aggregate accuracy, NFEs, and Wilson confidence intervals."""
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


@resource_phase("artifacts")
def save_e1_progress(output_dir: Path, summaries_dir: Path, stage: str,
                     all_checkpoint_rows: List[Dict[str, Any]],
                     methods: Sequence[str], budgets: Sequence[int],
                     manifest_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Immediately write and flush per-question and aggregate CSVs and manifest, syncing OS buffers."""
    summaries_dir.mkdir(parents=True, exist_ok=True)
    agg_rows = []
    if all_checkpoint_rows:
        per_q_path = summaries_dir / f"e1_{stage}_per_question.csv"
        write_csv(per_q_path, all_checkpoint_rows)
        write_csv(summaries_dir / "e1_per_question.csv", all_checkpoint_rows)

        agg_rows = compute_e1_aggregate_rows(all_checkpoint_rows, stage, methods, budgets)
        agg_path = summaries_dir / f"e1_{stage}_aggregate.csv"
        write_csv(agg_path, agg_rows)
        write_csv(summaries_dir / "e1_aggregate.csv", agg_rows)

    if manifest_data is not None:
        save_json(output_dir / "run_manifest.json", manifest_data)
        save_json(summaries_dir / f"run_manifest_{stage}.json", manifest_data)

    try:
        os.sync()
    except (AttributeError, OSError):
        pass

    return agg_rows


@profiled_stage
def run_e1_stage(wrapper, ctx, output_dir: Path, methods=("medal", "ab_medal"),
                 budgets=CHECKPOINT_BUDGETS, max_depth=MAX_SEARCH_DEPTH,
                 num_candidates=MIN_CANDIDATES, resume: bool = True,
                 batch_size: int = 1, profile_resources: bool = False):
    """Execute complete E1 stage for all methods with incremental saving, resumption, and optional cooperative batching."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    output_dir = Path(output_dir)
    stage_dir = output_dir / ctx["stage"]
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare(wrapper, ctx)
    record_prepared(prepared)
    signature = resume_signature(ctx, prepared, methods, budgets, max_depth, num_candidates, batch_size)
    validate_resume(output_dir, ctx["stage"], signature, resume)
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
        "resume_signature": signature,
    }
    # Initial save of manifest
    save_json(output_dir / "run_manifest.json", manifest_data)
    save_json(summaries_dir / f"run_manifest_{ctx['stage']}.json", manifest_data)

    total_q = len(prepared)
    rows_by_ex_method: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    ordered_ex_ids = [p["example"]["example_id"] for p in prepared]

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

            record_batch(pending_items, cached_by_ex, method)
            computed_by_ex = defaultdict(list)
            if pending_items:
                if batch_size == 1 or len(pending_items) == 1:
                    item = pending_items[0]
                    rows = run_question_e1(
                        wrapper, item, ctx, method, stage_dir,
                        budgets=budgets, max_depth=max_depth,
                        num_candidates=num_candidates, completion_cache=completion_cache,
                    )
                    computed_by_ex[item["example"]["example_id"]].extend(rows)
                else:
                    rows = run_questions_batch_e1(
                        wrapper, pending_items, ctx, method, stage_dir,
                        budgets=budgets, max_depth=max_depth,
                        num_candidates=num_candidates, completion_cache=completion_cache,
                    )
                    for r in rows:
                        computed_by_ex[r["example_id"]].append(r)

            record_completed([r for rows in computed_by_ex.values() for r in rows])
            for item in group_items:
                ex_id = item["example"]["example_id"]
                rows_by_ex_method[(ex_id, method)] = (
                    cached_by_ex[ex_id] if ex_id in cached_by_ex else computed_by_ex[ex_id]
                )

            all_checkpoint_rows = [
                row
                for ex_id in ordered_ex_ids
                for m in methods
                if (ex_id, m) in rows_by_ex_method
                for row in rows_by_ex_method[(ex_id, m)]
            ]
            manifest_data["completed_examples_count"] = end_idx
            agg_rows = save_e1_progress(
                output_dir, summaries_dir, ctx["stage"],
                all_checkpoint_rows, methods, budgets,
                manifest_data=manifest_data,
            )
            print(f"  💾 Saved progress to CSV ({len(all_checkpoint_rows)} rows recorded). Disk synced.", flush=True)

    manifest_data["status"] = "completed"
    agg_rows = save_e1_progress(
        output_dir, summaries_dir, ctx["stage"],
        all_checkpoint_rows, methods, budgets,
        manifest_data=manifest_data,
    )
    print(f"\n✅ All {total_q} questions completed successfully for methods {list(methods)}.", flush=True)

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
    started = perf_counter()
    wrapper = LLaDA(device=device, model_path=model_path, dtype=dtype, local_files_only=local_files_only)
    synchronize(wrapper.model)
    wrapper.model_load_seconds = perf_counter() - started
    if ctx is not None:
        ctx.setdefault("setup", {})["model_load_seconds"] = wrapper.model_load_seconds
    return wrapper

