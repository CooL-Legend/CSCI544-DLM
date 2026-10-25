"""Staged AB search/completion helpers, including explicit legacy quota mode."""

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import torch

from dlm_project.decoding.ab_medal import ABMEDALSearch, tree_record, tree_statistics
from dlm_project.decoding import medal_demo as demo
from dlm_project.decoding.mcts_decoding import SearchIncompleteError, derive_seed
from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.paths import create_run_directory


@torch.inference_mode()
def run_initialization(model, prompt_ids, mask_id, *, gen_length=256, mcts=None,
                       seed=1, tokenizer=None, search_excluded_token_ids=()):
    """Return complete or capped AB search results without performing decoding.

    Only the expected incomplete-search exception is handled here. Model errors
    and notebook interrupts still propagate; the CLI runner retains its own
    durable failure diagnostics.
    """
    demo._sync(model)
    started = perf_counter()
    search = ABMEDALSearch(model, prompt_ids, mask_id, gen_length, mcts,
                           tokenizer=tokenizer, seed=seed,
                           search_excluded_token_ids=search_excluded_token_ids)
    try:
        search.run()
    except SearchIncompleteError:
        pass
    finally:
        demo._sync(model)
    seconds = perf_counter() - started
    result = search.result()
    status = "complete" if search.config.stop_mode == "budget" or result.stats["target_reached"] else "incomplete"
    print(f"AB initialization {status}: candidates={len(result.candidates)}/{search.config.num_candidates}; "
          f"stop={search.stop_reason}", flush=True)
    return demo.InitializationResult(search, result, seconds)


def run_completion(initialization, *, method=None, seed=1, temperature=None,
                   tokenizer=None, termination_token_ids=None):
    """Confidence completion for budget runs, including deepest-state fallback."""
    legacy = initialization.search.config.stop_mode == "candidates"
    if not initialization.search.done:
        raise ValueError("Finish the search before completing its selected state.")
    if legacy and not initialization.result.stats["target_reached"]:
        print("Completion and scoring skipped: AB initialization did not reach the candidate target.", flush=True)
        return None
    return demo.run_completion(initialization, method=method or ("medal" if legacy else "llada"), seed=seed,
                               temperature=(1.0 if legacy else 0.0) if temperature is None else temperature,
                               tokenizer=tokenizer, termination_token_ids=termination_token_ids)


def full_target_dlm_gate(wrapper, initialization):
    """The existing Gaussian depth-20 / three-candidate LLaDA gate, from cache."""
    search = initialization.search
    config = search.config
    return bool(initialization.result.stats["target_reached"]
                and config.init_tokens == 20 and config.num_candidates == 3
                and config.search_policy == "gaussian"
                and getattr(wrapper, "identity", {}).get("family") == "llada"
                and all(node.depth == 20 for node in initialization.result.candidates))


def export_diagnostics(wrapper, initialization, output_dir, *, completion=None,
                       prompt=None, prompt_metadata=None, question=None, score=None,
                       temperature=1.0, completion_seed=None, print_history=False,
                       plot_max_nodes=120, plot_max_depth=20):
    """Export one initialization plus the supplied completion, using cached data.

    Calling this after multiple sampler trials exports only the supplied trial.
    No search, sampler, posterior draw, or model load happens during export.
    """
    from scripts.run_ab_medal_single import plot_tree

    search, stats = initialization.search, initialization.result.stats
    meta = prompt_metadata or {}
    report = demo.show_metrics(initialization, completion,
                               prompt_seconds=meta.get("prompt_wall_seconds", 0.0),
                               setup_seconds=meta.get("setup_wall_seconds", 0.0),
                               score=score if completion is not None else None, display=False)
    directory, _ = create_run_directory({"output_dir": str(output_dir)})
    seed = search.seed if completion_seed is None else completion_seed
    save_json(directory / "config.json", {
        "method": "ab_medal", "seed": search.seed,
        "completion_seed": derive_seed(seed, "final_completion"),
        "completion_method": completion.method if completion is not None else None,
        "temperature": temperature, "gen_length": search.root.state.shape[1] - search.prompt_len,
        "search": asdict(search.config), "model": getattr(wrapper, "identity", {}), "batch_size": 1,
    })
    save_json(directory / "prompt.json", {"question": question, "text": prompt, "metadata": meta})
    text = None
    if completion is not None:
        text = wrapper.tokenizer.decode(completion.token_ids[0, search.prompt_len:].tolist(), skip_special_tokens=True)
        save_json(directory / "completion.json", {"text": text, "score": score,
                                                   "token_ids": completion.token_ids.tolist()})
    with (directory / "trace.jsonl").open("w", encoding="utf-8") as trace:
        events = list(initialization.result.trace)
        if completion is not None:
            events.extend({**event, "phase": "decoding"} for event in completion.trace)
        for event in events:
            line = json.dumps(event, allow_nan=False)
            trace.write(line + "\n")
            if print_history:
                print(line, flush=True)
    tree = tree_record(search)
    save_json(directory / "tree.json", tree)
    save_json(directory / "history.json", search.history)
    report.update(
        method="ab_medal", status=("incomplete" if not stats["target_reached"]
                                    else "complete" if completion is not None else "initialized"),
        target_reached=stats["target_reached"], full_target_dlm_gate=full_target_dlm_gate(wrapper, initialization),
        decoding_performed=completion is not None, response=text, search=stats,
        structure=tree_statistics(search), seed=search.seed, artifact_directory=str(directory),
        initialization_model_calls=stats["search_model_calls"], initialization_nfe=stats["search_sequence_nfes"],
        decoding_model_calls=report["decoding_nfe"], total_model_calls=report["total_nfe"],
    )
    report["tree_plot"] = plot_tree(tree, directory / "ab_medal_tree.png",
                                   max_nodes=plot_max_nodes, max_depth=plot_max_depth)
    save_json(directory / "metrics.json", report)
    print("AB decisions:", stats["decision_counts"])
    print("Candidate diversity:", report["structure"]["candidate_diversity"])
    print("Full-target DLM gate:", "OBSERVED" if report["full_target_dlm_gate"] else "NOT SATISFIED")
    print("Artifacts:", directory, flush=True)
    return directory, report


def run_pilot(wrapper, initialization, demonstrations_path, manifest_path, output_dir, *,
              num_samples=5, gen_length=256, mcts=None, seed=1, temperature=1.0,
              cache_dir=None):
    """Run the existing gated AB pilot, sharing the model and pinned revision."""
    if not full_target_dlm_gate(wrapper, initialization):
        raise RuntimeError("Run and satisfy the full-target Gaussian DLM check first.")

    from dlm_project.tasks.gsm8k.benchmark import example_seed
    from dlm_project.tasks.gsm8k.benchmark_dataset import get_manifest, select_examples
    from scripts.run_ab_medal_single import run_attempt

    config = initialization.search.config if mcts is None else mcts
    path = Path(demonstrations_path)
    revision = json.loads(path.read_text(encoding="utf-8"))["dataset"]["revision"]
    manifest = get_manifest(manifest_path, cache_dir=cache_dir, revision=revision)
    examples = select_examples(manifest, "smoke", num_samples)
    directory, _ = create_run_directory({"output_dir": str(output_dir)})
    save_json(directory / "config.json", {
        "search": asdict(config), "seed": seed, "gen_length": gen_length,
        "temperature": temperature, "dataset": manifest["dataset"],
        "model": getattr(wrapper, "identity", {}), "example_ids": [row["example_id"] for row in examples],
    })
    rows = []

    def save_summary(status, error=None):
        scored = [row for row in rows if row["status"] == "complete" and row["accuracy"] is not None]
        count = len(rows)
        summary = {
            "status": status, "error": error, "requested": len(examples), "processed": count,
            "target_attained": sum(row["target_reached"] for row in rows),
            "target_coverage": sum(row["target_reached"] for row in rows) / len(examples),
            "scored": len(scored),
            "conditional_accuracy": sum(row["accuracy"] for row in scored) / len(scored) if scored else None,
            "total_nfe": sum(row["total_nfe"] for row in rows),
            "total_wall_seconds": sum(row["total_wall_seconds"] for row in rows),
            "mean_nfe": sum(row["total_nfe"] for row in rows) / count if count else None,
            "mean_wall_seconds": sum(row["total_wall_seconds"] for row in rows) / count if count else None,
            "artifact_directory": str(directory),
        }
        save_json(directory / "summary.json", summary)
        return summary

    save_summary("running")
    status, error = "running", None
    try:
        with (directory / "attempts.jsonl").open("w", encoding="utf-8") as output:
            for item in examples:
                prompt, ids, meta = demo.prepare_demo_prompt(
                    wrapper, item["question"], path, revision=revision, cache_dir=cache_dir)
                attempt_dir = directory / f"example_{item['row_id']}"
                attempt_dir.mkdir()
                save_json(attempt_dir / "prompt.json", {"text": prompt, "metadata": meta})
                row = run_attempt(
                    wrapper, ids, config, attempt_dir, seed=example_seed(seed, item["row_id"]),
                    gen_length=gen_length, temperature=temperature, reference=item["reference_solution"],
                    metadata=item, prompt_seconds=meta["prompt_wall_seconds"],
                    setup_seconds=meta["setup_wall_seconds"], print_history=False,
                    plot_max_nodes=120, plot_max_depth=20,
                )
                rows.append(row)
                output.write(json.dumps(row, allow_nan=False) + "\n")
                output.flush()
                save_summary("running")
                if row["status"] == "interrupted":
                    raise KeyboardInterrupt(row["error"])
                if row["status"] == "error":
                    raise RuntimeError(row["error"])
        status = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        error = f"{type(exc).__name__}: {exc}"
    summary = save_summary(status, error)
    print(json.dumps(summary, indent=2), flush=True)
    return directory, summary
