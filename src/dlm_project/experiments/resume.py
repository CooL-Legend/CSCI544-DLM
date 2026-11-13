"""Validate experiment identity and complete per-question artifact sets."""

import json
from pathlib import Path

from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.manifests import digest
from dlm_project.experiments.resources import source_fingerprint


def resume_signature(ctx, prepared, methods, budgets, max_depth, num_candidates, batch_size):
    model = dict(ctx.get("model", {}))
    if model.get("repo_id") and model.get("revision"):
        model.pop("model_path", None)  # A staged copy of the same pinned snapshot is compatible.
    settings = {
        "protocol": ctx["protocol_version"], "model": model, "common": ctx["common"],
        "methods": list(methods), "budgets": list(budgets), "max_depth": max_depth,
        "num_candidates": num_candidates, "batch_size": batch_size,
        "reward_mode": ctx.get("reward_mode"), "reward_lambda": ctx.get("reward_lambda"),
        "sim_cap_multiplier": ctx.get("sim_cap_multiplier"),
        "source_fingerprint": source_fingerprint(),
    }
    return {"settings_hash": digest(settings), "settings": settings,
            "examples": [{"example_id": p["example"]["example_id"],
                          "identity_hash": digest({"example": p["example"], "seed": p["seed"],
                                                   "prompt_ids": p["prompt_ids"]})} for p in prepared]}


def validate_resume(output_dir, stage, signature, resume):
    """Permit identical runs and frozen-prefix extensions; reject stale configuration."""
    output_dir = Path(output_dir)
    path = output_dir / "summaries" / f"run_manifest_{stage}.json"
    if not path.exists():
        if (output_dir / stage).exists() and any((output_dir / stage).rglob("metrics.json")):
            raise ValueError("Existing artifacts have no validated run manifest. Use a new output directory.")
        return
    try:
        old = json.loads(path.read_text())["resume_signature"]
    except (OSError, ValueError, KeyError) as error:
        raise ValueError("Legacy or invalid resume manifest. Preserve it and use a new output directory.") from error
    if old["settings_hash"] != signature["settings_hash"]:
        raise ValueError("Resume configuration differs (model, protocol, batching or decoding). Use a new output directory.")
    previous, current = old["examples"], signature["examples"]
    if len(current) < len(previous) or current[:len(previous)] != previous:
        raise ValueError("Resume samples are not an identical frozen prefix extension. Use a new output directory.")
    # `resume=False` may recompute an identical run, but cannot silently mix configurations.


def question_complete(stage_dir, method, ex_id, budgets):
    directory = Path(stage_dir) / method / ex_id
    try:
        marker = json.loads((directory / "complete.json").read_text())
        if marker != {"method": method, "example_id": ex_id, "budgets": sorted(budgets)}:
            return False
        trace = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines() if line.strip()]
        if not trace or not all(isinstance(row, dict) for row in trace):
            return False
        for budget in budgets:
            checkpoint = directory / "checkpoints" / str(budget)
            rows = {name: json.loads((checkpoint / f"{name}.json").read_text())
                    for name in ("metrics", "tree", "selection", "completion")}
            metrics, tree, selection, completion = (rows[n] for n in ("metrics", "tree", "selection", "completion"))
            if (metrics["method"] != method or metrics["example_id"] != ex_id
                    or metrics["search_budget"] != budget or tree["budget"] != budget
                    or tree["example_id"] != ex_id or tree["method"] != method
                    or not isinstance(metrics["exact_match"], bool)
                    or metrics["predicted_answer"] != completion["score"]["predicted_answer"]
                    or metrics["exact_match"] != completion["score"]["exact_match"]
                    or not completion["token_ids"] or not tree["nodes"]
                    or selection["selected_node_id"] != metrics["selected_node_id"]
                    or metrics["selected_node_id"] not in {n["node_id"] for n in tree["nodes"]}):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def begin_question(directory):
    # An interrupted overwrite must never retain a previous completion marker.
    (Path(directory) / "complete.json").unlink(missing_ok=True)


def finish_question(directory, method, ex_id, budgets):
    save_json(Path(directory) / "complete.json",
              {"method": method, "example_id": ex_id, "budgets": sorted(budgets)})
