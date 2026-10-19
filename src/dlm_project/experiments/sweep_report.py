"""Offline tables and paired statistics for milestone sweeps (no model loading).

Completed predictions are the source of truth. Main comparisons use the
intersection of example IDs completed under every requested condition, while
observed summaries retain all successfully completed generations.
"""

import csv
import io
import json
import math
from pathlib import Path
from statistics import mean, median

from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.manifests import digest


def wilson_interval(correct, count):
    """Two-sided Wilson 95% interval, including all-correct/all-incorrect cases."""
    if not count:
        return None, None
    z = 1.959963984540054
    p, denominator = correct / count, 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def percentile(values, fraction):
    """Linear interpolation between sorted observations (NumPy's default)."""
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def aggregate(rows, condition):
    """Summarize exactly the supplied sample; missing samples are never failures."""
    n = len(rows)
    correct = sum(row["exact"] for row in rows)
    lower, upper = wilson_interval(correct, n)
    def average(key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return mean(values) if values else None
    memory = [row["peak_gpu_memory_bytes"] for row in rows if row.get("peak_gpu_memory_bytes") is not None]
    latencies = [row["latency_seconds"] for row in rows]
    summary = {"condition": condition["id"], "method": condition["method"],
               "requested_search_budget": condition["search_budget"],
               "n": n, "correct": correct, "accuracy": correct / n if n else None,
               "accuracy_ci95_low": lower, "accuracy_ci95_high": upper,
               "parse_rate": average("parse_success"),
               "mean_search_nfes": average("search_nfes"),
               "mean_completion_nfes": average("completion_nfes"),
               "mean_total_nfes": average("total_nfes"),
               "median_latency_seconds": median(latencies) if n else None,
               "p95_latency_seconds": percentile(latencies, .95),
               "max_gpu_memory_bytes": max(memory) if memory else None}
    for key in ("cell_accuracy", "clues_preserved", "valid_sudoku", "selected_depth",
                "evaluated_children", "expanded_parents", "realized_branching_factor", "fallback_used"):
        summary["mean_" + key] = average(key)
    if any(row.get("implementation_version") for row in rows):
        for phase in ("tree_analysis", "simulation", "candidate_evaluation"):
            summary[f"mean_{phase}_nfes"] = average(f"{phase}_nfes")
            summary[f"total_{phase}_nfes"] = sum(row.get(f"{phase}_nfes", 0) for row in rows)
            summary[f"mean_{phase}_latency_seconds"] = average(f"{phase}_latency_seconds")
    return summary


def load_records(directory, config, manifest):
    """Read flushed JSONL and reject inconsistent comparison identities."""
    examples = {row["example_id"]: row for row in manifest["examples"]}
    conditions = {row["id"]: row for row in config["conditions"]}
    selected = set(config["run"]["example_ids"])
    if digest(manifest) != config["run"]["manifest_hash"]:
        raise ValueError("Saved manifest hash does not match the run configuration.")
    rows, seen, identities, warnings = [], set(), {}, []
    text = (directory / "predictions.jsonl").read_text()
    lines = text.splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            if line_number == len(lines) and not text.endswith("\n"):
                warnings.append("Ignored an incomplete final JSONL record; earlier completed predictions are retained. "
                                "The raw file was not modified.")
                break
            raise ValueError(f"Invalid predictions.jsonl line {line_number}; retain complete lines before reporting.") from error
        key = (row["example_id"], row["condition"])
        if key in seen or key[0] not in selected or key[1] not in conditions:
            raise ValueError(f"Duplicate or unexpected prediction: {key}")
        example, condition = examples[key[0]], conditions[key[1]]
        expected_seed = (config["decoding"]["seed"] + example["position"]) % 2**63
        if (row["seed"] != expected_seed or row["content_hash"] != example["content_hash"]
                or row["prompt_hash"] != digest(row["prompt"])
                or row["requested_search_budget"] != condition["search_budget"]
                or row["method"] != condition["method"]
                or row["total_nfes"] != row["search_nfes"] + row["completion_nfes"]):
            raise ValueError(f"Inconsistent identity or NFE accounting: {key}")
        if config.get("report_schema_version", 1) >= 2:
            if (row.get("base_prompt_hash") != digest(row["base_prompt"])
                    or row.get("effective_prompt_hash") != digest(row["prompt"])
                    or row.get("prompt_template") != condition.get("prompt_template")
                    or row["search_nfes"] != sum(row.get(f"{phase}_nfes", 0) for phase in
                        ("tree_analysis", "simulation", "candidate_evaluation"))):
                raise ValueError(f"Inconsistent effective prompt or phase accounting: {key}")
            if row.get("decomposition") and row["decomposition"]["effective_prompt_hash"] != row["effective_prompt_hash"]:
                raise ValueError(f"Decomposition prompt hash differs: {key}")
            identity = (row["base_prompt_hash"], row["seed"], row["model_revision"])
        else:
            identity = (row["prompt_hash"], row["seed"], row["model_revision"])
        if key[0] in identities and identity != identities[key[0]]:
            raise ValueError(f"Conditions do not share a prompt, seed and model: {key[0]}")
        identities[key[0]] = identity
        seen.add(key)
        rows.append(row)
    return rows, warnings


def summarize(config, state, rows):
    condition_ids = [condition["id"] for condition in config["conditions"]]
    completed = {(row["example_id"], row["condition"]) for row in rows}
    common = [example for example in config["run"]["example_ids"]
              if all((example, condition) in completed for condition in condition_ids)]
    paired = [row for row in rows if row["example_id"] in common]
    expected = len(config["run"]["example_ids"]) * len(condition_ids)
    status = state.get("status", "incomplete")
    if status == "completed" and len(rows) != expected:
        status = "incomplete"
    observed = [aggregate([row for row in rows if row["condition"] == c["id"]], c)
                for c in config["conditions"]]
    conditions = []
    for condition, available in zip(config["conditions"], observed):
        result = aggregate([row for row in paired if row["condition"] == condition["id"]], condition)
        result["available_n"] = available["n"]
        conditions.append(result)
    return {"task": config["task"], "status": status, "complete": status == "completed",
            "evaluation_variant": config["evaluation_variant"],
            "requested_examples": len(config["run"]["example_ids"]),
            "expected_predictions": expected, "completed_predictions": len(rows),
            "common_n": len(common), "common_example_ids": common,
            "comparison_sample": "intersection completed under every condition",
            "conditions": conditions, "observed_conditions": observed,
            "completed_prediction_nfes": sum(row["total_nfes"] for row in rows),
            "incomplete_prediction_nfes": state.get("active_prediction_model_calls", 0),
            "setup": {key: state.get(key) for key in ("model_load_count", "model_loading_seconds",
                                                       "warmup_model_calls", "warmup_seconds")},
            "error": state.get("error")}


def write_csv(path, rows, default_fields=()):
    """Atomic flat CSV; nested inspection fields are serialized as JSON cells."""
    fields = list(dict.fromkeys([*default_fields, *(key for row in rows for key in row)]))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(buffer.getvalue())
    temporary.replace(path)


def _number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _markdown(config, summary, rows, directory):
    lines = ["# Milestone 1 decoder sweep", "",
             f"Task: **{config['task']}** · profile: **{config['sweep']['profile']}** · status: **{summary['status']}**.", "",
             f"Evaluator: **{config['evaluation_variant']}**. " +
             ("Paper-based reconstruction; author-code parity is unverified. Decomposition is part of the method comparison."
              if config.get("report_schema_version", 1) >= 2 else "This is not a full-rollout MEDAL reproduction."), "",
             f"Completed {summary['completed_predictions']}/{summary['expected_predictions']} predictions. "
             f"Every main comparison uses **n={summary['common_n']} common examples** "
             f"out of {summary['requested_examples']} selected examples.", ""]
    if config.get("implementation_version") in {
            "medal_reconstruction_v2", "medal_reconstruction_v3_swd"}:
        lines += ["Immediate-child accounting: search = one root forward + one forward per evaluated child. "
                  "Stability-aware runs additionally recompute each active parent distribution for temporal KL. "
                  "Candidate ranking reuses cached rewards and adds no forwards. Final completion is additional "
                  "and starts with a fresh forward. Only completed child evaluations update tree visits.", ""]
    elif config.get("report_schema_version", 1) >= 2:
        lines += ["Full-rollout accounting: search = tree analysis + all branch simulations + candidate evaluation. "
                  "Final completion is additional and starts with a fresh forward. Interrupted simulation calls count "
                  "against the budget; their unfinished rewards are never backed up. Budget sweeps are ablations.", ""]
    if not summary["complete"]:
        lines += ["**Incomplete run:** completed records are retained. Execution failures are not scored as wrong answers. "
                  "Examples missing any condition are excluded from paired comparisons.", ""]
    if summary["error"]:
        lines += [f"Recorded error: {_cell(summary['error'])}", ""]
    for warning in summary.get("warnings", []):
        lines += [f"**Recovery notice:** {warning}", ""]
    lines += ["## Settings and accounting", "",
              f"Model: `{config['model']['model_path']}`; frozen identity: "
              f"`{config['run'].get('model', {}).get('revision', 'see config.json')}`.", "",
              f"Data/row IDs/order/content hashes: [frozen manifest](manifest.json). "
              f"All settings and package versions: [resolved config](config.json).", "",
              "```json", json.dumps({"generation": config["generation"], "decoding": config["decoding"],
                                     "budgets": config["sweep"]["budgets"], "setup_cost": summary["setup"]}, indent=2), "```", "",
              ("- Total NFEs = root analysis + evaluated children + any stability-reference forwards + fresh final completion."
               if config.get("implementation_version") in {
                   "medal_reconstruction_v2", "medal_reconstruction_v3_swd"} else
               "- Total NFEs = tree analysis + branch simulations + candidate evaluation + fresh final completion."
               if config.get("report_schema_version", 1) >= 2 else
               "- Total NFEs = search NFEs + completion NFEs. Search includes root and each evaluated child; first completion reuses cached analysis."),
              "- Budget mode ignores the candidate quota, closes target-depth nodes, and stops at the NFE cap or exhausted frontier. "
              "Collected evaluated candidates are ranked; when none exists, fallback completes the deepest evaluated state (or root).",
              "- Realized branching = evaluated children / distinct parents with evaluated children, only for MCTS.",
              "- Decoding latency includes prompt encoding, search, completion and trace capture during those stages; CUDA is synchronized. "
              "Loading, warm-up, scoring and plotting are excluded. Setup forwards/time are listed separately above.",
              "- Peak memory is peak allocated CUDA memory during each prediction, including resident weights. CPU memory is not substituted.",
              "- Accuracy is exact match; parsing failures count as incorrect. Intervals are Wilson 95%, not evidence of equivalence. "
              "p95 latency uses linear interpolation. No reference answer is provided to the decoder.",
              "- Reference solutions appear only in scoring/inspection artifacts. Entropy is summed over remaining masks; "
              "reward also includes removal of the committed mask. Context gain excludes that removed-position entropy.", "",
              "## Paired results", "",
              "| Condition | Paired n | Available n | Accuracy [95% CI] | Parse rate | Mean total NFEs | Median / p95 seconds | Peak GiB |",
              "|---|---:|---:|---|---:|---:|---:|---:|"]
    for item in summary["conditions"]:
        memory = item["max_gpu_memory_bytes"]
        interval = f"{_number(item['accuracy'])} [{_number(item['accuracy_ci95_low'])}, {_number(item['accuracy_ci95_high'])}]"
        lines.append(f"| {item['condition']} | {item['n']} | {item['available_n']} | {interval} | "
                     f"{_number(item['parse_rate'])} | {_number(item['mean_total_nfes'], 1)} | "
                     f"{_number(item['median_latency_seconds'])} / {_number(item['p95_latency_seconds'])} | "
                     f"{_number(memory / 2**30 if memory is not None else None)} |")
    lines += ["", "[Flat per-example metrics](results.csv) · [Paired summary](summary.csv) · "
              "[Summary plus all available observations](summary.json) · [Raw predictions](predictions.jsonl)", "",
              "## Plots", "", f"Comparison plots use the same n={summary['common_n']} paired examples.", ""]
    for path in sorted((directory / "plots").glob("*.png")):
        relative = path.relative_to(directory).as_posix()
        lines += [f"![{path.stem}]({relative})", "", f"[PDF]({Path(relative).with_suffix('.pdf').as_posix()})", ""]
    if config["task"] == "gsm8k":
        lines += ["## Per-example predictions", "", "Full question/response text is in [the inspection table](gsm8k_examples.csv).", "",
                  "| Example | Condition | Parsed | Reference | Correct | NFEs | Seconds | Trace |", "|---|---|---|---|---:|---:|---:|---|"]
        for row in rows:
            lines.append(f"| {row['example_id']} | {row['condition']} | {_cell(row['predicted_answer'])} | "
                         f"{_cell(row['reference_answer'])} | {int(row['exact'])} | {row['total_nfes']} | "
                         f"{row['latency_seconds']:.3f} | [JSONL]({row['trace_file']}) |")
        lines.append("")
    else:
        lines += ["## Sudoku inspection", "", "Each figure includes the puzzle, reference, and every condition. "
                  "Invalid and incomplete outputs are retained visibly; missing conditions are labeled.", ""]
    for folder, title in (("examples", "Example figures"), ("inspection", "Search traces: first three selected examples")):
        paths = sorted((directory / folder).glob("*.png"))
        if paths:
            lines += [f"### {title}", ""]
            for path in paths:
                relative = path.relative_to(directory).as_posix()
                lines.append(f"- [{path.stem}]({relative}) ([PDF]({Path(relative).with_suffix('.pdf').as_posix()}))")
            lines.append("")
    lines += ["All cycle events are saved in `traces/`; no vocabulary logits are saved. "
              "Inspection plots are diagnostic and can include individually completed records outside the paired sample.", ""]
    return "\n".join(lines)


def generate_report(directory, *, plots=True):
    """Regenerate only from saved artifacts, optionally skipping expensive plots."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    state = json.loads((directory / "run_state.json").read_text())
    rows, warnings = load_records(directory, config, manifest)
    if warnings and state["status"] == "completed":
        state = {**state, "status": "incomplete"}
    summary = summarize(config, state, rows)
    summary["warnings"] = warnings
    write_csv(directory / "results.csv", rows, ("example_id", "condition"))
    save_json(directory / "summary.json", summary)
    if config["task"] == "gsm8k":
        examples = {item["example_id"]: item for item in manifest["examples"]}
        inspection = [{"example_id": row["example_id"], "question": examples[row["example_id"]]["question"],
                       **{key: row[key] for key in ("condition", "raw_output", "predicted_answer", "reference_answer",
                                                   "parse_success", "exact", "total_nfes", "latency_seconds")}}
                      for row in rows]
        write_csv(directory / "gsm8k_examples.csv", inspection, ("example_id", "condition"))
    if plots:
        from dlm_project.experiments.sweep_plots import save_plots
        save_plots(directory, config, manifest, rows, summary)
        if (state["status"] in ("reporting", "report_failed") and not warnings
                and len(rows) == summary["expected_predictions"]):
            state.update(status="completed")
            state.pop("error", None)
            save_json(directory / "run_state.json", state)
            summary.update(status="completed", complete=True, error=None)
            save_json(directory / "summary.json", summary)
    write_csv(directory / "summary.csv", [{"run_status": summary["status"], "task": config["task"], **item}
                                           for item in summary["conditions"]])
    temporary = directory / "report.md.tmp"
    temporary.write_text(_markdown(config, summary, rows, directory))
    temporary.replace(directory / "report.md")
    return summary
