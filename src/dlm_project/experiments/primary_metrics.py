"""Offline accuracy/compute comparisons for the five GSM8K decoders."""

import json
from pathlib import Path
from statistics import mean, median

from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.sweep_report import percentile, wilson_interval, write_csv


def summarize_primary(records):
    """Aggregate completed predictions; missing execution results are excluded.

    decoding_steps counts all final-output trajectories (all five for Best-of-5),
    excluding search rollouts. mcts_simulations counts fully evaluated tree edges,
    independently of MCTS cycles and terminal candidate evaluation rollouts.
    """
    count = len(records)
    correct = sum(bool(row["exact_match"]) for row in records)
    low, high = wilson_interval(correct, count)

    def average(key):
        values = [row[key] for row in records if row.get(key) is not None]
        return mean(values) if values else None

    calls = [row.get("total_nfes", row.get("total_model_calls")) for row in records]
    times = [row["latency_seconds"] for row in records]
    invocations = sum(row.get("search_invocations", 0) for row in records)
    simulations = [row["mcts_simulations"] for row in records if row.get("mcts_simulations") is not None]
    result = {"evaluated_n": count, "accuracy_percent": 100 * correct / count if count else None,
              "accuracy_ci95_low": low, "accuracy_ci95_high": high,
              "mean_total_nfes": mean(calls) if count else None,
              "mean_latency_seconds": mean(times) if count else None,
              "median_latency_seconds": median(times) if count else None,
              "p95_latency_seconds": percentile(times, .95),
              "mean_decoding_steps": average("decoding_steps"),
              "mean_selected_output_steps": average("selected_output_steps"),
              "total_mcts_simulations": sum(simulations) if len(simulations) == count else None,
              "mean_mcts_simulations": average("mcts_simulations"),
              "total_search_invocations": invocations,
              "mean_search_invocations": average("search_invocations"),
              "mean_simulations_started": average("simulations_started"),
              "mean_simulations_interrupted": average("simulations_interrupted"),
              "mean_simulation_decoding_steps": average("simulation_decoding_steps"),
              "mean_simulations_per_invocation": sum(simulations) / invocations if invocations and len(simulations) == count else (0.0 if count and not invocations else None),
              "mean_average_search_depth": average("average_search_depth"),
              "mean_expanded_nodes": average("expanded_nodes"),
              "mean_selected_depth": average("selected_depth"),
              "mean_max_depth": average("max_depth"),
              "mean_realized_branching_factor": average("realized_branching_factor"),
              "fallback_rate": average("fallback_used")}
    result["mean_state_evaluations"] = average("state_evaluations")
    result["total_signal_extra_nfes"] = sum(row.get("signal_extra_nfes", 0) for row in records)
    for phase in ("search", "tree_analysis", "simulation", "candidate_evaluation", "completion"):
        result[f"mean_{phase}_nfes"] = average(f"{phase}_model_calls")
    return result


def _load_run(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    records, seen = [], set()
    text = (directory / "predictions.jsonl").read_text()
    for index, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(text.splitlines()) - 1 and not text.endswith("\n"):
                break
            raise
        if row["example_id"] in seen:
            raise ValueError("Repeated example in a decoder run.")
        seen.add(row["example_id"])
        row.setdefault("total_nfes", row["total_model_calls"])
        if row["total_nfes"] != row["total_model_calls"]:
            raise ValueError("Saved NFE totals do not reconcile.")
        is_search = row["method"] == "mcts_confidence"
        if is_search and all(key in row for key in ("search_model_calls", "completion_model_calls")):
            if row["total_nfes"] != row["search_model_calls"] + row["completion_model_calls"]:
                raise ValueError("Search/completion NFE accounting does not reconcile.")
        # Historical counts may be recovered from compact events, never from NFEs.
        if "decoding_steps" not in row or "mcts_simulations" not in row:
            trace_path = directory / row.get("trace_file", "trace-not-saved")
            events = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()] if trace_path.is_file() else None
            if events is not None:
                row.setdefault("decoding_steps", sum(event["stage"] == "commit" or
                    (event["stage"] == "rollout_step" and event.get("phase") == "completion") for event in events))
                if is_search and row.get("implementation_version") == "medal_reconstruction_v1":
                    row.setdefault("mcts_simulations", sum(event["stage"] == "evaluation" for event in events))
            if not is_search:
                row.setdefault("mcts_simulations", 0)
        row.setdefault("search_invocations", 1 if is_search else 0)
        records.append(row)
    return config, metrics, records


def generate_primary_report(run_dirs, output_dir, *, expected_conditions=None):
    """Build a paired report from explicitly selected run directories, offline.

    No directory discovery, model construction, or dataset download occurs.
    Comparison plots use only examples completed under every selected condition.
    """
    if not run_dirs:
        raise ValueError("Select at least one run directory.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = {label: _load_run(path) for label, path in run_dirs.items()}
    first_config = next(iter(loaded.values()))[0]
    requested = first_config["run"]["example_ids"]
    missing = [label for label in (expected_conditions or run_dirs) if label not in run_dirs]
    common = [example_id for example_id in requested if all(any(row["example_id"] == example_id
              for row in data[2]) for data in loaded.values())]
    if missing:
        common = []
    warnings, identities, summaries, flat = [], {}, [], []
    settings, gpu_names = set(), set()
    for label, (config, metrics, records) in loaded.items():
        if config["run"]["example_ids"] != requested:
            raise ValueError("Selected runs must request the same ordered examples.")
        settings.add((config["generation"]["gen_length"], config["generation"]["steps"],
                      config["generation"]["block_length"], config["generation"]["temperature"]))
        paired = [row for row in records if row["example_id"] in common]
        for row in paired:
            if row.get("gpu_name"):
                gpu_names.add(row["gpu_name"])
            identity = tuple(row.get(key) for key in ("model_revision", "dataset_revision", "content_hash", "seed"))
            previous = identities.setdefault(row["example_id"], identity)
            if previous != identity:
                raise ValueError("Compared examples must share model, data, content and seed.")
            flat.append({"condition": label, "run_dir": str(run_dirs[label]), **row})
        prompts = {row["example_id"]: row.get("effective_prompt_hash", row["prompt_hash"]) for row in paired}
        mcts = config["decoding"].get("mcts", {})
        item = {"condition": label, "method": config["decoding"]["method"],
                "status": metrics["status"], "available_n": len(records),
                "requested_simulation_budget": mcts.get("max_simulations"),
                "requested_nfe_budget": mcts.get("max_model_calls"),
                "run_dir": str(run_dirs[label]), "prompt_hashes": prompts,
                **summarize_primary(paired)}
        item["parse_rate"] = mean(row["parse_success"] for row in paired) if paired else None
        item["batch_size"] = config.get("batch_size", 1)
        summaries.append(item)
    if len(settings) > 1:
        warnings.append("Generation settings differ across selected runs; this comparison does not isolate the decoding method.")
    if len(gpu_names) > 1:
        warnings.append("GPU hardware differs across selected runs; latency is not a hardware-matched comparison.")
    if any(item["prompt_hashes"] != summaries[0]["prompt_hashes"] for item in summaries[1:]):
        warnings.append("Effective prompts differ (for example MEDAL decomposition); interpret accuracy differences as whole-pipeline comparisons.")
    complete = not missing and all(item["status"] == "completed" and item["available_n"] == len(requested) for item in summaries)
    if missing:
        warnings.append("Conditions not started: " + ", ".join(missing))
    if not complete:
        warnings.append("Incomplete runs: missing predictions are excluded, not scored as incorrect.")
    data = {"dataset": "GSM8K", "complete": complete, "requested_n": len(requested),
            "common_n": len(common), "common_example_ids": common,
            "comparison_sample": "intersection completed under every selected condition",
            "conditions": summaries, "warnings": warnings}
    data["missing_conditions"] = missing
    save_json(output_dir / "summary.json", data)
    save_json(output_dir / "plot_data.json", data)
    save_json(output_dir / "runs.json", {label: str(path) for label, path in run_dirs.items()})
    write_csv(output_dir / "results.csv", flat, ("condition", "example_id"))
    write_csv(output_dir / "summary.csv", [{key: value for key, value in item.items() if key != "prompt_hashes"} for item in summaries])
    _plots(data, output_dir / "plots")
    lines = ["# GSM8K accuracy and compute", "", f"Paired evaluation count: **{len(common)}/{len(requested)}**.", "",
             "| Method | Accuracy % (Wilson 95% CI) | Mean total NFEs | Mean seconds | Mean decoding steps | Mean MCTS simulations |",
             "|---|---:|---:|---:|---:|---:|"]
    def number(value):
        return "—" if value is None else f"{value:.3f}"
    for item in summaries:
        low, high = item["accuracy_ci95_low"], item["accuracy_ci95_high"]
        ci = "—" if low is None else f"{100 * low:.1f}–{100 * high:.1f}"
        values = [number(item[key]) for key in ("mean_total_nfes", "mean_latency_seconds", "mean_decoding_steps", "mean_mcts_simulations")]
        lines.append(f"| {item['condition']} | {number(item['accuracy_percent'])} ({ci}) | " + " | ".join(values) + " |")
    lines += ["", *warnings, "", "NFEs count every forward, including tree analysis, full simulations, terminal candidate evaluation, and final completion. "
              "Batch size is one, so state evaluations equal forward calls. Model loading is recorded separately. "
              "Latency includes tokenization, decoding, uncertainty reductions and trace writing, and excludes model loading and report generation.", "",
              "One MCTS simulation is one fully completed child rollout and backup. Interrupted simulations, candidate-evaluation rollouts and MCTS cycles are separate counts. "
              "A simulation cap limits child rollouts; terminal candidate evaluation can add NFEs after that cap. "
              "Decoding steps count output-generation commitments, summed over all five Best-of-5 candidates; internal search rollout steps are saved separately. "
              "Average search depth averages evaluated child depths. Expanded nodes counts distinct evaluated child states, excluding the root.", "",
              "Entropy is measured in nats and margin is the top-two probability gap over all unresolved output positions. "
              "Both reuse native full-vocabulary predictions and add zero model forwards; their arithmetic and trace overhead remains in latency.", ""]
    lines += ["| Search condition | Mean depth of evaluated children | Mean unique evaluated children | Mean selected / maximum depth | Mean simulations per invocation | Fallback rate |",
              "|---|---:|---:|---:|---:|---:|"]
    for item in summaries:
        if item["method"] == "mcts_confidence":
            lines.append(f"| {item['condition']} | {number(item['mean_average_search_depth'])} | "
                f"{number(item['mean_expanded_nodes'])} | {number(item['mean_selected_depth'])} / {number(item['mean_max_depth'])} | "
                f"{number(item['mean_simulations_per_invocation'])} | {number(item['fallback_rate'])} |")
    lines.append("")
    for stem, label in (("accuracy_vs_simulations", "Accuracy versus completed MCTS simulations"),
                        ("accuracy_vs_nfes", "Accuracy versus total NFEs"),
                        ("accuracy_vs_latency", "Accuracy versus mean wall-clock time")):
        lines += [f"![{label}](plots/{stem}.png)", f"[PDF](plots/{stem}.pdf)", ""]
    (output_dir / "report.md").write_text("\n".join(lines))
    return data


def _plots(data, directory):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    directory.mkdir(parents=True, exist_ok=True)
    for key, stem, xlabel in (("mean_mcts_simulations", "accuracy_vs_simulations", "Mean completed MCTS simulations per example"),
                              ("mean_total_nfes", "accuracy_vs_nfes", "Mean total NFEs per example"),
                              ("mean_latency_seconds", "accuracy_vs_latency", "Mean end-to-end inference time (seconds)")):
        figure = Figure(figsize=(9, 5), layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        available = [item for item in data["conditions"] if item[key] is not None and item["accuracy_percent"] is not None]
        search = sorted([item for item in available if item["method"] == "mcts_confidence"], key=lambda item: item[key])
        if len(search) > 1:
            axis.plot([item[key] for item in search], [item["accuracy_percent"] for item in search], alpha=.25, color="#7c3aed")
        for item in available:
            y = item["accuracy_percent"]
            axis.errorbar(item[key], y, yerr=[[max(0, y - 100 * item["accuracy_ci95_low"])],
                                           [max(0, 100 * item["accuracy_ci95_high"] - y)]],
                          fmt="o", capsize=3, label=item["condition"])
        if not available:
            axis.text(.5, .5, "No comparable completed examples / recorded counts", ha="center", transform=axis.transAxes)
        else:
            axis.legend(fontsize=8, loc="best")
        axis.set(xlabel=xlabel, ylabel="Strict accuracy (%) · Wilson 95% CI", ylim=(-3, 103),
                 title=f"GSM8K · paired n={data['common_n']}" + (" · settings differ" if data["warnings"] else ""))
        axis.grid(alpha=.2)
        for extension in ("png", "pdf"):
            figure.savefig(directory / f"{stem}.{extension}", dpi=160)
        figure.clear()
