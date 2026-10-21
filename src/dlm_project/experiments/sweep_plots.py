"""Scientific PNG/PDF figures derived exclusively from saved sweep metrics."""

import json
import math
from pathlib import Path
import textwrap

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from dlm_project.experiments.artifacts import save_json

COLORS = {"baseline": "#64748b", "confidence": "#0f766e", "mcts_entropy": "#7c3aed"}


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(fig)
    for extension in ("png", "pdf"):
        target = path.with_suffix("." + extension)
        temporary = target.with_suffix(target.suffix + ".tmp")
        fig.savefig(temporary, format=extension, dpi=160, bbox_inches="tight")
        temporary.replace(target)
    fig.clear()


def _empty(ax):
    ax.text(.5, .5, "No examples completed under every condition yet", ha="center", va="center",
            transform=ax.transAxes, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def comparison_data(rows, summary):
    """Return the exact paired values used in all comparison plots."""
    ids = summary["common_example_ids"]
    conditions = [item["condition"] for item in summary["conditions"]]
    lookup = {(row["example_id"], row["condition"]): row for row in rows}
    return {"common_n": len(ids), "example_ids": ids, "condition_ids": conditions,
            "conditions": summary["conditions"],
            "correctness": [[int(lookup[(example, condition)]["exact"]) for condition in conditions]
                            for example in ids]}


def _tradeoff(data, xkey, xlabel):
    fig = Figure(figsize=(8, 5), layout="constrained")
    ax = fig.subplots()
    ax.set_title(f"Accuracy and compute • paired n={data['common_n']}")
    if not data["common_n"]:
        _empty(ax)
        return fig
    search = [item for item in data["conditions"] if item["method"] == "mcts_entropy"]
    if len(search) > 1:
        ax.plot([item[xkey] for item in search], [item["accuracy"] for item in search],
                color=COLORS["mcts_entropy"], alpha=.25, linewidth=1)
    annotations = {}
    for item in data["conditions"]:
        y = item["accuracy"]
        ax.errorbar(item[xkey], y, yerr=[[max(0, y - item["accuracy_ci95_low"])],
                                       [max(0, item["accuracy_ci95_high"] - y)]],
                    fmt={"baseline": "o", "confidence": "s", "mcts_entropy": "D"}[item["method"]],
                    capsize=4, color=COLORS[item["method"]], label=item["condition"])
        label = (f"cap {item['requested_search_budget']}" if item["method"] == "mcts_entropy"
                 else item["condition"])
        annotations.setdefault((item[xkey], y), []).append(label)
    for (x, y), labels in annotations.items():
        ax.annotate("\n".join(labels), (x, y), xytext=(7, -10 if y > .8 else 10),
                    textcoords="offset points", va="top" if y > .8 else "bottom", fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 1})
    ax.set(xlabel=xlabel, ylabel="Exact-match accuracy (Wilson 95% CI)", ylim=(-.04, 1.14))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.2)
    ax.margins(x=.18)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncols=3, fontsize=8)
    return fig


def _breakdown(data):
    fig = Figure(figsize=(9, 5), layout="constrained")
    ax = fig.subplots()
    ax.set_title(f"Where model forwards are spent • paired n={data['common_n']}")
    if not data["common_n"]:
        _empty(ax)
        return fig
    labels = data["condition_ids"]
    search = [item["mean_search_nfes"] for item in data["conditions"]]
    completion = [item["mean_completion_nfes"] for item in data["conditions"]]
    ax.bar(labels, search, label="Search (all initialization forwards)", color="#7c3aed")
    ax.bar(labels, completion, bottom=search, label="Completion", color="#0f766e")
    ax.set_ylabel("Mean actual NFEs per example")
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend()
    return fig


def _heatmap(data):
    n = data["common_n"]
    fig = Figure(figsize=(9, max(3.5, min(28, .27 * n + 2))), layout="constrained")
    ax = fig.subplots()
    ax.set_title(f"Correctness • paired n={n} • 1 correct, 0 incorrect (including parse failure)")
    if not n:
        _empty(ax)
        return fig
    ax.imshow(data["correctness"], cmap=ListedColormap(["#f4b4ad", "#83cebc"]),
              vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(data["condition_ids"])), data["condition_ids"], rotation=20)
    ax.set_yticks(range(n), data["example_ids"], fontsize=8)
    for row, values in enumerate(data["correctness"]):
        for column, value in enumerate(values):
            ax.text(column, row, str(value), ha="center", va="center", fontsize=8)
    return fig


def _diagnostics(data):
    fig = Figure(figsize=(10, 7), layout="constrained")
    axes = fig.subplots(2, 2).flat
    fig.suptitle(f"Search diagnostics • paired n={data['common_n']}")
    search = [item for item in data["conditions"] if item["method"] == "mcts_entropy"]
    metrics = [("mean_selected_depth", "Mean selected depth"),
               ("mean_evaluated_children", "Mean evaluated children (excludes root)"),
               ("mean_realized_branching_factor", "Mean realized branching factor"),
               ("mean_fallback_used", "Fallback rate")]
    for ax, (key, label) in zip(axes, metrics):
        ax.set(xlabel="Requested search-forward cap", ylabel=label)
        if not data["common_n"]:
            _empty(ax)
            continue
        budgets = [item["requested_search_budget"] for item in search]
        ax.plot(budgets, [item[key] for item in search], "o-", color=COLORS["mcts_entropy"])
        ax.set_xticks(budgets)
        ax.grid(alpha=.2)
        if key == "mean_fallback_used":
            ax.set_ylim(-.04, 1.04)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    return fig


def trace_data(events, selected_id):
    """Selected-path entropy and every evaluated edge, indexed by actual NFE."""
    root = next(event for event in events if event["stage"] == "initialization")
    evaluations = [event for event in events if event["stage"] == "evaluation"]
    edges = {event["child_id"]: event for event in evaluations}
    path = []
    seen = set()
    while selected_id != 0:
        if selected_id in seen:
            raise ValueError("Cycle in saved selected-node ancestry.")
        seen.add(selected_id)
        edge = edges[selected_id]
        path.append({"node_id": selected_id, "entropy": edge["entropy_after"]})
        selected_id = edge["parent_id"]
    path.append({"node_id": 0, "entropy": root["entropy"]})
    path.reverse()
    return {"selected_path": [{"depth": index, **item} for index, item in enumerate(path)],
            "evaluations": [{key: event[key] for key in ("child_id", "parent_id", "search_model_calls",
                                                         "reward", "context_gain")} for event in evaluations]}


def _trace_figure(data, title):
    fig = Figure(figsize=(14, 4.3), layout="constrained")
    axes = fig.subplots(1, 3)
    fig.suptitle(title)
    path, evaluations = data["selected_path"], data["evaluations"]
    axes[0].plot([item["depth"] for item in path], [item["entropy"] for item in path], "o-", color="#0f766e")
    axes[0].set(xlabel="Selected-path depth", ylabel="Remaining-mask entropy (nats)")
    for ax, key, label in zip(axes[1:], ("reward", "context_gain"),
                              ("One-step normalized entropy reward", "Context gain (nats; removal excluded)")):
        ax.scatter([item["search_model_calls"] for item in evaluations],
                   [item[key] for item in evaluations], s=12, color="#7c3aed", alpha=.65)
        ax.axhline(0, color="#64748b", linewidth=.8)
        ax.set(xlabel="Cumulative search NFEs", ylabel=label)
        if not evaluations:
            ax.text(.5, .5, "Root only; no evaluated actions", transform=ax.transAxes, ha="center")
    for ax in axes:
        ax.grid(alpha=.2)
    return fig


def _sudoku_figure(example, conditions, records):
    from dlm_project.tasks.sudoku.visualization import _draw_grid, prediction_display

    columns = min(4, len(conditions) + 2)
    rows = math.ceil((len(conditions) + 2) / columns)
    fig = Figure(figsize=(4 * columns, 4.5 * rows), layout="constrained")
    axes = fig.subplots(rows, columns, squeeze=False).flat
    fig.suptitle(f"4×4 Sudoku • {example['example_id']}\nRed digits conflict with clues; missing cells stay blank.")
    for key, title in (("puzzle", "Puzzle"), ("solution", "Reference (scoring only)")):
        board = example[key]
        _draw_grid(next(axes), [list(board[i:i+4]) for i in range(0, 16, 4)], title,
                   hide_zeros=True, board_size=4)
    for condition in conditions:
        ax = next(axes)
        record = records.get(condition["id"])
        if record is None:
            ax.axis("off")
            ax.set_title(condition["id"])
            ax.text(.5, .5, "Not completed", ha="center", transform=ax.transAxes)
            continue
        display = prediction_display({**example, **record, "board_size": 4})
        title = f"{condition['id']} • exact={int(record['exact'])}\n{display['title'].split(chr(10))[-1]}"
        if display["rows"] is not None:
            _draw_grid(ax, display["rows"], title, display["color"], puzzle=example["puzzle"], board_size=4)
        else:
            ax.axis("off")
            ax.set_title(title, color=display["color"], fontsize=11)
            preview = display["raw_output"] or "(empty response)"
            if len(preview) > 400:
                preview = preview[:400] + "\n[full text: predictions.jsonl]"
            ax.text(0, .95, textwrap.fill(preview, 36), transform=ax.transAxes,
                    va="top", fontsize=8, family="monospace")
    for ax in axes:
        ax.axis("off")
    return fig


def save_plots(directory, config, manifest, rows, summary):
    """Save five paired plots, task figures and first-three-example MCTS traces."""
    data = comparison_data(rows, summary)
    save_json(directory / "plot_data.json", data)
    output = directory / "plots"
    _save(_tradeoff(data, "mean_total_nfes", "Mean actual total NFEs (search + completion)"), output / "accuracy_vs_nfes")
    _save(_tradeoff(data, "median_latency_seconds", "Median decoding latency (seconds)"), output / "accuracy_vs_latency")
    _save(_breakdown(data), output / "nfe_breakdown")
    _save(_heatmap(data), output / "correctness_heatmap")
    _save(_diagnostics(data), output / "search_diagnostics")
    by_example = {key: {} for key in config["run"]["example_ids"]}
    for row in rows:
        by_example[row["example_id"]][row["condition"]] = row
    examples = {row["example_id"]: row for row in manifest["examples"]}
    if config["task"] == "sudoku":
        for example_id, records in by_example.items():
            example = examples[example_id]
            _save(_sudoku_figure(example, config["conditions"], records),
                  directory / "examples" / f"example_{example['position']:04d}")
    for example_id in config["run"]["example_ids"][:3]:
        for record in by_example[example_id].values():
            if record["method"] != "mcts_entropy":
                continue
            events = [json.loads(line) for line in (directory / record["trace_file"]).read_text().splitlines()]
            inspection = trace_data(events, record["selected_node_id"])
            target = directory / "inspection" / f"example_{record['manifest_position']:04d}_{record['condition']}"
            target.parent.mkdir(parents=True, exist_ok=True)
            save_json(target.with_suffix(".json"), inspection)
            _save(_trace_figure(inspection, f"{example_id} • {record['condition']} • {record.get('implementation_version') or 'historical one-step entropy'}"), target)
