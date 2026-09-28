"""Display generated Sudoku text without repairing it or shifting partial rows."""

import re
from pathlib import Path

from dlm_project.tasks.sudoku.evaluation import answer_body, box_size, is_valid_sudoku, preserves_clues, score_prediction


def prediction_display(record):
    """Return display rows, or None when raw text has no unambiguous row layout.

    Reparse raw output so older saved records receive the same validation.
    Only a successfully parsed answer may be reshaped into a complete grid.
    """
    output = record.get("raw_output", record.get("prediction", ""))
    size = record.get("board_size", 9)
    score = score_prediction(output, record["solution"], size)
    prediction = score["prediction"]
    if score["parse_success"]:
        issues = []
        if not preserves_clues(record["puzzle"], prediction, size):
            issues.append("clue conflict")
        if not is_valid_sudoku(prediction, size):
            issues.append("Sudoku rules")
        return {
            "rows": [list(prediction[i:i + size]) for i in range(0, size ** 2, size)],
            "title": "Predicted solution\n" + (
                "Invalid: " + ", ".join(issues) if issues else "Valid; clues preserved"
            ),
            "color": "#b91c1c" if issues else "#166534",
            "parsed": True,
            "raw_output": output,
        }

    body = answer_body(output, size)
    count = len(re.findall(r"[0-9]", body or ""))
    label = f"Incomplete: {count}/{size ** 2} digits" if count < size ** 2 else f"Unparseable: {count} digits"
    if body is None:
        label = "Unparseable: expected one <answer> block"
    lines = (body or "").strip().splitlines()
    rows = None
    # Keep each original line at its original row. Never carry digits over to
    # another row, truncate extra digits, or pad the answer used for scoring.
    if lines and len(lines) <= size and all(re.fullmatch(r"[0-9\s]*", line) for line in lines):
        candidate = [re.findall(r"[0-9]", line) for line in lines]
        if all(len(row) <= size for row in candidate):
            rows = candidate
    return {
        "rows": rows, "title": f"Raw response\n{label}", "color": "#b45309",
        "parsed": False, "raw_output": output,
    }


def _draw_grid(ax, rows, title, color="black", puzzle=None, hide_zeros=False, board_size=9):
    box = box_size(board_size)
    ax.set_title(title, fontsize=13, pad=12, color=color)
    ax.set_xlim(0, board_size)
    ax.set_ylim(board_size, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for i in range(board_size + 1):
        width = 2 if i % box == 0 else 0.5
        ax.axhline(i, color="black", linewidth=width)
        ax.axvline(i, color="black", linewidth=width)
    for r, row in enumerate(rows):
        for c, digit in enumerate(row):
            if hide_zeros and digit == "0":
                continue
            conflict = puzzle is not None and puzzle[r * board_size + c] not in ("0", digit)
            ax.text(c + 0.5, r + 0.5, digit, ha="center", va="center", fontsize=14,
                    color="#b91c1c" if conflict else "black")


def plot_prediction(record):
    """Plot the puzzle, reference, and unchanged model answer; return the figure."""
    import matplotlib.pyplot as plt

    display = prediction_display(record)
    size = record.get("board_size", 9)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, key, title in zip(axes[:2], ("puzzle", "solution"), ("Original board", "Actual solution")):
        board = record[key]
        _draw_grid(ax, [list(board[i:i + size]) for i in range(0, size ** 2, size)],
                   title, hide_zeros=True, board_size=size)
    if display["rows"] is not None:
        _draw_grid(axes[2], display["rows"], display["title"], display["color"],
                   puzzle=record["puzzle"], board_size=size)
    else:
        axes[2].axis("off")
        axes[2].set_title(display["title"], color=display["color"], fontsize=13)
        axes[2].text(0, 0.95, display["raw_output"] or "(empty response)",
                     transform=axes[2].transAxes, va="top", fontsize=10,
                     fontfamily="monospace", wrap=True)
    fig.text(0.5, 0.025, "Red digits conflict with given clues. Missing cells stay blank; no solver repairs are applied.",
             ha="center", fontsize=11)
    if "index" in record:
        fig.suptitle(f"Puzzle Index {record['index']}", fontsize=16)
    fig.tight_layout(rect=(0, 0.06, 1, 0.90 if "index" in record else 1))
    return fig


def save_prediction_plot(record, path):
    """Save every answer, including failed parsing, without an interactive backend."""
    _save_figure(plot_prediction(record), path)


def plot_paired_prediction(baseline, bgpo):
    """Show both unmodified outputs alongside the shared 4x4 puzzle/reference."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, key, title in zip(axes[:2], ("puzzle", "solution"), ("Puzzle", "Reference")):
        board = baseline[key]
        _draw_grid(ax, [list(board[i:i + 4]) for i in range(0, 16, 4)], title,
                   hide_zeros=True, board_size=4)
    for ax, record, label in zip(axes[2:], (baseline, bgpo), ("Original LLaDA", "BGPO")):
        display = prediction_display(record)
        title = label + "\n" + display["title"].split("\n", 1)[-1]
        if display["rows"] is not None:
            _draw_grid(ax, display["rows"], title, display["color"],
                       puzzle=record["puzzle"], board_size=4)
        else:
            ax.axis("off")
            ax.set_title(title, color=display["color"], fontsize=11)
            # Full text remains in JSONL; wrap a bounded preview to keep the
            # comparison figure readable even after malformed generation.
            import textwrap
            preview = display["raw_output"] or "(empty response)"
            if len(preview) > 500:
                preview = preview[:500] + "\n[full response in paired_predictions.jsonl]"
            ax.text(0, 0.95, textwrap.fill(preview, width=38), transform=ax.transAxes,
                    va="top", fontsize=8, fontfamily="monospace")
    fig.suptitle(f"4×4 Sudoku | Test puzzle {baseline['source_index']}", fontsize=16)
    fig.text(0.5, 0.025, "Red digits conflict with clues. Outputs are never repaired.", ha="center")
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    return fig


def save_paired_prediction_plot(baseline, bgpo, path):
    _save_figure(plot_paired_prediction(baseline, bgpo), path)


def _save_figure(fig, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        FigureCanvasAgg(fig)
        temporary = path.with_suffix(".png.tmp")
        fig.savefig(temporary, format="png", dpi=150, bbox_inches="tight")
        temporary.replace(path)
    finally:
        plt.close(fig)
