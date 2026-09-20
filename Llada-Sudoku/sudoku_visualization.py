"""Display generated Sudoku text without repairing it or shifting partial rows."""

import re
from pathlib import Path

from sudoku_evaluation import is_valid_sudoku, preserves_clues, score_prediction


def prediction_display(record):
    """Return display rows, or None when raw text has no unambiguous row layout.

    Reparse raw output so older saved records receive the same validation.
    Only a successfully parsed answer may be reshaped into nine rows.
    """
    output = record.get("raw_output", record.get("prediction", ""))
    score = score_prediction(output, record["solution"])
    prediction = score["prediction"]
    if score["parse_success"]:
        issues = []
        if not preserves_clues(record["puzzle"], prediction):
            issues.append("clue conflict")
        if not is_valid_sudoku(prediction):
            issues.append("Sudoku rules")
        return {
            "rows": [list(prediction[i:i + 9]) for i in range(0, 81, 9)],
            "title": "Predicted solution\n" + (
                "Invalid: " + ", ".join(issues) if issues else "Valid; clues preserved"
            ),
            "color": "#b91c1c" if issues else "#166534",
            "parsed": True,
            "raw_output": output,
        }

    count = score["generated_digit_count"]
    label = f"Incomplete: {count}/81 digits" if count < 81 else f"Unparseable: {count} digits"
    lines = output.strip().splitlines()
    rows = None
    # Keep each original line at its original row. Never carry digits over to
    # another row, truncate extra digits, or pad the answer used for scoring.
    if lines and len(lines) <= 9 and all(re.fullmatch(r"[0-9\s]*", line) for line in lines):
        candidate = [re.findall(r"[0-9]", line) for line in lines]
        if all(len(row) <= 9 for row in candidate):
            rows = candidate
    return {
        "rows": rows, "title": f"Raw response\n{label}", "color": "#b45309",
        "parsed": False, "raw_output": output,
    }


def _draw_grid(ax, rows, title, color="black", puzzle=None, hide_zeros=False):
    ax.set_title(title, fontsize=13, pad=12, color=color)
    ax.set_xlim(0, 9)
    ax.set_ylim(9, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for i in range(10):
        width = 2 if i % 3 == 0 else 0.5
        ax.axhline(i, color="black", linewidth=width)
        ax.axvline(i, color="black", linewidth=width)
    for r, row in enumerate(rows):
        for c, digit in enumerate(row):
            if hide_zeros and digit == "0":
                continue
            conflict = puzzle is not None and puzzle[r * 9 + c] not in ("0", digit)
            ax.text(c + 0.5, r + 0.5, digit, ha="center", va="center", fontsize=14,
                    color="#b91c1c" if conflict else "black")


def plot_prediction(record):
    """Plot the puzzle, reference, and unchanged model answer; return the figure."""
    import matplotlib.pyplot as plt

    display = prediction_display(record)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, key, title in zip(axes[:2], ("puzzle", "solution"), ("Original board", "Actual solution")):
        board = record[key]
        _draw_grid(ax, [list(board[i:i + 9]) for i in range(0, 81, 9)], title, hide_zeros=True)
    if display["rows"] is not None:
        _draw_grid(axes[2], display["rows"], display["title"], display["color"], puzzle=record["puzzle"])
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
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_prediction(record)
    try:
        FigureCanvasAgg(fig)
        temporary = path.with_suffix(".png.tmp")
        fig.savefig(temporary, format="png", dpi=150, bbox_inches="tight")
        temporary.replace(path)
    finally:
        plt.close(fig)
