"""Full, numbered E3 checkpoint trees, rendered offline without model imports.

Inspired by AB-MCTS Figure 7: https://arxiv.org/html/2503.04412v5#S4.F7.
Here a node is an evaluated partial token state, rather than a complete answer.
"""

from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from dlm_project.tasks.gsm8k.e3_analysis import load_question_checkpoints


METHODS = ("medal", "ab_medal")
COLOR_METRICS = {
    "immediate_reward": ("Immediate information gain (IG)", "immediate_ig"),
    "mean_reward": ("Backed-up mean Q-value", "mean_q"),
}
MAX_PNG_DIMENSION = 8192
_GOLD = "#c58a00"
_NEUTRAL = "#e4e7ec"
_NODE_SIZE = 22 ** 2  # Matplotlib marker area in points squared.


@dataclass
class _Tree:
    method: str
    nodes: dict
    root_id: int
    children: dict
    positions: dict
    leaf_count: int
    max_depth: int
    selected_id: int
    shortlist_ids: list
    selected_path: list
    exact_match: bool | None


def _validated_tree(data, *, method, example_id, budget):
    """Validate checkpoint identity/topology before computing any plot geometry."""
    tree, selection, metrics = data["tree"], data["selection"], data["metrics"]
    for key, expected in (("method", method), ("example_id", example_id), ("budget", budget)):
        if tree.get(key) != expected:
            raise ValueError(f"tree.json {key}={tree.get(key)!r}; expected {expected!r}.")
    for key, expected in (("method", method), ("example_id", example_id), ("search_budget", budget)):
        if key in metrics and metrics[key] != expected:
            raise ValueError(f"metrics.json {key} disagrees with the requested checkpoint.")
    if selection.get("requested_budget", budget) != budget:
        raise ValueError("selection.json requested_budget disagrees with the requested checkpoint.")

    rows = tree.get("nodes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("tree.json must contain a nonempty nodes list.")
    nodes, orders = {}, set()
    for row in rows:
        node_id = row.get("node_id")
        if type(node_id) is not int or node_id in nodes:
            raise ValueError(f"Node IDs must be unique integers; got {node_id!r}.")
        for field in ("depth", "evaluation_order"):
            if type(row.get(field)) is not int or row[field] < 0:
                raise ValueError(f"Node {node_id} needs a nonnegative integer {field}.")
        order = row["evaluation_order"]
        if order in orders or order >= budget:
            raise ValueError(f"Node {node_id}: evaluation_order must be unique and below budget {budget}.")
        if "parent_id" not in row or (row["parent_id"] is not None and type(row["parent_id"]) is not int):
            raise ValueError(f"Node {node_id} needs an integer parent_id, or null for the root.")
        for field in COLOR_METRICS:
            value = row.get(field)
            if field == "immediate_reward" and row["parent_id"] is None and value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Node {node_id} needs a finite {field}.")
        nodes[node_id] = row
        orders.add(order)

    roots = [n for n in nodes.values() if n["parent_id"] is None]
    if len(roots) != 1 or roots[0]["depth"] != 0 or roots[0]["evaluation_order"] != 0:
        raise ValueError("Expected exactly one root, with depth 0 and evaluation_order 0.")
    root_id = roots[0]["node_id"]
    nodes = dict(sorted(nodes.items(), key=lambda item: item[1]["evaluation_order"]))
    children = {node_id: [] for node_id in nodes}
    for node_id, row in nodes.items():
        parent_id = row["parent_id"]
        if parent_id is None:
            continue
        if parent_id not in nodes:
            raise ValueError(f"Node {node_id} has missing parent {parent_id}.")
        parent = nodes[parent_id]
        if row["depth"] != parent["depth"] + 1:
            raise ValueError(f"Node {node_id}: depth must equal parent depth + 1 (invalid topology).")
        if parent["evaluation_order"] >= row["evaluation_order"]:
            raise ValueError(f"Node {node_id}: parent must be evaluated before child (invalid topology).")
        # Only parent links between evaluated records count. child_ids may also
        # contain unevaluated work at the point where a budget stopped search.
        children[parent_id].append(node_id)

    selected_id = selection.get("selected_node_id")
    shortlist = selection.get("shortlist_ids")
    if type(selected_id) is not int or selected_id not in nodes:
        raise ValueError("selection.json selected_node_id is absent from the tree.")
    if (not isinstance(shortlist, list) or not shortlist
            or any(type(i) is not int or i not in nodes for i in shortlist)
            or len(set(shortlist)) != len(shortlist) or selected_id not in shortlist):
        raise ValueError("selection.json shortlist_ids must be distinct tree nodes including selected_node_id.")
    if selection.get("selected_depth") != nodes[selected_id]["depth"]:
        raise ValueError("selection.json selected_depth disagrees with selected_node_id.")
    exact_match = metrics.get("exact_match")
    if exact_match is not None and type(exact_match) is not bool:
        raise ValueError("metrics.json exact_match must be a boolean or null.")

    # Iterative traversals also handle long chains without Python recursion limits.
    traversal, pending = [], [root_id]
    while pending:
        node_id = pending.pop()
        traversal.append(node_id)
        pending.extend(reversed(children[node_id]))
    if len(traversal) != len(nodes):
        raise ValueError("Tree contains disconnected nodes or a cycle.")
    leaves = [i for i in traversal if not children[i]]
    spans = {node_id: (float(x), float(x)) for x, node_id in enumerate(leaves)}
    positions = {}
    for node_id in reversed(traversal):
        if children[node_id]:
            spans[node_id] = (spans[children[node_id][0]][0], spans[children[node_id][-1]][1])
        left, right = spans[node_id]
        positions[node_id] = ((left + right) / 2, -nodes[node_id]["depth"])
    selected_path, current = [], selected_id
    while nodes[current]["parent_id"] is not None:
        parent_id = nodes[current]["parent_id"]
        selected_path.append((parent_id, current))
        current = parent_id
    return _Tree(method, nodes, root_id, children, positions, len(leaves),
                 max(n["depth"] for n in nodes.values()), selected_id, shortlist,
                 selected_path, exact_match)


def _load_trees(results_dir, *, example_id, budget):
    trees = []
    for method in METHODS:
        question_dir = results_dir / method / example_id
        checkpoint_dir = question_dir / "checkpoints" / str(budget)
        missing = [checkpoint_dir / name for name in ("tree.json", "selection.json", "metrics.json")
                   if not (checkpoint_dir / name).is_file()]
        if missing:
            raise FileNotFoundError("Missing E3 checkpoint files; no other question/budget is substituted:\n"
                                    + "\n".join(str(path) for path in missing))
        try:
            data = load_question_checkpoints(question_dir, budgets=(budget,))[budget]
            trees.append(_validated_tree(data, method=method, example_id=example_id, budget=budget))
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"Invalid E3 checkpoint {checkpoint_dir}: {error}") from error
    return trees


def _comparison_figure(trees, *, example_id, budget, color_by):
    label, _ = COLOR_METRICS[color_by]
    values = [n[color_by] for tree in trees for n in tree.nodes.values() if n.get(color_by) is not None]
    limit = max((abs(value) for value in values), default=0.0) or 1.0
    norm = Normalize(vmin=-limit, vmax=limit)
    cmap = mpl.colormaps["RdBu_r"]
    leaf_count = max(tree.leaf_count for tree in trees)
    max_depth = max(tree.max_depth for tree in trees)

    # Dimensions in inches preserve actual node spacing across both methods.
    # The PDF grows to accommodate the complete tree; only the PNG is downsampled.
    plot_width = max(9.5, (leaf_count + 1) * 0.42)
    plot_height = max(1.8, (max_depth + 1.6) * 0.58)
    left, right, bottom, gap, top = 0.6, 1.5, 1.05, 1.05, 1.55
    width = left + plot_width + right
    height = bottom + 2 * plot_height + gap + top
    fig = Figure(figsize=(width, height), dpi=100, facecolor="white")
    fig.suptitle(f"{example_id} | Search budget B={budget} | {label}",
                 y=1 - 0.16 / height, fontsize=16, fontweight="bold")
    legend = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=_GOLD, markeredgewidth=2.5, label="Shortlist"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=2, label="Selected node"),
        Line2D([], [], color="black", linewidth=1.6, label="Selected path"),
    ]
    if color_by == "immediate_reward":
        legend.append(Line2D([], [], marker="o", linestyle="none", markerfacecolor=_NEUTRAL,
                             markeredgecolor="#555555", label="Root: IG undefined"))
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.57 / height),
               ncol=len(legend), frameon=False, fontsize=9)

    for index, tree in enumerate(trees):
        y_bottom = bottom + (1 - index) * (plot_height + gap)
        ax = fig.add_axes([left / width, y_bottom / height, plot_width / width, plot_height / height])
        ax.set_gid(tree.method)
        offset = (leaf_count - tree.leaf_count) / 2
        positions = {i: (x + offset, y) for i, (x, y) in tree.positions.items()}
        edges = [(n["parent_id"], i) for i, n in tree.nodes.items() if n["parent_id"] is not None]
        ax.add_collection(LineCollection([[positions[a], positions[b]] for a, b in edges],
                                        colors="#9299a3", linewidths=0.5, zorder=1, gid="tree-edges"))
        ax.add_collection(LineCollection([[positions[a], positions[b]] for a, b in tree.selected_path],
                                        colors="black", linewidths=1.6, zorder=2, gid="selected-path"))
        ids = list(tree.nodes)
        colors = [cmap(norm(tree.nodes[i][color_by])) if tree.nodes[i].get(color_by) is not None
                  else mpl.colors.to_rgba(_NEUTRAL) for i in ids]
        nodes = ax.scatter([positions[i][0] for i in ids], [positions[i][1] for i in ids],
                          s=_NODE_SIZE, c=colors, edgecolors="#333333", linewidths=0.65,
                          zorder=3, gid="tree-nodes")
        nodes.set_norm(norm)
        nodes.set_cmap(cmap)
        for highlight_ids, color, linewidth, gid in (
            (tree.shortlist_ids, _GOLD, 3.4, "shortlist-nodes"),
            ([tree.selected_id], "black", 1.8, "selected-node"),
        ):
            ax.scatter([positions[i][0] for i in highlight_ids], [positions[i][1] for i in highlight_ids],
                       s=_NODE_SIZE, facecolors="none", edgecolors=color,
                       linewidths=linewidth, zorder=4, gid=gid)
        for node_id, color in zip(ids, colors):
            order = tree.nodes[node_id]["evaluation_order"]
            text = f"Root\n{order}" if node_id == tree.root_id else str(order)
            luminance = sum(c * weight for c, weight in zip(color[:3], (0.2126, 0.7152, 0.0722)))
            ax.text(*positions[node_id], text, ha="center", va="center", fontsize=6 if node_id == tree.root_id else 7,
                    color="white" if luminance < 0.45 else "#111111", zorder=5,
                    gid=f"node-label-{node_id}")
        outcome = {True: "correct", False: "incorrect", None: "unavailable"}[tree.exact_match]
        method_name = "MEDAL" if tree.method == "medal" else "AB-MEDAL"
        selected_depth = tree.nodes[tree.selected_id]["depth"]
        ax.set_title(f"{method_name} | {len(ids)} evaluated nodes | max depth {tree.max_depth} | "
                     f"selected depth {selected_depth} | final completion: {outcome}",
                     loc="left", fontsize=11, fontweight="bold", pad=16)
        ax.set_xlim(-1, leaf_count)
        ax.set_ylim(-max_depth - 0.8, 0.8)
        ax.set_xticks([])
        ax.set_yticks([-d for d in range(max_depth + 1)], labels=range(max_depth + 1))
        ax.set_ylabel("Depth", fontsize=10)
        ax.tick_params(axis="y", labelsize=8, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    color_axis = fig.add_axes([(left + plot_width + 0.35) / width, bottom / height,
                              0.16 / width, (2 * plot_height + gap) / height])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_axis)
    colorbar.set_label(label, fontsize=10, labelpad=12)
    colorbar.ax.tick_params(labelsize=8)
    fig.text(left / width, 0.35 / height,
             "Nodes are evaluated partial token states; numbers give evaluation order. "
             "Duplicate states remain separate.\n"
             "Colors show search rewards, not answer correctness. All nodes are shown. "
             "Zoom the vector PDF to read dense trees.", fontsize=9, va="center")
    return fig


def plot_e3_tree_comparison(results_dir, output_dir, *, example_id, budget, color_by):
    """Export MEDAL/AB-MEDAL trees for exactly one saved question and budget.

    ``results_dir`` contains the ``medal`` and ``ab_medal`` directories.
    ``output_dir`` is the question-specific destination chosen by the caller.
    ``color_by`` is ``immediate_reward`` (IG) or ``mean_reward`` (backed-up Q).
    Returns ``{"png": Path, "pdf": Path}``. Inputs are never changed, and no
    model, tokenizer, search object, or live notebook state is needed.
    """
    if color_by not in COLOR_METRICS:
        raise ValueError(f"color_by must be one of {tuple(COLOR_METRICS)}; got {color_by!r}.")
    if type(budget) is not int or budget < 1:
        raise ValueError("budget must be a positive integer.")
    if not isinstance(example_id, str) or not example_id or Path(example_id).name != example_id or example_id in (".", ".."):
        raise ValueError("example_id must be one question directory name.")
    trees = _load_trees(Path(results_dir), example_id=example_id, budget=budget)
    # Local rc settings prevent user notebook themes or tight bounding boxes from
    # changing geometry, exceeding the PNG size limit, or rasterizing PDF text.
    with mpl.rc_context({"font.family": "DejaVu Sans", "text.usetex": False,
                         "text.color": "#111111", "axes.facecolor": "white",
                         "axes.labelcolor": "#111111", "axes.titlecolor": "#111111",
                         "ytick.color": "#111111", "pdf.fonttype": 42, "savefig.bbox": None}):
        fig = _comparison_figure(trees, example_id=example_id, budget=budget, color_by=color_by)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"B{budget}_{COLOR_METRICS[color_by][1]}"
        paths = {extension: output_dir / f"{stem}.{extension}" for extension in ("png", "pdf")}
        png_dpi = min(180.0, (MAX_PNG_DIMENSION - 1) / max(fig.get_size_inches()))
        try:
            fig.savefig(paths["pdf"], format="pdf", facecolor="white", transparent=False)
            fig.savefig(paths["png"], format="png", dpi=png_dpi, facecolor="white", transparent=False)
        finally:
            fig.clear()
    return paths
