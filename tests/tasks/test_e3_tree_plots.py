"""Offline E3 tree plots preserve saved topology, scores and checkpoint identity."""

import json
import subprocess
import sys

import matplotlib as mpl
import numpy as np
from PIL import Image
import pytest

from dlm_project.tasks.gsm8k import e3_tree_plots as plots


EXAMPLE_ID = "train:6499"


def _small_data(method="medal", budget=5):
    # Deliberately shuffled records, nonsequential IDs, and duplicate states.
    rows = [
        dict(node_id=0, parent_id=None, depth=0, evaluation_order=0,
             immediate_reward=None, mean_reward=0.1, state_hash="root"),
        dict(node_id=40, parent_id=0, depth=1, evaluation_order=1,
             immediate_reward=-0.2, mean_reward=0.3, state_hash="duplicate"),
        dict(node_id=7, parent_id=0, depth=1, evaluation_order=2,
             immediate_reward=0.4, mean_reward=-0.7, state_hash="duplicate"),
        dict(node_id=90, parent_id=40, depth=2, evaluation_order=3,
             immediate_reward=0.0, mean_reward=0.2, state_hash="leaf-a"),
        dict(node_id=51, parent_id=40, depth=2, evaluation_order=4,
             immediate_reward=0.1, mean_reward=0.9, state_hash="leaf-b"),
    ]
    # A pending, unevaluated child must never become a plotted node or edge.
    rows[1]["child_ids"] = [90, 51, 999]
    if method == "ab_medal":
        rows[1]["immediate_reward"] = -0.8
    return {
        "tree": {"method": method, "example_id": EXAMPLE_ID, "budget": budget,
                 "nodes": [rows[i] for i in (4, 2, 0, 3, 1)]},
        "selection": {"selected_node_id": 90, "selected_depth": 2,
                      "shortlist_ids": [90, 51], "requested_budget": budget},
        "metrics": {"method": method, "example_id": EXAMPLE_ID, "search_budget": budget,
                    "exact_match": method == "medal"},
    }


def _large_data(method, budget):
    rows = []
    for order in range(budget):
        # A regular MEDAL tree and a wider, uneven AB tree, with the same prefix
        # at every checkpoint. IDs intentionally differ from evaluation order.
        parent_order = (order - 1) // 5 if method == "medal" else (order - 1) // 9
        parent = rows[parent_order] if order else None
        rows.append(dict(node_id=10 + order * 3, parent_id=parent["node_id"] if parent else None,
                         depth=parent["depth"] + 1 if parent else 0, evaluation_order=order,
                         immediate_reward=((order % 17) - 8) / 100 if order else None,
                         mean_reward=((order % 23) - 11) / 200,
                         state_hash=f"state-{order // 2}"))
    selected = rows[-1]
    return {
        "tree": {"method": method, "example_id": EXAMPLE_ID, "budget": budget, "nodes": rows},
        "selection": {"selected_node_id": selected["node_id"], "selected_depth": selected["depth"],
                      "shortlist_ids": [r["node_id"] for r in rows[-3:]], "requested_budget": budget},
        "metrics": {"method": method, "example_id": EXAMPLE_ID, "search_budget": budget,
                    "exact_match": method == "medal"},
    }


def _write_pair(results_dir, budget=5, factory=_small_data):
    for method in plots.METHODS:
        directory = results_dir / method / EXAMPLE_ID / "checkpoints" / str(budget)
        directory.mkdir(parents=True, exist_ok=True)
        for name, record in factory(method, budget).items():
            (directory / f"{name}.json").write_text(json.dumps(record))


def _artists(ax):
    return {artist.get_gid(): artist for artist in ax.collections}


def test_full_topology_order_duplicate_states_and_color_comparability(tmp_path):
    _write_pair(tmp_path)
    trees = plots._load_trees(tmp_path, example_id=EXAMPLE_ID, budget=5)
    assert trees[0].positions == {0: (1, 0), 40: (0.5, -1), 7: (2, -1), 90: (0, -2), 51: (1, -2)}
    assert trees[0].selected_path == [(40, 90), (0, 40)]
    figures = [plots._comparison_figure(trees, example_id=EXAMPLE_ID, budget=5, color_by=metric)
               for metric in plots.COLOR_METRICS]
    try:
        for figure, limit in zip(figures, (0.8, 0.9)):
            for ax in figure.axes[:2]:
                artists = _artists(ax)
                assert len(artists["tree-edges"].get_segments()) == 4
                assert len(artists["tree-nodes"].get_offsets()) == 5
                assert artists["tree-nodes"].get_clim() == (-limit, limit)
                np.testing.assert_allclose(artists["shortlist-nodes"].get_offsets(), [[0, -2], [1, -2]])
                np.testing.assert_allclose(artists["selected-node"].get_offsets(), [[0, -2]])
                np.testing.assert_allclose(artists["selected-path"].get_segments(),
                                           [[[0.5, -1], [0, -2]], [[1, 0], [0.5, -1]]])
                labels = {t.get_gid(): t.get_text() for t in ax.texts}
                assert labels == {"node-label-0": "Root\n0", "node-label-40": "1", "node-label-7": "2",
                                  "node-label-90": "3", "node-label-51": "4"}
            assert "final completion: correct" in figure.axes[0].get_title(loc="left")
            assert "final completion: incorrect" in figure.axes[1].get_title(loc="left")
        for ig_ax, q_ax in zip(figures[0].axes[:2], figures[1].axes[:2]):
            np.testing.assert_array_equal(_artists(ig_ax)["tree-nodes"].get_offsets(),
                                          _artists(q_ax)["tree-nodes"].get_offsets())
            assert ig_ax.get_xlim() == q_ax.get_xlim()
            assert ig_ax.get_ylim() == q_ax.get_ylim()
        ig_nodes = _artists(figures[0].axes[0])["tree-nodes"]
        np.testing.assert_allclose(ig_nodes.get_facecolors()[0], mpl.colors.to_rgba(plots._NEUTRAL))
        assert not np.array_equal(ig_nodes.get_facecolors()[1], ig_nodes.get_facecolors()[2])
    finally:
        for figure in figures:
            figure.clear()


def test_zero_rewards_root_only_and_chain():
    data = _small_data()
    rows = sorted(data["tree"]["nodes"], key=lambda row: row["evaluation_order"])
    for index, row in enumerate(rows):
        row.update(parent_id=rows[index - 1]["node_id"] if index else None, depth=index,
                   immediate_reward=0 if index else None, mean_reward=0)
    data["selection"].update(selected_node_id=51, selected_depth=4)
    tree = plots._validated_tree(data, method="medal", example_id=EXAMPLE_ID, budget=5)
    assert tree.leaf_count == 1 and tree.max_depth == 4
    assert {x for x, y in tree.positions.values()} == {0}
    assert len(tree.selected_path) == 4
    for root_only in (False, True):
        if root_only:
            rows[0].pop("immediate_reward")  # Missing root IG is also undefined.
            data["tree"].update(nodes=[rows[0]], budget=1)
            data["selection"].update(selected_node_id=0, selected_depth=0, shortlist_ids=[0], requested_budget=1)
            data["metrics"].update(search_budget=1, exact_match=None)
            tree = plots._validated_tree(data, method="medal", example_id=EXAMPLE_ID, budget=1)
        for metric in plots.COLOR_METRICS:
            fig = plots._comparison_figure([tree, tree], example_id=EXAMPLE_ID,
                                           budget=1 if root_only else 5, color_by=metric)
            assert _artists(fig.axes[0])["tree-nodes"].get_clim() == (-1, 1)
            if root_only:
                assert len(_artists(fig.axes[0])["tree-edges"].get_segments()) == 0
                assert "unavailable" in fig.axes[0].get_title(loc="left")
            fig.clear()


@pytest.mark.parametrize("mutation, message", [
    (lambda d: d["tree"]["nodes"][0].update(parent_id=999), "missing parent"),
    (lambda d: d["tree"]["nodes"][0].update(parent_id=51), "invalid topology"),
    (lambda d: d["tree"]["nodes"][0].update(depth=99), "depth must equal"),
    (lambda d: d["tree"]["nodes"][0].update(node_id=0), "unique integers"),
    (lambda d: d["tree"]["nodes"][0].update(evaluation_order=1), "evaluation_order must be unique"),
    (lambda d: d["tree"]["nodes"][0].update(evaluation_order=5), "below budget"),
    (lambda d: d["tree"]["nodes"][0].update(immediate_reward=float("nan")), "finite immediate_reward"),
    (lambda d: d["tree"]["nodes"][0].update(mean_reward=None), "finite mean_reward"),
    (lambda d: d["tree"].update(budget=512), "expected 5"),
    (lambda d: d["tree"].update(example_id="train:other"), "expected 'train:6499'"),
    (lambda d: d["selection"].update(selected_node_id=999), "selected_node_id is absent"),
    (lambda d: d["selection"].update(shortlist_ids=[999]), "shortlist_ids"),
    (lambda d: d["selection"].update(selected_depth=1), "selected_depth disagrees"),
])
def test_invalid_checkpoint_is_rejected(mutation, message):
    data = _small_data()
    mutation(data)
    with pytest.raises(ValueError, match=message):
        plots._validated_tree(data, method="medal", example_id=EXAMPLE_ID, budget=5)


def test_missing_checkpoint_does_not_fall_back_or_create_exports(tmp_path):
    _write_pair(tmp_path / "dev")
    output = tmp_path / "figures"
    with pytest.raises(FileNotFoundError, match="checkpoints/256/tree.json"):
        plots.plot_e3_tree_comparison(tmp_path / "dev", output, example_id=EXAMPLE_ID,
                                      budget=256, color_by="immediate_reward")
    (tmp_path / "dev" / "ab_medal" / EXAMPLE_ID / "checkpoints/5/selection.json").unlink()
    with pytest.raises(FileNotFoundError, match="ab_medal.*selection.json"):
        plots.plot_e3_tree_comparison(tmp_path / "dev", output, example_id=EXAMPLE_ID,
                                      budget=5, color_by="mean_reward")
    with pytest.raises(ValueError, match="color_by"):
        plots.plot_e3_tree_comparison(tmp_path / "dev", output, example_id=EXAMPLE_ID,
                                      budget=5, color_by="visits")
    assert not output.exists()


@pytest.mark.parametrize("budget", [256, 512])
def test_large_checkpoint_exports_are_complete_and_read_only(tmp_path, budget, monkeypatch):
    results = tmp_path / "dev"
    for b in (256, 512):
        _write_pair(results, b, _large_data)
    before = {p: p.read_bytes() for p in results.rglob("*.json")}
    trees = plots._load_trees(results, example_id=EXAMPLE_ID, budget=budget)
    assert all(len(tree.nodes) == budget for tree in trees)
    assert all(max(n["evaluation_order"] for n in tree.nodes.values()) == budget - 1 for tree in trees)
    figures = []
    original = plots._comparison_figure

    def inspect(*args, **kwargs):
        fig = original(*args, **kwargs)
        for ax in fig.axes[:2]:
            assert len(_artists(ax)["tree-nodes"].get_offsets()) == budget
            assert len(_artists(ax)["tree-edges"].get_segments()) == budget - 1
            assert len(ax.texts) == budget
        figures.append(fig)
        return fig

    monkeypatch.setattr(plots, "_comparison_figure", inspect)
    for metric in plots.COLOR_METRICS:
        paths = plots.plot_e3_tree_comparison(results, tmp_path / "figures", example_id=EXAMPLE_ID,
                                              budget=budget, color_by=metric)
        assert set(paths) == {"png", "pdf"}
        assert paths["pdf"].read_bytes().startswith(b"%PDF-")
        assert paths["pdf"].stat().st_size > 1000
        with Image.open(paths["png"]) as png:
            assert max(png.size) <= plots.MAX_PNG_DIMENSION
            assert min(png.size) > 100
    assert all(not fig.axes for fig in figures)
    assert before == {p: p.read_bytes() for p in results.rglob("*.json")}


def test_cpu_only_subprocess_never_imports_model_libraries(tmp_path):
    _write_pair(tmp_path / "dev")
    script = """
import sys
class NoModelImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'huggingface_hub'}:
            raise AssertionError('Unexpected model dependency: ' + fullname)
sys.meta_path.insert(0, NoModelImports())
from dlm_project.tasks.gsm8k.e3_tree_plots import plot_e3_tree_comparison
plot_e3_tree_comparison(sys.argv[1], sys.argv[2], example_id='train:6499',
                        budget=5, color_by='mean_reward')
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path / "dev"), str(tmp_path / "figures")],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_real_e1_checkpoint_schema_and_zero_additional_forward_calls(tmp_path):
    from dlm_project.tasks.gsm8k import experiment_1 as e1
    from tests.tasks.test_experiment_1 import mock_wrapper

    wrapper = mock_wrapper()
    ctx = e1.context(stage="dev", methods=plots.METHODS, num_samples=1)
    e1.run_e1_stage(wrapper, ctx, tmp_path, methods=plots.METHODS,
                    budgets=(2, 4), max_depth=4, num_candidates=2)
    calls = len(wrapper.model.calls)
    for metric in plots.COLOR_METRICS:
        paths = plots.plot_e3_tree_comparison(tmp_path / "dev", tmp_path / "figures",
                                              example_id=ctx["examples"][0]["example_id"],
                                              budget=4, color_by=metric)
        assert all(path.is_file() for path in paths.values())
    assert len(wrapper.model.calls) == calls
