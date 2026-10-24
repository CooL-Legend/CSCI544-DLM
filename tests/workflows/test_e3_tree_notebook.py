"""Run just the new notebook section against saved CPU-only checkpoint fixtures."""

import nbformat
import pytest

from tests.tasks.test_e3_tree_plots import EXAMPLE_ID, _write_pair


@pytest.mark.parametrize("standalone", [False, True])
def test_e3_tree_notebook_section(tmp_path, monkeypatch, standalone, capsys):
    book = nbformat.read("notebooks/gsm8k/Experiment_1_Dev.ipynb", as_version=4)
    nbformat.validate(book)
    cells = {cell.id: cell for cell in book.cells}
    ids = [cell.id for cell in book.cells]
    assert ids.index("rc6y64g72lST") < ids.index("e3-tree-settings") < ids.index("f5c4fbf2")
    assert "## 14. Synchronize" in cells["f5c4fbf2"].source
    for cell in book.cells:
        if cell.cell_type == "code":
            compile(cell.source, cell.id, "exec")
    for cell_id in ("e3-tree-settings", "e3-tree-render"):
        assert cells[cell_id].outputs == [] and cells[cell_id].execution_count is None

    root = tmp_path / "saved results"
    _write_pair(root / "dev")
    scope = {}
    settings = cells["e3-tree-settings"].source
    if standalone:
        settings = settings.replace("TREE_RESULTS_DIR_OVERRIDE = None", f"TREE_RESULTS_DIR_OVERRIDE = {str(root / 'dev')!r}")
        settings = settings.replace("TREE_EXAMPLE_ID_OVERRIDE = None", f"TREE_EXAMPLE_ID_OVERRIDE = {EXAMPLE_ID!r}")
        settings = settings.replace("TREE_BUDGET_OVERRIDE = None", "TREE_BUDGET_OVERRIDE = 5")
    else:
        scope.update(actual_output_dir=root, STAGE="dev", INSPECT_QUESTION_IDX=1, INSPECT_BUDGET=5,
                     ctx={"examples": [{"example_id": "train:wrong"}, {"example_id": EXAMPLE_ID}]},
                     ex_id="train:stale", OUTPUT_DIR=tmp_path / "wrong-results-root")
    displayed = []
    monkeypatch.setattr("IPython.display.display", lambda item: displayed.append(item))
    exec(settings, scope)
    exec(cells["e3-tree-render"].source, scope)
    assert scope["tree_example_id"] == EXAMPLE_ID
    assert scope["tree_results_dir"] == root / "dev"
    assert scope["tree_budget"] == 5
    assert len(displayed) == 4  # Two headings and two PNG previews.
    expected_dir = root / "summaries/e3_figures/trees/dev" / EXAMPLE_ID
    assert scope["tree_output_dir"] == expected_dir
    for paths in scope["tree_exports"].values():
        assert all(path.is_file() and path.parent == expected_dir for path in paths.values())
    output = capsys.readouterr().out
    for method in ("medal", "ab_medal"):
        for filename in ("tree.json", "selection.json", "metrics.json"):
            assert str(root / "dev" / method / EXAMPLE_ID / "checkpoints/5" / filename) in output
