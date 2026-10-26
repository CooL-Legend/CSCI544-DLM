"""Workflow test for notebooks/gsm8k/E1_Tree_Inspector.ipynb."""

import json
from pathlib import Path
import nbformat
import pytest

from tests.tasks.test_e3_tree_plots import EXAMPLE_ID, _write_pair


def test_e1_tree_inspector_notebook_compilation():
    path = Path("notebooks/gsm8k/E1_Tree_Inspector.ipynb")
    assert path.is_file()
    nb = nbformat.read(str(path), as_version=4)
    nbformat.validate(nb)

    # Confirm cell count and cell types
    assert len(nb.cells) == 8
    assert nb.cells[0].cell_type == "markdown"
    for cell in nb.cells[1:]:
        assert cell.cell_type == "code"
        compile(cell.source, f"cell_{cell.id}", "exec")


def test_e1_tree_inspector_execution_flow(tmp_path, monkeypatch):
    # Set up synthetic checkpoint fixtures in tmp_path / dev
    stage_dir = tmp_path / "dev"
    _write_pair(stage_dir, budget=5)

    # Add question metadata to metrics.json
    for method in ("medal", "ab_medal"):
        m_file = stage_dir / method / EXAMPLE_ID / "checkpoints" / "5" / "metrics.json"
        data = json.loads(m_file.read_text())
        data.update({
            "question": "Janet has 5 apples and buys 3 more.",
            "reference_answer": "8",
            "predicted_answer": "8",
            "exact_match": True,
            "selected_depth": 2,
            "selected_q": 0.5,
            "total_nfes": 15,
        })
        m_file.write_text(json.dumps(data))

    nb = nbformat.read("notebooks/gsm8k/E1_Tree_Inspector.ipynb", as_version=4)

    # Prepare scope for cell execution
    scope = {
        "IN_COLAB": False,
        "REPO_DIR": Path.cwd(),
        "STAGE": "dev",
        "BUDGET": 5,
        "MIN_C": 2,
        "DATA_DIR": tmp_path,
        "RESULTS_STAGE_DIR": stage_dir,
    }

    # Intercept display to prevent UI popups during testing
    displayed = []
    monkeypatch.setattr("IPython.display.display", lambda item: displayed.append(item))

    # Execute Cell 4 (Helpers)
    exec(nb.cells[3].source, scope)
    assert "extract_distinct_candidates" in scope
    assert "discover_stage_questions" in scope

    # Execute Cell 5 (Early Section: Score Comparison)
    exec(nb.cells[4].source, scope)
    assert "df_questions" in scope
    assert len(scope["df_questions"]) == 1
    assert scope["df_questions"].iloc[0]["Example ID"] == EXAMPLE_ID
    assert scope["df_questions"].iloc[0]["Outcome"] == "Both Correct (Tie)"

    # Execute Cell 6 (Question Selector)
    scope["QUESTION_INDEX"] = 0
    exec(nb.cells[5].source, scope)
    assert scope["SELECTED_EXAMPLE_ID"] == EXAMPLE_ID

    # Execute Cell 7 (Candidate Analysis at Depth with >= C candidates)
    exec(nb.cells[6].source, scope)
    assert "candidate_reports" in scope
    for method in ("medal", "ab_medal"):
        report = scope["candidate_reports"][method]
        assert report["candidate_depth"] == 2
        assert report["candidate_count"] == 2  # At depth 2, has 2 distinct states

    # Execute Cell 8 (Tree Visualization)
    exec(nb.cells[7].source, scope)
    assert "exported_figures" in scope
    for metric in ("immediate_reward", "mean_reward"):
        paths = scope["exported_figures"][metric]
        assert paths["png"].is_file()
        assert paths["pdf"].is_file()
