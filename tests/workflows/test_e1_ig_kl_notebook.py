"""Validate and exercise the copied notebook's real settings and runner cells."""

from pathlib import Path

import nbformat

from tests.tasks.test_experiment_1 import mock_wrapper


BOOK = Path("notebooks/gsm8k/Experiment_1_Eval_IG_KL.ipynb")


def test_notebook_valid_and_original_remains_ig():
    book = nbformat.read(BOOK, as_version=4)
    nbformat.validate(book)
    original = nbformat.read(BOOK.with_name("Experiment_1_Eval.ipynb"), as_version=4)
    old_text = "\n".join(c.source for c in original.cells)
    text = "\n".join(c.source for c in book.cells)
    assert "| **Immediate Reward** | Normalized IG" in old_text
    assert 'BRANCH = "adi"' in old_text
    assert 'BRANCH = "sean-1105"' in text
    assert "Experiment_1_Eval_IG_KL.ipynb" in text
    assert "previous and current predictions" in text
    for cell in book.cells:
        if cell.cell_type == "code":
            compile(cell.source, cell.id, "exec")
            assert cell.outputs == [] and cell.execution_count is None
    inspector = next(c.source for c in book.cells if c.id == "e78ec7a3")
    assert "search_roots" not in inspector  # Never fall back to original IG results.


def test_notebook_settings_and_execution_reach_reward_b(tmp_path):
    cells = {c.id: c.source for c in nbformat.read(BOOK, as_version=4).cells}
    scope = {"OUTPUT_BASE": tmp_path / "e1_e3_ig_kl", "wrapper": mock_wrapper()}
    exec(cells["fd1c6c8c"], scope)
    assert scope["REWARD_MODE"] == "ig_kl"
    assert scope["ctx"]["reward"]["kl_weight"] == 0.05
    assert scope["ctx"]["reward"]["kl_top_k"] == 32
    assert len(scope["ctx"]["examples"]) == 20
    assert scope["BUDGETS"] == [256, 512]
    assert scope["MAX_DEPTH"] == 10 and scope["MIN_C"] == 3
    assert scope["OUTPUT_DIR"].name == "lambda_0.05_topk_32"
    # Keep the actual notebook runner cell but use a tiny CPU test workload.
    scope["ctx"]["examples"] = scope["ctx"]["examples"][:1]
    scope.update(BUDGETS=[3, 6], MAX_DEPTH=3, MIN_C=2)
    exec(cells["09969851"], scope)
    assert len(scope["q_rows"]) == 4
    assert all(r["reward_mode"] == "ig_kl" for r in scope["q_rows"])
    assert all(r["kl_extra_model_calls"] == 0 for r in scope["q_rows"])
    assert {r["method"] for r in scope["q_rows"]} == {"medal", "ab_medal"}
