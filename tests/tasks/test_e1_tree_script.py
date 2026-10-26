"""Tests for scripts/e1_tree.py CLI."""

import json
from pathlib import Path
import pytest

from scripts.e1_tree import (
    discover_available_questions,
    find_results_dir,
    main,
    resolve_question_id,
)
from tests.tasks.test_e3_tree_plots import EXAMPLE_ID, _write_pair


def test_discover_and_resolve_questions(tmp_path):
    stage_dir = tmp_path / "dev"
    _write_pair(stage_dir, budget=5)

    for method in ("medal", "ab_medal"):
        m_file = stage_dir / method / EXAMPLE_ID / "checkpoints" / "5" / "metrics.json"
        data = json.loads(m_file.read_text())
        data.update({"question": "A bakery makes 12 cakes.", "reference_answer": "12"})
        m_file.write_text(json.dumps(data))

    available = discover_available_questions(stage_dir, budget=5)
    assert len(available) == 1
    assert available[0]["example_id"] == EXAMPLE_ID
    assert available[0]["reference_answer"] == "12"

    # Test resolve by index
    assert resolve_question_id(available, q_idx=0)["example_id"] == EXAMPLE_ID
    # Test resolve by example_id
    assert resolve_question_id(available, example_id=EXAMPLE_ID)["example_id"] == EXAMPLE_ID
    # Test resolve by search
    assert resolve_question_id(available, q_search="bakery")["example_id"] == EXAMPLE_ID
    assert resolve_question_id(available, q_search="nonexistent") is None


def test_e1_tree_cli_run(tmp_path, capsys):
    stage_dir = tmp_path / "dev"
    _write_pair(stage_dir, budget=5)

    for method in ("medal", "ab_medal"):
        m_file = stage_dir / method / EXAMPLE_ID / "checkpoints" / "5" / "metrics.json"
        data = json.loads(m_file.read_text())
        data.update({
            "question": "A bookstore sells 15 books.",
            "reference_answer": "15",
            "predicted_answer": "15",
            "exact_match": True,
            "selected_depth": 2,
            "selected_q": 0.7,
            "total_nfes": 12,
        })
        m_file.write_text(json.dumps(data))

    out_dir = tmp_path / "figures"
    ret = main([
        "--stage", "dev",
        "--results-dir", str(stage_dir),
        "--output-dir", str(out_dir),
        "--budget", "5",
        "--min-c", "2",
        "--question-idx", "0",
        "--show-comparison",
        "--show-candidates",
    ])
    assert ret == 0

    stdout = capsys.readouterr().out
    assert "PER-QUESTION SCORE COMPARISON" in stdout
    assert "AGGREGATE ANALYSIS SUMMARY" in stdout
    assert "DISTINCT CANDIDATE ANALYSIS (Threshold C = 2)" in stdout
    assert "RENDERING AB-MCTS COMPARISON TREES" in stdout

    # Verify PNG and PDF figures were generated
    assert (out_dir / "B5_immediate_ig.png").is_file()
    assert (out_dir / "B5_immediate_ig.pdf").is_file()
    assert (out_dir / "B5_mean_q.png").is_file()
    assert (out_dir / "B5_mean_q.pdf").is_file()
