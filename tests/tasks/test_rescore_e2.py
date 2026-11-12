"""Rescoring saved E2 predictions with the unit-tolerant parser, without model inference."""

import json

import pytest

from scripts.rescore_e2 import main


def write_record(stage_dir, method, example_id, budget, scoring_text, reference, recorded_match, recorded_parse):
    directory = stage_dir / method / example_id / "checkpoints" / str(budget)
    directory.mkdir(parents=True)
    row = {"method": method, "search_budget": budget, "example_id": example_id, "row_id": 0,
           "question": "q", "reference_solution": reference, "scoring_text": scoring_text,
           "predicted_answer": None if not recorded_parse else "0", "reference_answer": None,
           "exact_match": recorded_match, "parse_success": recorded_parse, "search_nfes": budget,
           "completion_nfes": 10, "total_nfes": budget + 10, "search_wall_seconds": 1.0,
           "completion_wall_seconds": 1.0, "total_wall_seconds": 2.0, "selected_depth": 3,
           "selected_q": 0.5, "fallback_used": False}
    (directory / "metrics.json").write_text(json.dumps(row))
    return directory / "metrics.json"


def build_run(tmp_path):
    stage_dir = tmp_path / "dev"
    paths = [
        write_record(stage_dir, "medal", "train:1978", 256, "...\n#### 110 miles", "x\n#### 110", False, False),
        write_record(stage_dir, "ab_medal", "train:5224", 256, "...\n#### 15 pounds", "x\n#### 15", False, False),
        write_record(stage_dir, "medal", "train:1", 256, "#### 7", "#### 7", True, True),
        write_record(stage_dir, "ab_medal", "train:1", 256, "#### 8 apples", "#### 7", False, False),
        write_record(stage_dir, "medal", "train:2", 256, "no marker", "#### 3", False, False),
    ]
    return paths


def test_rescore_recovers_unit_answers_and_leaves_originals_untouched(tmp_path, capsys):
    paths = build_run(tmp_path)
    originals = [p.read_text() for p in paths]
    table, changes = main(["--output-dir", str(tmp_path), "--expected-count", "5"])

    by_key = {(t["method"], t["search_budget"]): t for t in table}
    assert (by_key[("medal", 256)]["recorded_correct"], by_key[("medal", 256)]["corrected_correct"]) == (1, 2)
    assert (by_key[("ab_medal", 256)]["recorded_correct"], by_key[("ab_medal", 256)]["corrected_correct"]) == (0, 1)
    flipped = {(c["example_id"], c["method"]) for c in changes if c["corrected"]["exact_match"]}
    assert flipped == {("train:1978", "medal"), ("train:5224", "ab_medal")}
    # "#### 8 apples" now parses but is still wrong.
    assert any(c["example_id"] == "train:1" and not c["corrected"]["exact_match"] for c in changes)

    assert [p.read_text() for p in paths] == originals
    out = tmp_path / "summaries" / "rescored"
    report = json.loads((out / "rescore_report.json").read_text())
    assert report["record_count"] == 5 and len(report["changed_records"]) == 3
    assert (out / "e2_dev_per_question_rescored.csv").is_file()
    aggregate = (out / "e2_dev_aggregate_rescored.csv").read_text()
    assert "accuracy" in aggregate and "parse_rate" in aggregate


def test_rescore_dry_run_and_count_guard(tmp_path):
    build_run(tmp_path)
    main(["--output-dir", str(tmp_path), "--dry-run"])
    assert not (tmp_path / "summaries").exists()
    with pytest.raises(SystemExit, match="Expected 80"):
        main(["--output-dir", str(tmp_path), "--expected-count", "80"])

