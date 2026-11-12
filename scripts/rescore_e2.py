#!/usr/bin/env python3
"""Rescore saved Experiment 2 predictions with the current GSM8K answer parser.

No model inference is performed: each checkpoint record already stores the decoded
``scoring_text`` and the ``reference_solution``. Original artifacts are never modified;
rescored per-question/aggregate CSVs and a JSON report are written to
``<output-dir>/summaries/rescored/``.

Usage:
    python scripts/rescore_e2.py --output-dir results/e2_dev [--stage dev] [--expected-count 80]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dlm_project.experiments.artifacts import save_json  # noqa: E402
from dlm_project.experiments.sweep_report import write_csv  # noqa: E402
from dlm_project.tasks.gsm8k.evaluation import score_answer  # noqa: E402
from dlm_project.tasks.gsm8k.experiment_2 import compute_e2_aggregate_rows  # noqa: E402

PARSER_VERSION = "units_v2"
METHOD_ORDER = ("medal", "ab_medal")
METHOD_LABELS = {"medal": "MEDAL + FIG", "ab_medal": "AB-MEDAL + FIG"}
# CSV cells are strings; only these fields need numeric/boolean values for aggregation.
NUMERIC_FIELDS = ("search_budget", "search_nfes", "completion_nfes", "total_nfes", "search_wall_seconds",
                  "completion_wall_seconds", "total_wall_seconds", "selected_depth", "selected_q")
BOOLEAN_FIELDS = ("exact_match", "parse_success", "fallback_used")


def load_metrics_rows(stage_dir):
    """Load one row per checkpoint from <stage>/<method>/<example>/checkpoints/<budget>/metrics.json."""
    rows = []
    for path in sorted(stage_dir.glob("*/*/checkpoints/*/metrics.json")):
        row = json.loads(path.read_text())
        if isinstance(row, dict) and "scoring_text" in row and "reference_solution" in row:
            rows.append(row)
    return rows


def load_csv_rows(path):
    """Fallback loader for summaries/e2_<stage>_per_question.csv (or e2_per_question.csv)."""
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in NUMERIC_FIELDS:
            if row.get(key) not in (None, ""):
                value = float(row[key])
                row[key] = int(value) if value.is_integer() else value
        for key in BOOLEAN_FIELDS:
            if key in row:
                row[key] = row[key].strip().lower() == "true"
    return rows


def load_rows(output_dir, stage):
    stage_dir = output_dir / stage
    if stage_dir.is_dir():
        rows = load_metrics_rows(stage_dir)
        if rows:
            return rows, str(stage_dir)
    for name in (f"e2_{stage}_per_question.csv", "e2_per_question.csv"):
        path = output_dir / "summaries" / name
        if path.is_file():
            return load_csv_rows(path), str(path)
    raise FileNotFoundError(f"No E2 checkpoint metrics under {stage_dir} and no per-question CSV in "
                            f"{output_dir / 'summaries'}.")


def final_marker_line(text):
    marked = [line.strip() for line in (text or "").splitlines() if line.lstrip().startswith("####")]
    return marked[-1] if marked else None


def rescore(rows):
    rescored, changes = [], []
    for row in rows:
        score = score_answer(row["scoring_text"] or "", row["reference_solution"])
        new = {**row,
               "recorded_predicted_answer": row.get("predicted_answer"),
               "recorded_parse_success": bool(row.get("parse_success")),
               "recorded_exact_match": bool(row.get("exact_match")),
               "predicted_answer": score["predicted_answer"],
               "reference_answer": score["reference_answer"],
               "parse_success": bool(score["parse_success"]),
               "exact_match": bool(score["exact_match"]),
               "parser_version": PARSER_VERSION}
        rescored.append(new)
        if (new["recorded_exact_match"], new["recorded_parse_success"]) != (new["exact_match"], new["parse_success"]):
            changes.append({"example_id": row["example_id"], "method": row["method"],
                            "search_budget": row["search_budget"],
                            "final_marker_line": final_marker_line(row["scoring_text"]),
                            "reference_answer": score["reference_answer"],
                            "recorded": {"predicted_answer": new["recorded_predicted_answer"],
                                         "parse_success": new["recorded_parse_success"],
                                         "exact_match": new["recorded_exact_match"]},
                            "corrected": {"predicted_answer": new["predicted_answer"],
                                          "parse_success": new["parse_success"],
                                          "exact_match": new["exact_match"]}})
    return rescored, changes


def comparison_table(rescored, methods, budgets):
    table = []
    for method in methods:
        for budget in budgets:
            subset = [r for r in rescored if r["method"] == method and r["search_budget"] == budget]
            if subset:
                table.append({"method": method, "search_budget": budget, "sample_count": len(subset),
                              "recorded_correct": sum(r["recorded_exact_match"] for r in subset),
                              "corrected_correct": sum(r["exact_match"] for r in subset),
                              "recorded_parsed": sum(r["recorded_parse_success"] for r in subset),
                              "corrected_parsed": sum(r["parse_success"] for r in subset)})
    return table


def print_report(table, changes, source):
    print(f"Source: {source}\n")
    print(f"{'Method':<16} {'Budget':>6}  {'Recorded':>12}  {'Corrected':>12}  {'Parsed (rec->cor)':>18}")
    for t in table:
        n = t["sample_count"]
        rec, cor = t["recorded_correct"], t["corrected_correct"]
        print(f"{METHOD_LABELS.get(t['method'], t['method']):<16} {t['search_budget']:>6}  "
              f"{f'{rec}/{n} ({100 * rec / n:.0f}%)':>12}  {f'{cor}/{n} ({100 * cor / n:.0f}%)':>12}  "
              f"{t['recorded_parsed']:>8} -> {t['corrected_parsed']:<6}")
    print(f"\nChanged records: {len(changes)}")
    for c in changes:
        print(f"  {c['example_id']:<11} {c['method']:<9} {c['search_budget']:>4} | "
              f"{c['final_marker_line']!r} | ref={c['reference_answer']} | "
              f"match {c['recorded']['exact_match']} -> {c['corrected']['exact_match']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, type=Path, help="E2 output directory, e.g. results/e2_dev")
    parser.add_argument("--stage", default="dev")
    parser.add_argument("--expected-count", type=int, default=None,
                        help="Fail unless exactly this many checkpoint records are found (e.g. 80).")
    parser.add_argument("--dry-run", action="store_true", help="Print the report without writing files.")
    args = parser.parse_args(argv)

    rows, source = load_rows(args.output_dir, args.stage)
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise SystemExit(f"Expected {args.expected_count} records but found {len(rows)} in {source}.")

    rescored, changes = rescore(rows)
    regressions = [c for c in changes if c["recorded"]["exact_match"] and not c["corrected"]["exact_match"]]
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rescored)]
    methods += sorted({r["method"] for r in rescored} - set(methods))
    budgets = sorted({r["search_budget"] for r in rescored})
    table = comparison_table(rescored, methods, budgets)
    print_report(table, changes, source)
    if regressions:
        raise SystemExit(f"Refusing to write: {len(regressions)} record(s) went from correct to incorrect.")
    if args.dry_run:
        return table, changes

    out_dir = args.output_dir / "summaries" / "rescored"
    out_dir.mkdir(parents=True, exist_ok=True)
    order = {(m, b): i for i, (m, b) in enumerate((m, b) for b in budgets for m in methods)}
    rescored.sort(key=lambda r: (order[(r["method"], r["search_budget"])], r["example_id"]))
    write_csv(out_dir / f"e2_{args.stage}_per_question_rescored.csv", rescored)
    write_csv(out_dir / f"e2_{args.stage}_aggregate_rescored.csv",
              compute_e2_aggregate_rows(rescored, args.stage, methods, budgets))
    save_json(out_dir / "rescore_report.json", {"parser_version": PARSER_VERSION, "source": source,
                                                "record_count": len(rescored), "comparison": table,
                                                "changed_records": changes})
    print(f"\nWrote rescored outputs to {out_dir}")
    return table, changes


if __name__ == "__main__":
    main()

