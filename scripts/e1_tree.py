"""CLI tool for inspecting E1 questions, candidate history, and rendering AB-MCTS comparison trees.

Usage examples:
    python scripts/e1_tree.py --question-idx 0 --stage dev --budget 512
    python scripts/e1_tree.py --example-id train:6499 --stage dev --show-candidates
    python scripts/e1_tree.py --question "bookstore" --show-comparison
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from dlm_project.tasks.gsm8k.candidate_history import (
    extract_distinct_candidates,
    load_question_comparison,
)
from dlm_project.tasks.gsm8k.e3_tree_plots import COLOR_METRICS, plot_e3_tree_comparison


def find_results_dir(stage: str, custom_dir: Optional[Path] = None) -> Path:
    if custom_dir:
        candidate = custom_dir.expanduser().resolve()
        if (candidate / stage).is_dir():
            return candidate / stage
        if (candidate / "medal").is_dir() or (candidate / "ab_medal").is_dir():
            return candidate
        return candidate

    search_roots = [
        Path(f"/content/drive/MyDrive/CSCI544-DLM/LLaDA/GSM8K/experiment_1/results/e1_e3/{stage}"),
        Path(f"/content/drive/MyDrive/CSCI544-DLM/results/e1_e3/{stage}"),
        Path(f"results/e1_e3/{stage}"),
        Path("results/e1_e3"),
        Path(f"/content/CSCI544-DLM/results/e1_e3/{stage}"),
    ]
    for root in search_roots:
        if root.is_dir() and ((root / "medal").is_dir() or (root / "ab_medal").is_dir()):
            return root
    return Path(f"results/e1_e3/{stage}")


def discover_available_questions(stage_dir: Path, budget: int) -> List[Dict[str, Any]]:
    questions = {}
    for method in ("medal", "ab_medal"):
        m_dir = stage_dir / method
        if not m_dir.is_dir():
            continue
        for q_dir in sorted(m_dir.iterdir()):
            if not q_dir.is_dir():
                continue
            ex_id = q_dir.name
            chk_dir = q_dir / "checkpoints" / str(budget)
            met_file = chk_dir / "metrics.json"
            if met_file.is_file():
                if ex_id not in questions:
                    questions[ex_id] = {
                        "example_id": ex_id,
                        "methods": [],
                        "question": "",
                        "reference_answer": "",
                    }
                questions[ex_id]["methods"].append(method)
                try:
                    met = json.loads(met_file.read_text())
                    questions[ex_id]["question"] = met.get("question", "")
                    questions[ex_id]["reference_answer"] = met.get("reference_answer", "")
                except Exception:
                    pass
    return list(questions.values())


def resolve_question_id(
    available: List[Dict[str, Any]],
    q_idx: Optional[int] = None,
    example_id: Optional[str] = None,
    q_search: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not available:
        return None
    if example_id:
        match = next((q for q in available if q["example_id"] == example_id), None)
        if match:
            return match
    if q_idx is not None:
        if 0 <= q_idx < len(available):
            return available[q_idx]
    if q_search:
        search_lower = q_search.lower()
        match = next((q for q in available if search_lower in q["question"].lower() or search_lower in q["example_id"].lower()), None)
        if match:
            return match
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("dev", "evaluation", "test"), default="dev")
    parser.add_argument("--results-dir", type=Path, help="Path to results directory containing medal/ and ab_medal/")
    parser.add_argument("--output-dir", type=Path, help="Custom directory to export generated tree figures")
    parser.add_argument("--budget", "-b", type=int, default=512, help="Checkpoint budget (default: 512)")
    parser.add_argument("--min-c", type=int, default=3, help="Minimum distinct candidates threshold C (default: 3)")
    parser.add_argument("--question-idx", "-i", type=int, help="0-based index of question in results directory")
    parser.add_argument("--example-id", "-e", type=str, help="Question example ID (e.g. 'train:6499')")
    parser.add_argument("--question", "-q", type=str, help="Question text or substring search")
    parser.add_argument("--color-by", choices=("both", "immediate_reward", "mean_reward"), default="both")
    parser.add_argument("--show-comparison", action="store_true", help="Display per-question and aggregate score comparison table")
    parser.add_argument("--show-candidates", action="store_true", help="Display detailed candidate list at depth with >= C candidates")

    args = parser.parse_args(argv)

    stage_dir = find_results_dir(args.stage, args.results_dir)
    print(f"E1-Tree Inspector: stage={args.stage}, budget={args.budget}, min-C={args.min_c}")
    print(f"Results Directory: {stage_dir} (exists: {stage_dir.exists()})")

    # 1. Early Comparison Section if requested or by default when multiple questions exist
    if args.show_comparison:
        print("\n" + "=" * 80)
        print("PER-QUESTION SCORE COMPARISON (MEDAL vs AB-MEDAL)")
        print("=" * 80)
        q_rows, agg = load_question_comparison(stage_dir, stage=args.stage, budget=args.budget)
        if not q_rows:
            print("No completed questions found at this budget to compare.")
        else:
            header = f"{'Example ID':<15} | {'Ref Ans':<8} | {'MEDAL':<12} | {'AB-MEDAL':<12} | {'Outcome':<18} | {'M-NFEs':<6} | {'AB-NFEs':<6}"
            print(header)
            print("-" * len(header))
            for r in q_rows:
                m_str = f"{'✓' if r.get('medal_exact_match') else '✗'} ({r.get('medal_predicted') or 'N/A'})"
                ab_str = f"{'✓' if r.get('ab_medal_exact_match') else '✗'} ({r.get('ab_medal_predicted') or 'N/A'})"
                print(f"{r['example_id']:<15} | {str(r.get('reference_answer')):<8} | {m_str:<12} | {ab_str:<12} | {r.get('outcome', ''):<18} | {str(r.get('medal_total_nfes') or '-'):<6} | {str(r.get('ab_medal_total_nfes') or '-'):<6}")

            print("\n" + "=" * 80)
            print("AGGREGATE ANALYSIS SUMMARY")
            print("=" * 80)
            h2h = agg.get("head_to_head", {})
            print(f"Total Questions Compared: {h2h.get('compared_count', 0)}")
            print(f"  AB-MEDAL Wins: {h2h.get('ab_medal_wins', 0)}")
            print(f"  MEDAL Wins:    {h2h.get('medal_wins', 0)}")
            print(f"  Both Correct:  {h2h.get('both_correct_ties', 0)}")
            print(f"  Both Incorrect:{h2h.get('both_incorrect_ties', 0)}")

            for m, stats in agg.get("methods", {}).items():
                print(f"\n  [{m.upper()}] Accuracy: {stats['accuracy_percent']:.1f}% [{stats['ci_95_low']*100:.1f}%, {stats['ci_95_high']*100:.1f}%], Mean NFEs: {stats['mean_total_nfes']:.1f}, Mean Depth: {stats['mean_selected_depth']:.1f}")

    # 2. Discover available questions
    available = discover_available_questions(stage_dir, args.budget)
    if not available:
        print(f"\n[Error] No evaluated questions found in {stage_dir} for budget {args.budget}.")
        print("Check that data exists or run experiment 1 first.")
        return 1

    # 3. Resolve Target Question
    target_q = resolve_question_id(available, args.question_idx, args.example_id, args.question)
    if not target_q:
        if args.question_idx is None and args.example_id is None and args.question is None:
            # Default to first available question if none specified
            target_q = available[0]
            print(f"\n[Info] No question specified; defaulting to first available: {target_q['example_id']}")
        else:
            print(f"\n[Error] Could not find requested question in available set of {len(available)} questions.")
            print("Available questions:")
            for idx, q in enumerate(available):
                print(f"  [{idx}] {q['example_id']} | methods: {q['methods']} | {q['question'][:60]}...")
            return 1

    ex_id = target_q["example_id"]
    print("\n" + "=" * 80)
    print(f"TARGET QUESTION: {ex_id}")
    print("=" * 80)
    print(f"Question:\n{target_q.get('question')}\n")
    print(f"Reference Answer: {target_q.get('reference_answer')}")

    # 4. Candidate Analysis at Depth with Candidates >= C
    print("\n" + "=" * 80)
    print(f"DISTINCT CANDIDATE ANALYSIS (Threshold C = {args.min_c})")
    print("=" * 80)

    for m in ("medal", "ab_medal"):
        tree_file = stage_dir / m / ex_id / "checkpoints" / str(args.budget) / "tree.json"
        sel_file = stage_dir / m / ex_id / "checkpoints" / str(args.budget) / "selection.json"
        if not tree_file.is_file():
            print(f"\n[{m.upper()}] tree.json not found at {tree_file}")
            continue

        tree_data = json.loads(tree_file.read_text())
        sel_data = json.loads(sel_file.read_text()) if sel_file.is_file() else {}
        cand_stats = extract_distinct_candidates(tree_data, min_c=args.min_c, selection_record=sel_data)

        d_star = cand_stats["candidate_depth"]
        c_count = cand_stats["candidate_count"]
        qualifying = cand_stats["qualifying_depths"]
        dist = cand_stats["distinct_candidates_by_depth"]

        print(f"\n--- {m.upper()} Candidate Results ---")
        print(f"  Max Depth Reached:               {cand_stats['max_depth_reached']}")
        print(f"  Candidate Selection Depth (d*):  {d_star}")
        print(f"  Distinct Candidates at d*:       {c_count} (>= {args.min_c} threshold: {c_count >= args.min_c})")
        print(f"  Qualifying Depths (>= {args.min_c} states): {qualifying if qualifying else 'None (used fallback)'}")
        print(f"  Distinct Candidates by Depth:    {dist}")

        if args.show_candidates or len(cand_stats["candidates"]) <= 10:
            print(f"  Candidates at Depth {d_star} (ranked by mean Q):")
            for c in cand_stats["candidates"][:10]:
                sel_mark = " [SELECTED]" if c["is_selected"] else (" [SHORTLIST]" if c["is_shortlist"] else "")
                action_str = f"pos {c['action']['output_position']}->tok {c['action']['token_id']}" if c.get("action") else "root"
                print(f"    Node {c['node_id']:<4} | Visits: {c['visits']:<3} | Q: {c['mean_reward']:.4f} | Order: {c['evaluation_order']:<4} | Action: {action_str}{sel_mark}")

    # 5. Tree Drawing
    print("\n" + "=" * 80)
    print("RENDERING AB-MCTS COMPARISON TREES")
    print("=" * 80)

    # Verify both methods have checkpoints before drawing comparison
    missing_methods = [m for m in ("medal", "ab_medal") if not (stage_dir / m / ex_id / "checkpoints" / str(args.budget) / "tree.json").is_file()]
    if missing_methods:
        print(f"[Warning] Missing checkpoint tree files for {missing_methods}. Both 'medal' and 'ab_medal' are needed for side-by-side comparison.")
        return 0

    output_dir = args.output_dir or (stage_dir.parent / "summaries" / "e3_figures" / "trees" / stage_dir.name / ex_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = ("immediate_reward", "mean_reward") if args.color_by == "both" else (args.color_by,)
    for metric in metrics_to_plot:
        label = COLOR_METRICS[metric][0]
        print(f"\nRendering {label} tree...")
        paths = plot_e3_tree_comparison(
            results_dir=stage_dir,
            output_dir=output_dir,
            example_id=ex_id,
            budget=args.budget,
            color_by=metric,
        )
        print(f"  ✓ PNG preview: {paths['png']}")
        print(f"  ✓ Vector PDF:  {paths['pdf']}")

    print(f"\nAll trees generated successfully in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
