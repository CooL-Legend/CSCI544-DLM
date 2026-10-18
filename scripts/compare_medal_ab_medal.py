"""Paired GSM8K MEDAL / Adaptive MEDAL experiments with explicit coverage.

Run the single-example full-target gate before real-model comparisons. Target
experiments are uncapped by default; budget experiments cap initialization NFEs
only (root included). Incomplete attempts are saved and never decoded.

Use --from-results DIR [DIR ...] to regenerate summaries/plots without a model.
"""

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import mean

try:  # Direct script execution and imports from repository tests.
    from scripts.run_ab_medal_single import (
        add_common_arguments, load_inputs, prepare_demo_prompt, run_attempt, search_config,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    # python -I excludes the script directory from sys.path. Load this sibling
    # explicitly, without requiring a fourth production module or path edits.
    spec = importlib.util.spec_from_file_location("ab_medal_single", Path(__file__).with_name("run_ab_medal_single.py"))
    single = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(single)
    add_common_arguments = single.add_common_arguments
    load_inputs = single.load_inputs
    prepare_demo_prompt = single.prepare_demo_prompt
    run_attempt = single.run_attempt
    search_config = single.search_config

from dlm_project.decoding.mcts_decoding import derive_seed
from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.paths import create_run_directory
from dlm_project.experiments.sweep_report import write_csv


METHODS = ("medal", "ab_medal")


def _mean(values):
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def _scored(row):
    return (row["status"] == "complete" and row["target_reached"]
            and row.get("score") is not None and row["score"]["exact_match"] is not None)


def summarize_attempts(rows, *, expected_pairs=None):
    """Primary accuracy is paired; conditional accuracy always carries coverage."""
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    groups = []
    for condition in conditions:
        group = [r for r in rows if r["condition"] == condition]
        pairs = {}
        for row in group:
            key = (row["base_seed"], row["example_id"])
            if row["method"] in pairs.setdefault(key, {}):
                raise ValueError(f"Duplicate paired attempt: {condition}, {key}, {row['method']}.")
            pairs[key][row["method"]] = row
        common = [pair for pair in pairs.values() if all(m in pair and _scored(pair[m]) for m in METHODS)]
        requested = expected_pairs if expected_pairs is not None else len(pairs)
        for method in METHODS:
            attempts = [r for r in group if r["method"] == method]
            scored = [r for r in attempts if _scored(r)]
            paired = [pair[method] for pair in common]
            structural = [r["structure"] for r in attempts]
            stages = {}
            for row in attempts:
                for stage in row.get("stages", []):
                    if stage["stage"] == "Initialization":
                        stages.setdefault(stage["substage"], []).append(stage["wall_seconds"])
            groups.append({
                "condition": condition, "method": method, "requested_pairs": requested,
                "attempted": len(attempts), "target_attained": sum(r["target_reached"] for r in attempts),
                "target_coverage": sum(r["target_reached"] for r in attempts) / requested if requested else None,
                "scored": len(scored), "conditional_accuracy": _mean([float(r["score"]["exact_match"]) for r in scored]),
                "paired_n": len(paired), "paired_coverage": len(paired) / requested if requested else None,
                "paired_accuracy": _mean([float(r["score"]["exact_match"]) for r in paired]),
                "mean_total_nfe": _mean([r["total_nfe"] for r in attempts]),
                "mean_initialization_nfe": _mean([r["initialization_nfe"] for r in attempts]),
                "mean_total_wall_seconds": _mean([r["total_wall_seconds"] for r in attempts]),
                "paired_mean_total_nfe": _mean([r["total_nfe"] for r in paired]),
                "paired_mean_total_wall_seconds": _mean([r["total_wall_seconds"] for r in paired]),
                "mean_expanded_parents": _mean([r["expanded_parents"] for r in structural]),
                "mean_evaluated_children": _mean([r["evaluated_children"] for r in structural]),
                "mean_evaluated_depth": _mean([r["mean_evaluated_depth"] for r in structural]),
                "mean_max_depth": _mean([r["max_evaluated_depth"] for r in structural]),
                "mean_branching_factor": _mean([r["mean_branching_factor"] for r in structural]),
                "mean_candidate_diversity": _mean([r["candidate_diversity"]["mean_jaccard_distance"] for r in structural]),
                "mean_initialization_cost_seconds": {name: mean(values) for name, values in stages.items()},
            })
    return {"accuracy_cohort": "same condition, base seed and example; both methods completed and scored",
            "structural_cohort": "all attempts, including incomplete searches",
            "groups": groups, "attempts": len(rows),
            "observed_full_target_dlm_gate": any(r.get("full_target_dlm_gate", False) for r in rows)}


def plot_comparisons(rows, summary, directory):
    """Read only saved costs and structures. These plots never invoke a model."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    groups = summary["groups"]
    conditions = list(dict.fromkeys(g["condition"] for g in groups))
    colors = {"medal": "#4e79a7", "ab_medal": "#e15759"}

    def save(figure, name):
        FigureCanvasAgg(figure).print_png(str(directory / name))

    for field, label, filename in (
        ("paired_mean_total_nfe", "Mean actual total NFEs (paired complete cohort)", "accuracy_vs_nfe.png"),
        ("paired_mean_total_wall_seconds", "Mean pipeline seconds (paired complete cohort)", "accuracy_vs_time.png"),
    ):
        figure = Figure(figsize=(9, 5), constrained_layout=True)
        axis = figure.subplots()
        plotted = False
        for method in METHODS:
            points = [g for g in groups if g["method"] == method and g["paired_accuracy"] is not None]
            points.sort(key=lambda g: g[field])
            if points:
                plotted = True
                axis.plot([g[field] for g in points], [g["paired_accuracy"] for g in points], "o-",
                          label=method, color=colors[method])
                for point in points:
                    axis.annotate(f"{point['condition']} n={point['paired_n']}/{point['requested_pairs']}",
                                  (point[field], point["paired_accuracy"]), xytext=(3, 7), textcoords="offset points", fontsize=7)
        if not plotted:
            axis.text(.5, .5, "Accuracy unavailable: no examples completed by both methods", ha="center", transform=axis.transAxes)
        else:
            axis.legend()
        axis.set(xlabel=label, ylabel="Paired exact-match accuracy", ylim=(-.03, 1.12))
        axis.set_title("Coverage is shown as n/N; incomplete attempts have no correctness score")
        save(figure, filename)

    for field, label, filename in (
        ("mean_expanded_parents", "Expanded parents", "expanded_nodes.png"),
        ("mean_evaluated_depth", "Mean evaluated non-root depth", "depth.png"),
        ("mean_branching_factor", "Mean evaluated children per expanded parent", "branching.png"),
        ("target_coverage", "Fraction attaining the full candidate target", "target_coverage.png"),
    ):
        figure = Figure(figsize=(max(8, len(conditions) * 2), 5), constrained_layout=True)
        axis = figure.subplots()
        for offset, method in ((-.18, "medal"), (.18, "ab_medal")):
            values = [next((g[field] for g in groups if g["condition"] == c and g["method"] == method), None) for c in conditions]
            for index, value in enumerate(values):
                if value is not None:
                    axis.bar(index + offset, value, width=.35, color=colors[method], label=method if index == 0 else None)
        axis.set_xticks(range(len(conditions)), conditions)
        axis.set_ylabel(label)
        axis.set_title("All saved attempts, including incomplete searches")
        axis.legend()
        save(figure, filename)

    figure = Figure(figsize=(max(10, len(groups) * 1.6), 6), constrained_layout=True)
    axis = figure.subplots()
    stages = list(dict.fromkeys(name for g in groups for name in g["mean_initialization_cost_seconds"]))
    bottoms = [0.0] * len(groups)
    for name in stages:
        values = [g["mean_initialization_cost_seconds"].get(name, 0.0) for g in groups]
        axis.bar(range(len(groups)), values, bottom=bottoms, label=name)
        bottoms = [a + b for a, b in zip(bottoms, values)]
    axis.set_xticks(range(len(groups)), [f"{g['condition']}\n{g['method']}" for g in groups], rotation=15)
    axis.set_ylabel("Mean initialization seconds (exclusive stages)")
    if stages:
        axis.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    save(figure, "initialization_cost.png")

    # Depth and branching distributions retain details hidden by their means.
    figure = Figure(figsize=(12, 5), constrained_layout=True)
    axes = figure.subplots(1, 2)
    for method in METHODS:
        attempts = [r for r in rows if r["method"] == method]
        for axis, key, label in ((axes[0], "depth_histogram", "Evaluated depth"),
                                 (axes[1], "branching_factor_distribution", "Children per expanded parent")):
            counts = {}
            for row in attempts:
                for value, count in row["structure"][key].items():
                    counts[int(value)] = counts.get(int(value), 0) + count
            if counts:
                xs = sorted(counts)
                axis.plot(xs, [counts[x] for x in xs], "o-", label=method, color=colors[method])
            axis.set(xlabel=label, ylabel="Count across all saved attempts")
            if counts:
                axis.legend()
    save(figure, "structural_distributions.png")


def save_comparison(directory, rows, *, expected_pairs=None, status="complete", error=None, plots=True):
    summary = summarize_attempts(rows, expected_pairs=expected_pairs)
    summary.update(status=status, error=error)
    save_json(directory / "summary.json", summary)
    write_csv(directory / "summary.csv", summary["groups"])
    flat = []
    for row in rows:
        flat.append({key: row.get(key) for key in (
            "condition", "method", "example_id", "base_seed", "seed", "status", "target_reached",
            "accuracy", "initialization_nfe", "decoding_nfe", "total_nfe", "total_model_calls",
            "initialization_wall_seconds", "decoding_wall_seconds", "total_wall_seconds",
            "artifact_directory", "error")})
        flat[-1].update({key: row["structure"][key] for key in (
            "expanded_parents", "evaluated_children", "unique_states", "mean_evaluated_depth",
            "max_evaluated_depth", "mean_branching_factor", "candidate_diversity")})
    write_csv(directory / "attempts.csv", flat)
    if rows and plots:
        plot_comparisons(rows, summary, directory)
    return summary


def read_attempts(sources):
    rows = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            path = path / ("attempts.jsonl" if (path / "attempts.jsonl").is_file() else "metrics.json")
        if path.suffix == ".jsonl":
            incoming = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            incoming = [json.loads(path.read_text())]
        for row in incoming:
            row.setdefault("condition", "saved_search")
            row.setdefault("base_seed", row["seed"])
            row.setdefault("example_id", row.get("metadata", {}).get("example_id", "single"))
            # The structure was computed from the saved tree at search time.
            # Loading it adds no forwards and preserves incomplete attempts.
            rows.append(row)
    return rows


def run_comparison(args):
    for seed in args.seeds:
        derive_seed(seed, "validation")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.budgets)) != len(args.budgets):
        raise ValueError("Seeds and budgets must be distinct.")
    conditions = []
    if args.experiment in ("target", "all"):
        conditions.append(("target", args.target_max_nfe))
    if args.experiment in ("budget", "all"):
        conditions.extend((f"budget_{budget}", budget) for budget in args.budgets)
    configs = {(name, method): search_config(args, cap, adaptive=method == "ab_medal")
               for name, cap in conditions for method in METHODS}
    wrapper, manifest, setup = load_inputs(args)
    from dlm_project.tasks.gsm8k.benchmark import example_seed
    examples = manifest["examples"][args.start_index:args.start_index + args.num_examples]
    if len(examples) != args.num_examples:
        raise ValueError("Requested example range exceeds the pinned manifest.")
    directory, _ = create_run_directory({"output_dir": str(args.output_dir)})
    save_json(directory / "config.json", {"cli": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                           "dataset": manifest["dataset"], "setup": setup,
                                           "models_loaded": 1, "budget_scope": "initialization including root",
                                           "accuracy_cohort": "paired completed and scored examples"})
    rows, error, status = [], None, "running"
    expected = len(examples) * len(args.seeds)
    save_comparison(directory, rows, expected_pairs=expected, status=status, plots=False)
    print(f"Comparison artifacts: {directory}", flush=True)
    try:
        with (directory / "attempts.jsonl").open("w", encoding="utf-8") as attempts:
            for example in examples:
                prompt, ids, prompt_meta = prepare_demo_prompt(
                    wrapper, example["question"], args.demonstrations_path,
                    revision=manifest["dataset"]["revision"], cache_dir=args.cache_dir)
                save_json(directory / f"prompt_{example['row_id']}.json", {"text": prompt, "metadata": prompt_meta})
                for base_seed in args.seeds:
                    seed = example_seed(base_seed, example["row_id"])
                    for condition, _ in conditions:
                        # Alternate execution order while keeping paired inputs
                        # and completion seeds exactly identical.
                        order = METHODS if example["row_id"] % 2 == 0 else tuple(reversed(METHODS))
                        for method in order:
                            attempt_dir = directory / condition / f"seed{base_seed}_example{example['row_id']}" / method
                            report = run_attempt(
                                wrapper, ids, configs[condition, method], attempt_dir, method=method,
                                seed=seed, gen_length=args.gen_length, temperature=args.temperature,
                                reference=example["reference_solution"], metadata=example,
                                prompt_seconds=prompt_meta["prompt_wall_seconds"], setup_seconds=prompt_meta["setup_wall_seconds"],
                                plot_max_nodes=args.plot_max_nodes, plot_max_depth=args.plot_max_depth,
                                print_history=not args.quiet_history)
                            report.update(condition=condition, base_seed=base_seed, example_id=example["example_id"])
                            rows.append(report)
                            attempts.write(json.dumps(report, allow_nan=False) + "\n")
                            attempts.flush()
                            save_comparison(directory, rows, expected_pairs=expected, status="running", plots=False)
                            if report["status"] == "interrupted":
                                raise KeyboardInterrupt(report["error"])
                            if report["status"] == "error":
                                raise RuntimeError(report["error"])
        status = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        summary = save_comparison(directory, rows, expected_pairs=expected, status=status, error=error)
    return directory, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--experiment", choices=("target", "budget", "all"), default="budget")
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--target-max-nfe", type=int, help="Optional cap for target runs; default is uncapped.")
    parser.add_argument("--quiet-history", action="store_true", help="Save full history while suppressing stdout events.")
    parser.add_argument("--from-results", type=Path, nargs="+", help="Replot saved attempt directories/JSONL; no inference.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/ab_medal/comparison"))
    args = parser.parse_args(argv)
    if not 1 <= args.num_examples <= 1319 or not 0 <= args.start_index < 1319 or args.start_index + args.num_examples > 1319:
        parser.error("Invalid GSM8K example range.")
    if args.from_results:
        directory, _ = create_run_directory({"output_dir": str(args.output_dir)})
        rows = read_attempts(args.from_results)
        summary = save_comparison(directory, rows)
        print(f"Saved-search report (zero inference): {directory}")
    else:
        directory, summary = run_comparison(args)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
