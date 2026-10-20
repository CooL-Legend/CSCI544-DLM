"""Offline E0 accuracy/compute reports; no tokenizer, weights, or dataset downloads."""

import json
from pathlib import Path
from statistics import mean, median

from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.manifests import digest
from dlm_project.experiments.sweep_report import percentile, wilson_interval, write_csv
from dlm_project.tasks.gsm8k.e0_prompt import build_e0_prompt
from dlm_project.tasks.gsm8k.evaluation import score_answer


def _aggregate(condition, rows):
    n = len(rows)
    correct = sum(bool(row["exact_match"]) for row in rows)
    low, high = wilson_interval(correct, n)
    result = {**condition, "n": n, "correct": correct, "accuracy_percent": 100 * correct / n if n else None,
              "accuracy_ci95_low": low, "accuracy_ci95_high": high,
              "parse_rate": mean(row["parse_success"] for row in rows) if n else None,
              "search_incomplete_count": sum(row["status"] == "search_incomplete" for row in rows),
              "fallback_rate": mean(row["fallback_decode_used"] for row in rows) if n else None,
              "shortlist_underfilled_rate": mean(row["shortlist_underfilled"] for row in rows) if n else None}
    keys = ("search_nfes", "completion_nfes", "total_nfes", "decoding_seconds", "search_seconds",
            "completion_seconds", "selected_depth", "max_depth", "average_search_depth",
            "candidate_count", "shortlist_size", "budget_utilization")
    for key in keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        result["mean_" + key] = mean(values) if values else None
    for key in ("search_nfes", "completion_nfes", "total_nfes", "decoding_seconds"):
        result["total_" + key] = sum(row[key] for row in rows)
    latencies = [row["decoding_seconds"] for row in rows]
    result.update(median_decoding_seconds=median(latencies) if n else None,
                  p95_decoding_seconds=percentile(latencies, .95),
                  stop_reasons={reason: sum(row["stop_reason"] == reason for row in rows)
                                for reason in sorted({row["stop_reason"] for row in rows})})
    return result


def summarize(config, setup, records):
    conditions, requested = config["conditions"], config["run"]["example_ids"]
    terminal = [row for row in records if row["status"] in ("completed", "search_incomplete")]
    pairs = {(row["example_id"], row["condition"]) for row in terminal}
    common = [example for example in requested if all((example, c["id"]) in pairs for c in conditions)]
    paired = [row for row in terminal if row["example_id"] in common]
    summaries = [_aggregate(c, [row for row in paired if row["condition"] == c["id"]]) for c in conditions]
    gains = []
    for searched in (item for item in summaries if item["method"] == "medal"):
        for baseline in (item for item in summaries if item["method"] != "medal"):
            gain = (searched["accuracy_percent"] - baseline["accuracy_percent"]) if common else None
            extra = (searched["mean_total_nfes"] - baseline["mean_total_nfes"]) if common else None
            gains.append({"condition": searched["id"], "baseline": baseline["id"], "n": len(common),
                          "accuracy_gain_pp": gain, "additional_mean_nfes": extra,
                          "accuracy_gain_pp_per_additional_nfe": gain / extra if extra is not None and extra > 0 else None})
    expected = len(requested) * len(conditions)
    return {"protocol_version": config["protocol_version"], "stage": config["evaluation"]["stage"],
            "search_mode": config["evaluation"]["search_mode"], "init_tokens": config["decoding"]["mcts"]["init_tokens"],
            "status": setup["status"], "complete": setup["status"] == "completed" and len(terminal) == expected,
            "requested_n": len(requested), "common_n": len(common), "common_example_ids": common,
            "expected_predictions": expected, "terminal_predictions": len(terminal),
            "comparison_sample": "questions with a terminal outcome under every condition; incomplete searches count as incorrect",
            "conditions": summaries,
            "observed_conditions": [_aggregate(c, [row for row in terminal if row["condition"] == c["id"]]) for c in conditions],
            "gains": gains, "completed_prediction_nfes": sum(row["total_nfes"] for row in records),
            "unfinished_attempt_nfes": setup.get("active_total_nfes", 0),
            "model_load_count": setup.get("model_load_count", 0), "warmup_nfes": setup.get("warmup_nfes", 0),
            "error": setup.get("error")}


def load_records(directory, config):
    manifest = json.loads((directory / "manifest.json").read_text())
    if digest(manifest) != config["run"]["manifest_hash"]:
        raise ValueError("E0 manifest hash mismatch.")
    examples = {row["example_id"]: row for row in manifest["examples"]}
    conditions = {row["id"]: row for row in config["conditions"]}
    rows, seen, identities, warnings = [], set(), {}, []
    text = (directory / "predictions.jsonl").read_text()
    lines = text.splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not text.endswith("\n"):
                warnings.append("Ignored an incomplete trailing JSONL record; the raw file was retained.")
                break
            raise ValueError("Invalid E0 prediction JSONL.")
        key = (row["example_id"], row["condition"])
        if key in seen or key[0] not in config["run"]["example_ids"] or key[1] not in conditions:
            raise ValueError("Duplicate or unexpected E0 prediction.")
        example, condition = examples[key[0]], conditions[key[1]]
        expected_prompt = build_e0_prompt(example["question"], config["prompt"])
        expected_seed = (config["decoding"]["seed"] + 5 * example["row_id"]) % 2**63
        if (row["content_hash"] != example["content_hash"] or row["prompt"] != expected_prompt
                or row["prompt_hash"] != digest(expected_prompt) or row["seed"] != expected_seed
                or row["stage"] != config["evaluation"]["stage"]
                or row["method"] != condition["method"] or row["temperature"] != condition["temperature"]
                or row["model_revision"] != config["provenance"]["model"]["revision"]
                or row["dataset_revision"] != config["data"]["revision"]
                or row["requested_search_budget"] != condition["search_budget"]
                or row["protocol_version"] != config["protocol_version"]
                or row["total_nfes"] != row["search_nfes"] + row["completion_nfes"]
                or any(type(row[field]) is not int or row[field] < 0 for field in ("search_nfes", "completion_nfes", "total_nfes"))):
            raise ValueError("E0 identity or NFE accounting mismatch.")
        if condition["search_budget"] is None:
            if row["search_nfes"] != 0:
                raise ValueError("No-search baseline used search NFEs.")
        elif row["search_nfes"] > condition["search_budget"]:
            raise ValueError("Search exceeded its forward cap.")
        if row["status"] == "completed":
            if (row["completion_nfes"] != config["generation"]["gen_length"] - row["selected_depth"]
                    or len(row["full_output_ids"]) != config["generation"]["gen_length"]):
                raise ValueError("E0 completion length/NFEs mismatch.")
        elif row["status"] != "search_incomplete" or config["evaluation"]["search_mode"] != "candidates" or row["completion_nfes"]:
            raise ValueError("Unexpected terminal prediction status.")
        ids = row["full_output_ids"]
        first = next((i for i, token in enumerate(ids) if token in config["termination"]["token_ids"]), None)
        if row["first_terminator_position"] != first or row["scoring_output_ids"] != (ids if first is None else ids[:first]):
            raise ValueError("E0 termination prefix mismatch.")
        scores = score_answer(row["scoring_text"], example["reference_solution"])
        if any(row[key] != value for key, value in scores.items()):
            raise ValueError("E0 saved answer score mismatch.")
        identity = (row["prompt_ids_hash"], row["prompt_hash"], row["seed"], row["model_revision"])
        if identities.setdefault(key[0], identity) != identity:
            raise ValueError("E0 conditions did not share tokenized prompts and seeds.")
        seen.add(key)
        rows.append(row)
    return rows, warnings


def _plots(summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    for field, label, stem in (("mean_total_nfes", "Mean total NFEs (search + completion)", "accuracy_vs_total_nfes"),
                               ("mean_decoding_seconds", "Mean decoding runtime (seconds)", "accuracy_vs_runtime")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for index, item in enumerate(summary["conditions"]):
            if not item["n"]:
                continue
            accuracy = item["accuracy_percent"]
            ax.errorbar(item[field], accuracy,
                        yerr=[[max(0, accuracy - 100 * item["accuracy_ci95_low"])],
                              [max(0, 100 * item["accuracy_ci95_high"] - accuracy)]],
                        fmt="o" if item["method"] == "medal" else "s", capsize=3, label=item["id"])
        if summary["common_n"]:
            ax.legend(fontsize=8, loc="best")
        else:
            ax.text(.5, .5, "No paired outcomes yet", ha="center", transform=ax.transAxes)
        ax.set(xlabel=label, ylabel="Accuracy (%)", ylim=(-3, 103),
               title=f"E0 · {summary['search_mode']} · Lc={summary['init_tokens']} · n={summary['common_n']}")
        ax.grid(alpha=.2)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(destination / f"{stem}.{extension}", dpi=160)
        plt.close(fig)


def generate_report(run_dir):
    directory = Path(run_dir)
    config = json.loads((directory / "config.json").read_text())
    setup = json.loads((directory / "setup.json").read_text())
    rows, warnings = load_records(directory, config)
    summary = summarize(config, setup, rows)
    summary["warnings"] = warnings
    destination = directory / "report"
    destination.mkdir(exist_ok=True)
    save_json(destination / "summary.json", summary)
    save_json(destination / "plot_data.json", summary)
    write_csv(destination / "results.csv", rows, ("example_id", "condition", "status"))
    write_csv(destination / "summary.csv", summary["conditions"])
    write_csv(destination / "observed_summary.csv", summary["observed_conditions"])
    write_csv(destination / "accuracy_gains.csv", summary["gains"])
    _plots(summary, destination / "plots")

    def number(value):
        return "—" if value is None else f"{value:.3f}"

    lines = ["# Experiment 0: Baseline Test", "",
             f"Stage: **{summary['stage']}** · mode: **{summary['search_mode']}** · Lc={summary['init_tokens']} · "
             f"status: **{summary['status']}** · paired questions: **{summary['common_n']}/{summary['requested_n']}**.", "",
             "| Condition | Accuracy % (Wilson 95% CI) | Search NFEs | Completion NFEs | Total NFEs | Decode seconds | Fallback rate |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for item in summary["conditions"]:
        ci = ("—" if not item["n"] else
              f"{100 * item['accuracy_ci95_low']:.1f}–{100 * item['accuracy_ci95_high']:.1f}")
        values = [number(item[key]) for key in ("mean_search_nfes", "mean_completion_nfes", "mean_total_nfes",
                                                "mean_decoding_seconds", "fallback_rate")]
        lines.append(f"| {item['id']} | {number(item['accuracy_percent'])} ({ci}) | " + " | ".join(values) + " |")
    lines += ["", "NFEs and seconds above are means per paired question. The CSV also includes aggregate NFEs, "
              "search/completion runtime, median/p95 runtime, depths, shortlist counts, budget utilization, and termination reasons.",
              "", "| Search condition | Selected / maximum / average depth | Candidates / shortlist | Budget utilization | Quota failures |",
              "|---|---:|---:|---:|---:|"]
    for item in summary["conditions"]:
        if item["method"] == "medal":
            lines.append(f"| {item['id']} | {number(item['mean_selected_depth'])} / {number(item['mean_max_depth'])} / "
                         f"{number(item['mean_average_search_depth'])} | {number(item['mean_candidate_count'])} / "
                         f"{number(item['mean_shortlist_size'])} | {number(item['mean_budget_utilization'])} | "
                         f"{item['search_incomplete_count']} |")
    lines += ["", "| Search condition | Baseline | Accuracy gain (pp) | Additional mean NFEs | pp per additional NFE |",
              "|---|---|---:|---:|---:|"]
    for item in summary["gains"]:
        lines.append(f"| {item['condition']} | {item['baseline']} | {number(item['accuracy_gain_pp'])} | "
                     f"{number(item['additional_mean_nfes'])} | {number(item['accuracy_gain_pp_per_additional_nfe'])} |")
    lines += ["", "The T=1 no-search MEDAL control uses the identical downstream decoder and completion seed. "
              "Greedy baselines use T=0. Search excludes EOS/EOT; every completion permits them and fills all positions. "
              "Scoring uses only the prefix before the first terminator, with the existing strict #### parser.",
              "", "Fallback outputs count normally. Strict quota failures count as incorrect. Missing/interrupted outcomes "
              "are excluded from the paired comparison and remain visible in observed_summary.csv. "
              f"Unfinished attempt forwards: {summary['unfinished_attempt_nfes']}. "
              "Model loading, warm-up, tokenization, scoring, artifact writing and reporting are recorded separately in setup.json. "
              "Decode timing includes decoder bookkeeping and CUDA synchronization; disk trace writing is outside that interval.",
              "", "This is the repository's MEDAL reconstruction with the E0 protocol, not a claim of author-code parity.",
              "", *warnings]
    (destination / "report.md").write_text("\n".join(lines) + "\n")
    return summary
