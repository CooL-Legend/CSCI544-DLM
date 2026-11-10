"""Offline E0 accuracy/compute reports; no tokenizer, weights, or dataset downloads."""

import json
import math
from pathlib import Path
from statistics import mean, median

from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.manifests import digest
from dlm_project.experiments.sweep_report import percentile, wilson_interval, write_csv
from dlm_project.tasks.gsm8k.e0_prompt import build_e0_prompt
from dlm_project.tasks.gsm8k.evaluation import score_answer
from dlm_project.tasks.gsm8k.experiment_protocol import validate_manifest, select_examples


# Fixed-budget E0 protocols. v5: deepest evaluated depth. v6: Lc = 10 and the
# deepest depth holding at least C distinct states.
FIXED_BUDGET_VERSIONS = ("e0_fixed_budget_v5", "e0_fixed_budget_v6")


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
            "candidate_count", "target_depth_candidate_count", "shortlist_size", "budget_utilization",
            "expanded_parents", "evaluated_children", "realized_branching_factor")
    for key in keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        result["mean_" + key] = mean(values) if values else None
    for key in ("search_nfes", "completion_nfes", "total_nfes", "decoding_seconds"):
        result["total_" + key] = sum(row[key] for row in rows)
    audited = [row["search_audit_metrics"] for row in rows if row.get("search_audit_metrics")]
    result["allocation_audited_n"] = len(audited)
    for key in ("branching_factor_min", "branching_factor_max", "unique_evaluated_states",
                "duplicate_state_count", "duplicate_state_percentage", "root_subtree_coverage",
                "negative_ig_count", "negative_ig_percentage", "ig_min", "ig_mean", "ig_max", "selected_q"):
        values = [m[key] for m in audited if m.get(key) is not None]
        result["mean_" + key] = mean(values) if values else None
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
            "run_order": config["evaluation"].get("run_order", "example"),
            "search_mode": config["evaluation"]["search_mode"], "init_tokens": config["decoding"]["mcts"].get("init_tokens"),
            "depths": sorted({c["requested_depth"] for c in conditions if c.get("requested_depth") is not None}),
            "common_protocol_hash": config.get("common_protocol_hash"),
            "status": setup["status"], "complete": setup["status"] == "completed" and len(terminal) == expected,
            "requested_n": len(requested), "common_n": len(common), "common_example_ids": common,
            "expected_predictions": expected, "terminal_predictions": len(terminal),
            "comparison_sample": ("questions with completed outcomes under every condition"
                                  if config["protocol_version"] in FIXED_BUDGET_VERSIONS else
                                  "questions with a terminal outcome under every condition; incomplete searches count as incorrect"),
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
    current = config["protocol_version"] in FIXED_BUDGET_VERSIONS
    if current:
        common = config["common_protocol"]
        validate_manifest(manifest, common)
        expected_ids = [r["example_id"] for r in select_examples(manifest, config["evaluation"]["stage"])]
        if config["data"].get("num_samples") is not None:
            expected_ids = expected_ids[:config["data"]["num_samples"]]
        if (digest(common) != config["common_protocol_hash"]
                or config["run"]["example_ids"] != expected_ids
                or config["prompt"] != common["prompt"]
                or config["prompt_template_hash"] != digest(common["prompt"]["template"])):
            raise ValueError("E0 shared protocol, stage selection or prompt mismatch.")
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
        if current:
            _validate_current_record(row, condition, config)
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
        expected_completion = (condition["completion_method"] if current else
                               condition.get("completion_method", "medal") if condition["method"] in ("medal", "medal_no_search") else None)
        if row.get("completion_method", expected_completion) != expected_completion:
            raise ValueError("E0 completion decoder identity mismatch.")
        if not condition["search_budget"]:
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
        if row.get("search_audit_file"):
            from dlm_project.decoding import search_audit
            audit = json.loads((directory / row["search_audit_file"]).read_text())
            prompt_record = json.loads((directory / "prompts" / f"{row['example_id'].replace(':', '_')}.json").read_text())
            if digest([prompt_record["prompt_ids"]]) != row["prompt_ids_hash"]:
                raise ValueError("Saved audit prompt token IDs differ from the E0 prompt hash.")
            search_audit.reconstruct(audit, prompt_record["prompt_ids"])
            measured = search_audit.metrics(audit)
            if measured != row["search_audit_metrics"] or measured["search_nfes"] != row["search_nfes"]:
                raise ValueError("E0 offline allocation metrics differ from saved evidence.")
        elif row["search_nfes"] and row.get("trace_file") and (directory / row["trace_file"]).is_file():
            from dlm_project.decoding import search_audit
            events = [json.loads(line) for line in (directory / row["trace_file"]).read_text().splitlines() if line.strip()]
            recovered = search_audit.from_events(events, mask_id=config["provenance"]["model"].get("mask_id", 126336),
                gen_length=config["generation"]["gen_length"], requested_depth=row.get("requested_depth") or config["decoding"]["mcts"].get("init_tokens"),
                requested_budget=condition["search_budget"], search_nfes=row["search_nfes"],
                stop_mode=config["evaluation"]["search_mode"], allow_legacy=True)
            if recovered["nodes"]:
                row["search_audit_metrics"] = search_audit.metrics(recovered)
        seen.add(key)
        rows.append(row)
    return rows, warnings


def _validate_current_record(row, condition, config):
    """Check fixed-budget selection evidence and metric semantics without model access."""
    if (row["common_protocol_hash"] != config["common_protocol_hash"]
            or row["prompt_template_hash"] != config["prompt_template_hash"]
            or row["requested_depth"] != condition["requested_depth"]
            or row["generation_length"] != config["generation"]["gen_length"]
            or row["status"] != "completed"):
        raise ValueError("E0 fixed-budget record identity mismatch.")
    for field in ("search_seconds", "completion_seconds", "decoding_seconds", "average_search_depth",
                  "realized_branching_factor"):
        if not math.isfinite(row[field]) or row[field] < 0:
            raise ValueError("Invalid E0 metric.")
    for field in ("selected_depth", "max_depth", "candidate_count", "target_depth_candidate_count",
                  "shortlist_size", "expanded_parents", "evaluated_children"):
        if type(row[field]) is not int or row[field] < 0:
            raise ValueError("Invalid E0 count.")
    if condition["method"] in ("confidence", "entropy"):
        if (any(row[k] for k in ("selected_depth", "max_depth", "candidate_count", "shortlist_size",
                                "target_depth_candidate_count", "expanded_parents", "evaluated_children",
                                "fallback_decode_used", "shortlist_underfilled", "realized_branching_factor"))
                or row["budget_utilization"] is not None):
            raise ValueError("No-search baseline contains search metrics.")
        return

    depth = row["selected_depth"]
    details = row["details"]
    candidates, shortlist = details["candidate_evaluations"], details["candidate_shortlist"]
    limit, lc = config["decoding"]["mcts"].get("num_candidates", 3), condition["requested_depth"]
    if config["protocol_version"] == "e0_fixed_budget_v5":
        depth_ok = (0 <= depth == row["max_depth"] <= lc
                    and row["target_depth_candidate_count"] == (len(candidates) if depth == lc else 0))
    else:
        # Deepest depth with >= C distinct states; else the most-populated depth.
        # A shallower choice implies fewer than C distinct states at Lc.
        depth_ok = (0 <= depth <= row["max_depth"] <= lc
                    and (len(candidates) >= limit or depth == row["max_depth"] or depth == 0
                         or row["target_depth_candidate_count"] <= len(candidates))
                    and (row["target_depth_candidate_count"] == len(candidates) if depth == lc
                         else row["target_depth_candidate_count"] < limit))
    if (not candidates or len(candidates) != row["candidate_count"]
            or len({c["state_hash"] for c in candidates}) != len(candidates)
            or len({tuple(c["output_ids"]) for c in candidates}) != len(candidates)
            or not depth_ok
            or row["fallback_decode_used"] != (depth < lc)
            or row["shortlist_size"] != min(3, len(candidates))
            or row["shortlist_underfilled"] != (len(candidates) < 3)
            or row["budget_utilization"] != row["search_nfes"] / condition["search_budget"]
            or row["stop_reason"] not in ("model_call_budget", "frontier_exhausted")
            or (row["stop_reason"] == "model_call_budget" and row["search_nfes"] != condition["search_budget"])
            or details["candidate_evaluation_model_calls"] != 0):
        raise ValueError("E0 candidate/depth/budget metrics mismatch.")
    for candidate in candidates:
        visits, total = candidate["visits"], candidate["reward_sum"]
        if (type(visits) is not int or visits < 0 or (visits == 0 and depth != 0)
                or not math.isfinite(total) or candidate["depth"] != depth
                or candidate["q_value"] != (total / visits if visits else 0.0)
                or candidate["reward"] != candidate["q_value"]):
            raise ValueError("E0 candidate expected reward mismatch.")
    ranked = sorted(candidates, key=lambda c: (-c["q_value"], c["node_id"]))[:3]
    if ([c["node_id"] for c in shortlist] != [c["node_id"] for c in ranked]
            or any(any(c[key] != ranked[i][key] for key in ("depth", "visits", "reward_sum", "q_value"))
                   for i, c in enumerate(shortlist))
            or details["selected_node_id"] != ranked[0]["node_id"]
            or row["selected_score"] != ranked[0]["q_value"]
            or details["selected_initialization"]["output_ids"] != ranked[0]["output_ids"]):
        raise ValueError("E0 shortlist or selected state mismatch.")
    parents, children = row["expanded_parents"], row["evaluated_children"]
    if row["realized_branching_factor"] != (children / parents if parents else 0.0):
        raise ValueError("E0 branching metric mismatch.")


def _depth_label(summary):
    return ",".join(map(str, summary["depths"])) if summary["depths"] else str(summary["init_tokens"])


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
               title=f"E0 · {summary['search_mode']} · Lc={_depth_label(summary)} · n={summary['common_n']}")
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
    write_csv(destination / "compute_allocation.csv", [
        {"example_id": row["example_id"], "condition": row["condition"], **row["search_audit_metrics"]}
        for row in rows if row.get("search_audit_metrics")])
    _plots(summary, destination / "plots")

    def number(value):
        return "—" if value is None else f"{value:.3f}"

    current = config["protocol_version"] in FIXED_BUDGET_VERSIONS
    diagnostic_header = "Stop reasons" if current else "Quota failures"
    lines = ["# Experiment 0: Baseline Test", "",
             f"Stage: **{summary['stage']}** · mode: **{summary['search_mode']}** · Lc={_depth_label(summary)} · "
             f"status: **{summary['status']}** · paired questions: **{summary['common_n']}/{summary['requested_n']}**.", "",
             f"Execution order: **{summary['run_order']}** (condition means all selected questions per method).", "",
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
              "", f"| Search condition | Selected / maximum / average depth | Candidates / shortlist | Budget utilization | {diagnostic_header} |",
              "|---|---:|---:|---:|---:|"]
    for item in summary["conditions"]:
        if item["method"] == "medal":
            diagnostic = (", ".join(f"{reason}: {count}" for reason, count in item["stop_reasons"].items())
                          if current else str(item["search_incomplete_count"]))
            lines.append(f"| {item['id']} | {number(item['mean_selected_depth'])} / {number(item['mean_max_depth'])} / "
                         f"{number(item['mean_average_search_depth'])} | {number(item['mean_candidate_count'])} / "
                         f"{number(item['mean_shortlist_size'])} | {number(item['mean_budget_utilization'])} | "
                         f"{diagnostic or '—'} |")
    lines += ["", "| Search condition | Baseline | Accuracy gain (pp) | Additional mean NFEs | pp per additional NFE |",
              "|---|---|---:|---:|---:|"]
    for item in summary["gains"]:
        lines.append(f"| {item['condition']} | {item['baseline']} | {number(item['accuracy_gain_pp'])} | "
                     f"{number(item['additional_mean_nfes'])} | {number(item['accuracy_gain_pp_per_additional_nfe'])} |")
    decoder = config.get("decoding", {}).get("completion_method", "medal")
    controls = [c for c in config["conditions"] if c["method"] == "medal_no_search"]
    completion_note = (f"The T={controls[0]['temperature']:g} no-search MEDAL control uses the identical downstream decoder, temperature and completion seed as searched MEDAL. "
                       if controls else f"All {len(config['conditions'])} conditions use LLaDA Confidence completion at T=0. ")
    if config["protocol_version"] in FIXED_BUDGET_VERSIONS:
        lines += ["", f"Common protocol SHA256: `{summary['common_protocol_hash']}`.", "",
                  "| Condition | Parse rate | Search seconds | Completion seconds | Expanded parents | Mean branching | Requested-depth candidates |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for item in summary["conditions"]:
            values = [number(item[key]) for key in ("parse_rate", "mean_search_seconds", "mean_completion_seconds",
                      "mean_expanded_parents", "mean_realized_branching_factor", "mean_target_depth_candidate_count")]
            lines.append(f"| {item['id']} | " + " | ".join(values) + " |")
        lines += ["", "Candidate counts refer to all distinct states at the deepest evaluated depth, including fallback. "
                  "The shortlist retains up to three ranked by mean backed-up reward; only its first state is completed. "
                  "Requested-depth candidate counts are recorded separately. Fallback means selected depth < requested depth."]
    lines += ["", f"MEDAL completion decoder: **{decoder}**. " + completion_note +
              "Greedy baselines use T=0. Search excludes EOS/EOT; every completion permits them and fills all positions. "
              "Scoring uses only the prefix before the first terminator, with the existing strict #### parser.",
              "", "Fallback outputs count normally. " + ("" if current else "Strict quota failures count as incorrect. ") + "Missing/interrupted outcomes "
              "are excluded from the paired comparison and remain visible in observed_summary.csv. "
              f"Unfinished attempt forwards: {summary['unfinished_attempt_nfes']}. "
              "Model loading, warm-up, tokenization, scoring, artifact writing and reporting are recorded separately in setup.json. "
              "Decode timing includes decoder bookkeeping and CUDA synchronization; disk trace writing is outside that interval.",
              "", "This is the repository's MEDAL reconstruction with the E0 protocol, not a claim of author-code parity.",
              "", *warnings]
    (destination / "report.md").write_text("\n".join(lines) + "\n")
    return summary
