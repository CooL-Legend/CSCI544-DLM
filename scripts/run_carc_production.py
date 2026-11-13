"""Launch the locked 500-question run using a validated pilot on matching hardware."""

import argparse
import json
import csv
from pathlib import Path

import torch

from dlm_project.experiments.resources import _revision, source_fingerprint
from dlm_project.models.cache import validate_snapshot
from dlm_project.models.profiles import get_profile
from scripts.run_e1_test import main as run_test


def verify_production(directory):
    from dlm_project.tasks.gsm8k.experiment_1 import context, is_question_complete
    ctx = context(stage="test", num_samples=500)
    expected = {(ex["example_id"], method, budget) for ex in ctx["examples"]
                for method in ("medal", "ab_medal") for budget in (256, 512)}
    summaries = Path(directory) / "summaries"
    for name in ("e1_test_per_question.csv", "e3_test_per_question.csv"):
        with (summaries / name).open() as stream:
            rows = list(csv.DictReader(stream))
        keys = {(r["example_id"], r["method"], int(r["search_budget"])) for r in rows}
        if len(rows) != 2000 or keys != expected:
            raise ValueError(f"Incomplete or duplicated production outcomes in {name}.")
    for name in ("e1_test_aggregate.csv", "e3_test_aggregate.csv"):
        with (summaries / name).open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 4 or any(int(r["sample_count"]) != 500 for r in rows):
            raise ValueError(f"Expected four 500-question aggregate rows in {name}.")
    for ex in ctx["examples"]:
        for method in ("medal", "ab_medal"):
            if not is_question_complete(Path(directory) / "test", method, ex["example_id"], (256, 512)):
                raise ValueError(f"Incomplete checkpoint artifacts: {method}/{ex['example_id']}")
    for stem in ("fig1_nfe_by_depth", "fig2_branching_factors", "fig3_candidate_density",
                 "fig4_root_subtrees", "fig5_reward_redundancy", "fig6_accuracy_vs_budget"):
        for suffix in ("png", "pdf"):
            if not (summaries / "e3_figures" / f"{stem}.{suffix}").is_file():
                raise ValueError(f"Missing E3 figure {stem}.{suffix}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = json.loads(args.benchmark_report.read_text())
    batch = report.get("recommended_batch_size")
    if not report.get("production_validation_passed") or batch not in (1, 4, 8):
        parser.error("The pilot did not pass memory, length-stress and parity checks.")
    summary = report["results"][str(batch)]["summary"]
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; launch on an allocated GPU node.")
    if (summary["gpu_name"] != torch.cuda.get_device_name(0)
            or summary["torch_version"] != torch.__version__
            or summary["cuda_version"] != torch.version.cuda
            or summary["repository_commit"] != _revision()
            or summary.get("source_fingerprint") != source_fingerprint()):
        parser.error("GPU, software or repository differs from the pilot; benchmark this environment first.")
    validate_snapshot(args.model_path, "llada")
    source_path = Path(args.model_path) / ".model_source.json"
    if not source_path.exists():
        parser.error("Stage the pinned checkpoint with scripts/cache_model.py to record its provenance.")
    source = json.loads(source_path.read_text())
    profile = get_profile("llada")
    if any(source.get(k) != profile[k] or summary["model"].get(k) != profile[k]
           for k in ("repo_id", "revision")):
        parser.error("The staged model and pilot must use the pinned LLaDA checkpoint.")
    code = run_test(["--num-samples", "500", "--batch-size", str(batch), "--profile-resources",
                     "--model-path", args.model_path, "--local-files-only", "--output-dir", str(args.output_dir)])
    verify_production(args.output_dir)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
