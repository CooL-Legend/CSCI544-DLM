"""Attach Slurm MaxRSS after a job ends (accounting may take time to appear)."""

import argparse
import json
from pathlib import Path
import subprocess

from dlm_project.experiments.artifacts import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    if not args.job_id.isdigit():
        parser.error("--job-id must be a numeric Slurm job ID")
    result = subprocess.run(["sacct", "-j", args.job_id, "--noheader", "--parsable2",
                             "--format=JobID,State,Elapsed,MaxRSS"], check=True, capture_output=True, text=True)
    rows = [dict(zip(("job_id", "state", "elapsed", "max_rss"), line.strip().split("|")))
            for line in result.stdout.splitlines() if line.strip()]
    path = args.output_dir / "resource_summary.json"
    summary = json.loads(path.read_text())
    if summary.get("slurm_job_id") != args.job_id:
        parser.error("Job ID does not match the latest resource attempt in this directory.")
    summary["slurm_max_rss"] = {"scope": "per Slurm job step; values retain sacct units", "steps": rows}
    save_json(path, summary)
    save_json(args.output_dir / "resource_attempts" / f"{summary['attempt_id']}.json", summary)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
