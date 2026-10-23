"""Run pure LLaDA and/or fixed-budget MCTS sequentially using paths from .env."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import signal
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dlm_project.models.profiles import get_profile


METHODS = {"both": ["confidence", "medal"], "llada": ["confidence"], "mcts": ["medal"]}


def read_env(path):
    """Read literal KEY=value paths; never execute or source the file as a shell."""
    values = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"{path}:{number}: expected KEY=value.")
        parts = shlex.split(value, comments=True)
        if len(parts) > 1:
            raise ValueError(f"{path}:{number}: quote values containing spaces.")
        values[key] = parts[0] if parts else ""
    return values


def env_paths(path):
    values = read_env(path)

    def resolve(key, default=None):
        value = values.get(key) or default
        if not value:
            raise ValueError(f"Set {key} in {path}.")
        expanded = Path(value).expanduser()
        return (expanded if expanded.is_absolute() else path.parent / expanded).resolve()

    paths = {key: resolve(key) for key in ("MODEL_CACHE_DIR", "DATASET_DIR", "RESULTS_DIR")}
    # Accept either the Hub directory itself or the user's /workspace/model-cache root.
    root = paths["MODEL_CACHE_DIR"]
    if (root / "huggingface/hub").is_dir():
        paths["MODEL_CACHE_DIR"] = root / "huggingface/hub"
    paths["DATASET_MANIFEST"] = resolve(
        "DATASET_MANIFEST", str(paths["DATASET_DIR"] / "manifests/gsm8k_e0.json"))
    revision = values.get("DATASET_REVISION") or None
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("DATASET_REVISION must be a 40-character cached dataset commit SHA.")
    return paths, revision


def configure_offline(paths):
    # Set before importing Transformers, datasets or huggingface_hub; these cache
    # environment settings at import time. .env takes precedence over stale shell values.
    os.environ.update(
        HF_HUB_CACHE=str(paths["MODEL_CACHE_DIR"]),
        HF_DATASETS_CACHE=str(paths["DATASET_DIR"]),
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    # Transformers 4.x lets these legacy variables override HF_HUB_CACHE.
    # Clear them in this process so the .env path remains authoritative.
    for key in ("TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE"):
        os.environ.pop(key, None)


def build_config(args, paths, revision):
    from dlm_project.experiments.config import load_config
    from dlm_project.tasks.gsm8k.experiment_0 import resolve_settings

    config = load_config(args.config)
    config["experiment_root"] = str(paths["RESULTS_DIR"])
    config["data"].update(cache_dir=str(paths["DATASET_DIR"]), revision=revision)
    config["decoding"].update(seed=args.seed, completion_method="llada", completion_temperature=0.0)
    for key in ("init_tokens", "num_candidates", "top_k_tokens", "top_k_actions"):
        if getattr(args, key) is not None:
            config["decoding"]["mcts"][key] = getattr(args, key)
    return resolve_settings(
        config, stage=args.stage, num_samples=args.num_samples, run_order="condition",
        methods=METHODS[args.method], search_mode="budget", search_budgets=[args.search_budget],
        # Use the pinned repo ID with HF_HUB_CACHE and local_files_only. This both
        # reads existing snapshots and preserves the checkpoint identity E0 requires.
        model_path=get_profile()["repo_id"], local_files_only=True,
    )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--method", choices=METHODS, default="both")
    result.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    result.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/gsm8k/experiment_0.yaml")
    result.add_argument("--stage", choices=("dev", "evaluation", "test"), default="test")
    result.add_argument("--num-samples", type=int, default=5)
    result.add_argument("--search-budget", type=int, default=256, help="Maximum search forwards per question.")
    result.add_argument("--init-tokens", "--lc", dest="init_tokens", type=int, help="Lc (YAML default: 5).")
    result.add_argument("--num-candidates", type=int, help="C / shortlist size (YAML default: 3).")
    result.add_argument("--top-k-tokens", type=int, help="K1 (YAML default: 3).")
    result.add_argument("--top-k-actions", type=int, help="K2 (YAML default: 5).")
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--dry-run", action="store_true", help="Print paths and settings without loading data or weights.")
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        paths, revision = env_paths(args.env_file.expanduser().resolve())
        configure_offline(paths)
        config = build_config(args, paths, revision)
    except (OSError, ValueError) as error:
        cli.error(str(error))
    profile = get_profile()
    snapshot = (paths["MODEL_CACHE_DIR"] / ("models--" + profile["repo_id"].replace("/", "--"))
                / "snapshots" / profile["revision"])
    print(json.dumps({
        "method": args.method, "paths": {key: str(value) for key, value in paths.items()},
        "model_snapshot": str(snapshot), "offline": True,
        "stage": args.stage, "num_samples": config["data"]["num_samples"],
        "run_order": config["evaluation"]["run_order"], "conditions": config["conditions"],
        "generation": config["generation"], "decoding": config["decoding"],
    }, indent=2), flush=True)
    if args.dry_run:
        return
    if not (snapshot / "config.json").is_file():
        cli.error(f"Local LLaDA snapshot not found: {snapshot}. Set MODEL_CACHE_DIR to its Hugging Face Hub directory.")
    if not paths["DATASET_DIR"].is_dir():
        cli.error(f"Dataset directory does not exist: {paths['DATASET_DIR']}")
    if not paths["DATASET_MANIFEST"].is_file() and revision is None:
        cli.error("No E0 manifest found. Set DATASET_MANIFEST to an existing gsm8k_e0.json, or set "
                  "DATASET_REVISION to build it from already-cached GSM8K train/test splits offline.")

    from dlm_project.tasks.gsm8k.experiment_0 import run_experiment

    def interrupted(signum, frame):
        raise KeyboardInterrupt("GSM8K experiment terminated")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        directory, _ = run_experiment(
            config, manifest_path=paths["DATASET_MANIFEST"],
            output_dir=paths["RESULTS_DIR"] / "LLaDA/GSM8K/llada_vs_mcts" / args.method,
        )
        print(f"Results: {directory}", flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
