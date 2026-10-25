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
        "DATASET_MANIFEST", str(PROJECT_ROOT / "configs/gsm8k/experiment_samples.json"))
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


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env-file', type=Path, default=PROJECT_ROOT / '.env')
    p.add_argument('--method', choices=METHODS, default='both')
    p.add_argument('--stage', choices=('dev', 'evaluation', 'test'), default='test')
    p.add_argument('--num-samples', type=int, default=5)
    p.add_argument('--search-budget', type=int, default=256)
    p.add_argument('--init-tokens', '--lc', dest='init_tokens', type=int, default=10)
    p.add_argument('--num-candidates', type=int, default=3)
    p.add_argument('--top-k-tokens', type=int, default=3)
    p.add_argument('--top-k-actions', type=int, default=5)
    p.add_argument('--reward-mode', choices=('ig', 'adalook_one_step', 'adalook_adaptive'), default='ig')
    p.add_argument('--adalook-alpha', type=float, default=0.1)
    p.add_argument('--adalook-confidence-threshold', type=float, default=0.8)
    p.add_argument('--adalook-max-depth', type=int, default=2)
    p.add_argument('--adalook-variance-threshold', type=float, default=0.01)
    p.add_argument('--adalook-trigger-threshold', type=float, default=0.2)
    p.add_argument('--adalook-min-masks', type=int, default=2)
    p.add_argument('--exploration-c', type=float, default=0.01)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', type=Path, help='Resume this run directory with identical arguments/configuration.')
    p.add_argument('--dry-run', action='store_true')
    return p


def build_config(args, paths, revision):
    from dataclasses import asdict
    from dlm_project.decoding.mcts_decoding import MCTSConfig
    from dlm_project.tasks.gsm8k.experiment_protocol import load_shared_protocol, STAGE_COUNTS
    common = load_shared_protocol()
    if not 1 <= args.num_samples <= STAGE_COUNTS[args.stage]:
        raise ValueError(f'num-samples must be between 1 and {STAGE_COUNTS[args.stage]}.')
    if not 0 <= args.seed < 2**63:
        raise ValueError('seed must be in [0, 2**63).')
    if revision and revision != common['data']['revision']:
        raise ValueError('Dataset revision must match the frozen shared manifest.')
    fields = ('init_tokens', 'num_candidates', 'top_k_tokens', 'top_k_actions', 'reward_mode', 'exploration_c',
              'adalook_alpha', 'adalook_confidence_threshold', 'adalook_max_depth',
              'adalook_variance_threshold', 'adalook_trigger_threshold', 'adalook_min_masks')
    mcts = MCTSConfig(**{k: getattr(args, k) for k in fields}, stop_mode='budget', max_model_calls=args.search_budget)
    mcts.validate_length(common['generation']['gen_length'])
    return dict(version='gsm8k_local_adalook_v1', common=common,
                paths={k: str(v) for k, v in paths.items()}, method=args.method, stage=args.stage,
                num_samples=args.num_samples, seed=args.seed, device=args.device,
                run_order='condition', search=asdict(mcts), offline=True)


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        paths, revision = env_paths(args.env_file.expanduser().resolve())
        configure_offline(paths)
        config = build_config(args, paths, revision)
        print(json.dumps(config, indent=2), flush=True)
        if args.dry_run:
            return config
        from dlm_project.tasks.gsm8k.local_experiments import run_experiments
        def interrupted(signum, frame):
            raise KeyboardInterrupt('GSM8K experiment terminated')
        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            directory = run_experiments(config, resume=args.resume)
        finally:
            signal.signal(signal.SIGTERM, previous)
        print(f'Results: {directory}', flush=True)
        return directory
    except (OSError, ValueError) as error:
        cli.error(str(error))


if __name__ == '__main__':
    main()


