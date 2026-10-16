#!/bin/bash
#SBATCH --job-name=llada-sudoku
#SBATCH --account=xiangren_1987
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Submit this job while the prepared dlm environment is active. This script does
# not install packages, activate an environment, or download model/dataset files.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/project2/xiangren_1987/group_37/model-cache}"
DATASET_DIR="${HF_DATASETS_CACHE:-/project2/xiangren_1987/group_37/datasets}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/LLaDA/Sudoku}"
NUM_SAMPLES="${1:-${NUM_SAMPLES:-1}}"

if [[ ! "$NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_SAMPLES must be a positive integer; received: $NUM_SAMPLES" >&2
    exit 2
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    echo "Prepared model not found at $MODEL_DIR" >&2
    exit 1
fi

if [[ ! -d "$DATASET_DIR" ]] || [[ -z "$(find "$DATASET_DIR" -mindepth 1 -print -quit)" ]]; then
    echo "Prepared dataset cache not found at $DATASET_DIR" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR"

# Force local-only execution. If either prepared cache is incomplete, fail
# instead of fetching files during the experiment job.
export HF_DATASETS_CACHE="$DATASET_DIR"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

python - "$MODEL_DIR" <<'PY'
import sys

import torch

from dlm_project.models.cache import validate_snapshot

model_dir = sys.argv[1]
validate_snapshot(model_dir, "llada")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; submit this script to a GPU node.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Model: {model_dir}")
PY

echo "Dataset cache: $DATASET_DIR"
echo "Results: $RESULTS_DIR"
echo "Puzzles: $NUM_SAMPLES"

exec python -u scripts/run_sudoku.py \
    --config configs/sudoku/llada_mcts.yaml \
    --num-samples "$NUM_SAMPLES" \
    --model-path "$MODEL_DIR" \
    --output-dir "$RESULTS_DIR" \
    --local-files-only
