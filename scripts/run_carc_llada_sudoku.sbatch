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

# Submit this job while the prepared dlm environment is active. The model and
# dataset are loaded by the experiment runner without separate cache preflight.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/project2/xiangren_1987/group_37/model-cache}"
DATASET_DIR="${HF_DATASETS_CACHE:-/project2/xiangren_1987/group_37/datasets}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results}"
NUM_SAMPLES="${1:-${NUM_SAMPLES:-1}}"

mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR"

export HF_DATASETS_CACHE="$DATASET_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
unset HF_DATASETS_OFFLINE HF_HUB_OFFLINE HF_HUB_CACHE

echo "Model: $MODEL_DIR"
echo "Dataset cache: $DATASET_DIR"
echo "Results: $RESULTS_DIR"
echo "Puzzles: $NUM_SAMPLES"

exec python -u scripts/run_sudoku.py \
    --config configs/sudoku/llada_mcts.yaml \
    --num-samples "$NUM_SAMPLES" \
    --model-path "$MODEL_DIR" \
    --output-dir "$RESULTS_DIR" \
    --local-files-only
