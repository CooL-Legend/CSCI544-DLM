#!/usr/bin/env bash

# Run the original LLaDA Sudoku sampler only. This script never downloads or
# prepares the model, dataset, or Python environment.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNPOD_MODEL="/workspace/model-cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07"
CARC_MODEL="/project2/xiangren_1987/group_37/model-cache"
if [[ -z "${MODEL_DIR:-}" ]]; then
    if [[ -d "$RUNPOD_MODEL" ]]; then
        MODEL_DIR="$RUNPOD_MODEL"
    else
        MODEL_DIR="$CARC_MODEL"
    fi
fi

if [[ -z "${HF_DATASETS_CACHE:-}" ]]; then
    if [[ -d /workspace/datasets ]]; then
        HF_DATASETS_CACHE="/workspace/datasets"
    else
        HF_DATASETS_CACHE="/project2/xiangren_1987/group_37/datasets"
    fi
fi

RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results}"
NUM_SAMPLES="${1:-${NUM_SAMPLES:-1}}"
CONFIG_FILE="$REPO_DIR/configs/sudoku/llada.yaml"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    echo "Model snapshot not found at: $MODEL_DIR" >&2
    echo "Set MODEL_DIR to the local LLaDA snapshot directory." >&2
    exit 2
fi
if [[ ! -d "$HF_DATASETS_CACHE" ]]; then
    echo "Dataset cache not found at: $HF_DATASETS_CACHE" >&2
    echo "Set HF_DATASETS_CACHE to the prepared Sudoku dataset cache." >&2
    exit 2
fi

mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR"

export HF_DATASETS_CACHE
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Decoder: original LLaDA baseline (no MCTS)"
echo "Config: $CONFIG_FILE"
echo "Model: $MODEL_DIR"
echo "Dataset cache: $HF_DATASETS_CACHE"
echo "Results: $RESULTS_DIR"
echo "Puzzles: $NUM_SAMPLES"

exec python -u scripts/run_sudoku.py \
    --config "$CONFIG_FILE" \
    --num-samples "$NUM_SAMPLES" \
    --model-path "$MODEL_DIR" \
    --output-dir "$RESULTS_DIR" \
    --local-files-only
