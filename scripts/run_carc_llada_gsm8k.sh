#!/usr/bin/env bash
#SBATCH --job-name=llada-gsm8k
#SBATCH --account=xiangren_1987
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Run the LLaDA GSM8K experiments from the former Colab workflow on CARC.

Usage:
  sbatch scripts/run_carc_llada_gsm8k.sh [NUM_SAMPLES] [METHOD] [PROFILE]
  bash   scripts/run_carc_llada_gsm8k.sh [NUM_SAMPLES] [METHOD] [PROFILE]

Arguments:
  NUM_SAMPLES  Questions to run (default: 2)
  METHOD       mcts_confidence (default), confidence, random_mdm,
               top_k_margin, best_of_5, all, or metrics
  PROFILE      smoke (default), initial, or full

Environment overrides:
  MODEL_DIR       Prepared LLaDA snapshot/cache root
  MODEL_PATH      Exact prepared snapshot directory (skips cache-root discovery)
  DATASET_DIR     Prepared Hugging Face datasets cache
  RESULTS_DIR     Experiment output root (default: <repository>/results)
  MANIFEST_PATH   Frozen GSM8K test manifest
  PYTHON_BIN      Python from the prepared environment (default: python)
  ALLOW_NETWORK   0 by default. Set to 1 only to allow creation of a missing
                  frozen dataset manifest through the Hugging Face Hub.
  MCTS_BUDGETS    Space-separated budgets for METHOD=metrics

Examples:
  sbatch scripts/run_carc_llada_gsm8k.sh 2 mcts_confidence smoke
  sbatch scripts/run_carc_llada_gsm8k.sh 20 all smoke
  MCTS_BUDGETS="4 8 16" sbatch scripts/run_carc_llada_gsm8k.sh 2 metrics smoke
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/project2/xiangren_1987/group_37/model-cache}"
DATASET_DIR="${DATASET_DIR:-/project2/xiangren_1987/group_37/datasets}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results}"
MANIFEST_PATH="${MANIFEST_PATH:-$RESULTS_DIR/manifests/gsm8k_test.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_SAMPLES="${1:-${NUM_SAMPLES:-2}}"
METHOD="${2:-${METHOD:-mcts_confidence}}"
PROFILE="${3:-${PROFILE:-smoke}}"
ALLOW_NETWORK="${ALLOW_NETWORK:-0}"
MCTS_BUDGETS="${MCTS_BUDGETS:-4 8 16 32 64}"
RUNTIME_CONFIG=""

cleanup() {
    if [[ -n "$RUNTIME_CONFIG" && -f "$RUNTIME_CONFIG" ]]; then
        rm -f -- "$RUNTIME_CONFIG"
    fi
}
trap cleanup EXIT

if ! [[ "$NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]] || (( NUM_SAMPLES > 1319 )); then
    echo "NUM_SAMPLES must be an integer between 1 and 1319." >&2
    exit 2
fi
case "$PROFILE" in
    smoke|initial|full) ;;
    *)
        echo "PROFILE must be smoke, initial, or full." >&2
        exit 2
        ;;
esac
case "$METHOD" in
    mcts_confidence|confidence|random_mdm|top_k_margin|best_of_5|all|metrics) ;;
    *)
        echo "Unknown METHOD: $METHOD" >&2
        usage >&2
        exit 2
        ;;
esac
case "$ALLOW_NETWORK" in
    0|1) ;;
    *)
        echo "ALLOW_NETWORK must be 0 or 1." >&2
        exit 2
        ;;
esac

resolve_model_path() {
    if [[ -n "${MODEL_PATH:-}" ]]; then
        printf '%s\n' "$MODEL_PATH"
        return
    fi

    local revision="08b83a6feb34df1a6011b80c3c00c7563e963b07"
    local snapshot="models--GSAI-ML--LLaDA-8B-Instruct/snapshots/$revision"
    local candidate
    for candidate in \
        "$MODEL_DIR" \
        "$MODEL_DIR/$snapshot" \
        "$MODEL_DIR/hub/$snapshot" \
        "$MODEL_DIR/huggingface/hub/$snapshot"; do
        if [[ -f "$candidate/config.json" && -f "$candidate/.model_source.json" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done

    echo "Prepared LLaDA snapshot not found under: $MODEL_DIR" >&2
    echo "The snapshot must contain config.json and .model_source.json." >&2
    echo "Set MODEL_PATH to its exact directory if it lives elsewhere." >&2
    return 1
}

MODEL_PATH="$(resolve_model_path)"

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "Prepared dataset cache not found: $DATASET_DIR" >&2
    exit 2
fi
if ! "$PYTHON_BIN" -c 'import torch, datasets, transformers, yaml' >/dev/null 2>&1; then
    echo "The active Python environment is missing project requirements." >&2
    echo "Activate the prepared dlm environment before submitting this script." >&2
    exit 2
fi
if ! "$PYTHON_BIN" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    echo "CUDA is unavailable. Run inside a GPU allocation or submit with sbatch." >&2
    exit 2
fi

mkdir -p "$RESULTS_DIR" "$(dirname "$MANIFEST_PATH")" "$RESULTS_DIR/run-receipts"
cd "$REPO_DIR"

export HF_DATASETS_CACHE="$DATASET_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$ALLOW_NETWORK" == "0" ]]; then
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    if [[ ! -f "$MANIFEST_PATH" ]]; then
        echo "Frozen GSM8K manifest not found: $MANIFEST_PATH" >&2
        echo "Prepare/copy the manifest first, or explicitly set ALLOW_NETWORK=1 for its first creation." >&2
        exit 2
    fi
else
    unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
fi

echo "GPU: $($PYTHON_BIN -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "Method: $METHOD"
echo "Profile: $PROFILE"
echo "Questions: $NUM_SAMPLES"
echo "Model: $MODEL_PATH"
echo "Dataset cache: $DATASET_DIR"
echo "Manifest: $MANIFEST_PATH"
echo "Results: $RESULTS_DIR"

receipt_path() {
    local label="$1"
    printf '%s/run-receipts/%s-%s-%s.json\n' \
        "$RESULTS_DIR" "$label" "${SLURM_JOB_ID:-manual}" "$(date +%s)"
}

run_method() {
    local selected="$1"
    local entrypoint config
    case "$selected" in
        mcts_confidence)
            entrypoint="scripts/eval_gsm8k_mcts_confidence.py"
            config="configs/gsm8k/mcts_confidence.yaml"
            ;;
        confidence|random_mdm|top_k_margin|best_of_5)
            entrypoint="scripts/eval_gsm8k_${selected}.py"
            config="configs/gsm8k/decoding.yaml"
            ;;
        *)
            echo "Internal error: unsupported single method $selected" >&2
            return 2
            ;;
    esac

    "$PYTHON_BIN" -u "$entrypoint" \
        --config "$config" \
        --profile "$PROFILE" \
        --num-samples "$NUM_SAMPLES" \
        --model-path "$MODEL_PATH" \
        --local-files-only \
        --manifest "$MANIFEST_PATH" \
        --output-dir "$RESULTS_DIR/LLaDA/GSM8K" \
        --run-manifest "$(receipt_path "$selected")"
}

run_metrics() {
    RUNTIME_CONFIG="$(mktemp "${TMPDIR:-/tmp}/llada-gsm8k-carc.XXXXXX.yaml")"

    "$PYTHON_BIN" - "$REPO_DIR/configs/gsm8k/decoding.yaml" "$RUNTIME_CONFIG" \
        "$MODEL_PATH" "$DATASET_DIR" "$RESULTS_DIR" <<'PY'
from pathlib import Path
import sys
import yaml

source, destination, model_path, dataset_dir, results_dir = sys.argv[1:]
config = yaml.safe_load(Path(source).read_text())
config["model"].update(model_path=model_path, local_files_only=True)
config["data"]["cache_dir"] = dataset_dir
config["experiment_root"] = results_dir
config["decoding"]["mcts"] = {
    "init_tokens": 20,
    "num_candidates": 3,
    "top_k_tokens": 3,
    "top_k_actions": 5,
    "max_model_calls": None,
    "margin_gamma": 10.0,
    "exploration_c": 0.01,
    "epsilon": 1e-8,
    "stop_mode": "candidates",
}
Path(destination).write_text(yaml.safe_dump(config, sort_keys=False))
PY

    local -a budgets
    read -r -a budgets <<< "$MCTS_BUDGETS"
    if (( ${#budgets[@]} == 0 )); then
        echo "MCTS_BUDGETS must contain at least one positive integer." >&2
        return 2
    fi
    local budget
    for budget in "${budgets[@]}"; do
        if ! [[ "$budget" =~ ^[1-9][0-9]*$ ]]; then
            echo "Invalid MCTS budget: $budget" >&2
            return 2
        fi
    done

    "$PYTHON_BIN" -u scripts/run_gsm8k_metrics.py \
        --config "$RUNTIME_CONFIG" \
        --profile "$PROFILE" \
        --num-samples "$NUM_SAMPLES" \
        --budgets "${budgets[@]}" \
        --manifest "$MANIFEST_PATH" \
        --output-dir "$RESULTS_DIR/LLaDA/GSM8K/metrics_sweeps" \
        --run-manifest "$(receipt_path metrics)"
}

case "$METHOD" in
    all)
        for selected in confidence random_mdm top_k_margin best_of_5 mcts_confidence; do
            echo
            echo "Running $selected"
            run_method "$selected"
        done
        ;;
    metrics)
        run_metrics
        ;;
    *)
        run_method "$METHOD"
        ;;
esac
