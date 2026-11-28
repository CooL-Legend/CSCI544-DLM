#!/bin/bash
#SBATCH --job-name=e2-cmer-500
#SBATCH --account=xiangren_1987
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=results/slurm/e2-cmer-%j.out
#SBATCH --error=results/slurm/e2-cmer-%j.err

# Adapted from notebooks/gsm8k/E2_Experiment_CMER_Test.ipynb on adi-e2:
# https://github.com/CooL-Legend/CSCI544-DLM/blob/08efc305b35311c3643eb77f1a8b6b0ac8e86c7c/notebooks/gsm8k/E2_Experiment_CMER_Test.ipynb
# From the repository root:
#   mkdir -p results/slurm
#   sbatch scripts/run_carc_e2_cmer_test.sbatch 500 8
# Positional arguments: number of questions, cooperative batch size.
# Model files, dataset manifests and the Python environment must already exist.
set -euo pipefail

if (( $# > 2 )); then
    echo "Usage: sbatch scripts/run_carc_e2_cmer_test.sbatch [num_samples=500] [batch_size=8]" >&2
    exit 2
fi
NUM_SAMPLES="${1:-${NUM_SAMPLES:-500}}"
BATCH_SIZE="${2:-${BATCH_SIZE:-8}}"
if [[ ! "$NUM_SAMPLES" =~ ^[1-9][0-9]{0,2}$ ]] || (( NUM_SAMPLES > 500 )); then
    echo "num_samples must be an integer from 1 to 500." >&2
    exit 2
fi
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]{0,2}$ ]] || (( BATCH_SIZE > 500 )); then
    echo "batch_size must be an integer from 1 to 500." >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit with sbatch, or run this script inside an existing Slurm GPU allocation." >&2
    exit 2
fi

# Slurm runs a copy from its spool directory; BASH_SOURCE alone is not the repo.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO_DIR"
REPO_DIR="$PWD"
if [[ ! -f "$REPO_DIR/scripts/run_e2.py" ]]; then
    echo "Submit from the repository root, or set REPO_DIR to your adi-e2-carc checkout." >&2
    exit 2
fi

ENV_DIR="${ENV_DIR:-${DLM_ENV:-/project2/xiangren_1987/group_37/envs/dlm}}"
export MODEL_DIR="${MODEL_DIR:-/project2/xiangren_1987/group_37/model-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/project2/xiangren_1987/group_37/datasets}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$RESULTS_DIR/e2_cmer_test_500}"
PERSIST_ROOT="${PERSIST_ROOT:-$REPO_DIR/.cache/carc}"
PYTHON="$ENV_DIR/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Prepared Python environment not found: $PYTHON. Set ENV_DIR to your existing environment." >&2
    exit 2
fi

export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PERSIST_ROOT/huggingface"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$MODEL_DIR/huggingface/hub}"
# Transformers 4.38 writes version.txt even for a local model. Keep it writable
# and separate from the prepared weights; no weights are downloaded here.
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PERSIST_ROOT/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
mkdir -p "$ARTIFACT_ROOT" "$HF_MODULES_CACHE" "$TRANSFORMERS_CACHE" "$MPLCONFIGDIR"

# Read the prepared flat model folder or pinned HF snapshot. Never fall back
# to an online Hub ID, and do not reconstruct or validate dataset caches.
MODEL_PATH="$("$PYTHON" - <<'PY'
import os
from scripts.carc_paths import resolve_local_llada

print(resolve_local_llada(
    os.environ["MODEL_DIR"],
    model_path=os.environ.get("MODEL_PATH") or None,
    hub_cache=os.environ["HF_HUB_CACHE"],
))
PY
)"

echo "Job: $SLURM_JOB_ID | Host: $(hostname)"
echo "Repository: $REPO_DIR"
echo "Python: $PYTHON"
echo "Model: $MODEL_PATH"
echo "Dataset: same frozen 500-question test pool as E1/SWD (repository manifests; no dataset download)"
echo "Dataset cache: $HF_DATASETS_CACHE"
echo "Results: $ARTIFACT_ROOT"
echo "Questions: $NUM_SAMPLES | Batch size: $BATCH_SIZE"
echo "Methods: MEDAL + AB-MEDAL | Budgets: 256, 512 | Lc: 10 | Min candidates: 3"
echo "Reward: two-step CMER (common-mask entropy reduction) | Simulation cap: 4B"

# Lambda is retained for notebook consistency; the CMER formula does not use it.
exec srun "$PYTHON" -u "$REPO_DIR/scripts/run_e2.py" \
    --stage test \
    --num-samples "$NUM_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --methods medal ab_medal \
    --budgets 256 512 \
    --max-depth 10 \
    --num-candidates 3 \
    --reward-mode cmer \
    --reward-lambda 1.0 \
    --sim-cap-multiplier 4 \
    --device cuda \
    --model-path "$MODEL_PATH" \
    --local-files-only \
    --profile-resources \
    --run-e3 \
    --output-dir "$ARTIFACT_ROOT"
