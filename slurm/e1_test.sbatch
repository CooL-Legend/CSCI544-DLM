#!/bin/bash
#SBATCH --job-name=dlm-e1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/e1-%j.out
#SBATCH --error=logs/e1-%j.err

set -euo pipefail
: "${REPO_DIR:?Set REPO_DIR to the CARC repository checkout}"
: "${ENV_DIR:?Set ENV_DIR to the prepared Python environment}"
: "${MODEL_PATH:?Set MODEL_PATH to the complete pinned model snapshot}"
: "${ARTIFACT_ROOT:?Set ARTIFACT_ROOT to a durable project directory}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${PERSIST_ROOT:-$ARTIFACT_ROOT}/cache/huggingface"
export HF_HUB_OFFLINE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

case "${RUN_MODE:-production}" in
  benchmark)
    srun "$ENV_DIR/bin/python" scripts/benchmark_carc.py \
      --model-path "$MODEL_PATH" --local-files-only --output-dir "$ARTIFACT_ROOT/pilot"
    ;;
  smoke)
    srun "$ENV_DIR/bin/python" scripts/run_e1_test.py \
      --num-samples 4 --batch-size 4 --profile-resources \
      --model-path "$MODEL_PATH" --local-files-only --output-dir "$ARTIFACT_ROOT/smoke"
    ;;
  production)
    : "${BENCHMARK_REPORT:?Set BENCHMARK_REPORT to a successful benchmark_report.json}"
    srun "$ENV_DIR/bin/python" scripts/run_carc_production.py \
      --benchmark-report "$BENCHMARK_REPORT" --model-path "$MODEL_PATH" \
      --output-dir "$ARTIFACT_ROOT/production"
    ;;
  *) echo "RUN_MODE must be smoke, benchmark, or production" >&2; exit 2 ;;
esac
