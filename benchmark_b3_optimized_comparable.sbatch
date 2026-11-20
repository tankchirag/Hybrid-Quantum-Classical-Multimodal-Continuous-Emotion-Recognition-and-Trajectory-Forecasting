#!/usr/bin/env bash
#SBATCH --job-name=b3_opt_comparable
#SBATCH --partition=gpu
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/b3_opt_comparable_%j.out
#SBATCH --error=slurm/b3_opt_comparable_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrid-qemotion

export OMP_NUM_THREADS=4

RESULTS_DIR="benchmark_results/b3_optimized_comparable/job_${SLURM_JOB_ID}"

mkdir -p "$RESULTS_DIR" slurm

echo "===== HARDWARE ====="
nvidia-smi
echo "===== SOURCE ====="
if command -v git >/dev/null 2>&1; then
    git rev-parse HEAD
    git status --short
else
    echo "[INFO] Git unavailable on compute node; skipping metadata."
fi

echo "===== OPTIMIZED B3 — ORIGINAL BENCHMARK CRITERIA ====="

python -u tests/benchmark_all_models.py \
  --factory tools.diagnostics.optimized_benchmark_factory:build_benchmark \
  --only B3_4Q_DEFAULT \
  --device cuda \
  --batch-size 2 \
  --workers 0 \
  --warmup 10 \
  --steps 50 \
  --timing-repeats 3 \
  --no-amp \
  --output-dir "$RESULTS_DIR"

echo "===== RESULT FILES ====="
find "$RESULTS_DIR" -type f
