#!/usr/bin/env bash
#SBATCH --job-name=case_batch8_short
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/case_batch8_short_%j.out
#SBATCH --error=slurm/case_batch8_short_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrid-qemotion

export OMP_NUM_THREADS=4

RESULTS_DIR="benchmark_results/case_batch8_short/job_${SLURM_JOB_ID}"

mkdir -p "$RESULTS_DIR"

echo "=== GPU ==="
nvidia-smi -L

echo "=== OPTIMIZED B3 + QSAL, BATCH SIZE 8 ==="

python -u tests/benchmark_all_models.py \
  --factory tools.diagnostics.optimized_benchmark_factory:build_benchmark \
  --only B3_4Q_DEFAULT QSAL_4Q_R4_F20 \
  --device cuda \
  --batch-size 8 \
  --workers 0 \
  --warmup 5 \
  --steps 20 \
  --timing-repeats 1 \
  --no-amp \
  --output-dir "$RESULTS_DIR"
