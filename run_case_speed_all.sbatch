#!/usr/bin/env bash
#SBATCH --job-name=case_speed_all
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=case_speed_all_%j.out
#SBATCH --error=case_speed_all_%j.err

set -euo pipefail

PROJECT_DIR="$HOME/projects/Hybrid-Quantum-Classical-Multimodal-Continuous-Emotion-Recognition-and-Trajectory-Forecasting"
RESULTS_DIR="$PROJECT_DIR/benchmark_results/case_speed"

cd "$PROJECT_DIR"

if [[ ! -f ".venv/bin/activate" ]]; then
    echo "ERROR: virtual environment not found at:"
    echo "$PROJECT_DIR/.venv/bin/activate"
    exit 1
fi

source .venv/bin/activate

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "CASE ALL-MODEL TRAINING SPEED BENCHMARK"
echo "============================================================"
echo "Started UTC        : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Host               : $(hostname)"
echo "Working directory  : $(pwd)"
echo "Slurm job ID       : ${SLURM_JOB_ID:-not-set}"
echo "Slurm job name     : ${SLURM_JOB_NAME:-not-set}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Results directory  : $RESULTS_DIR"
echo "============================================================"

nvidia-smi

python - <<'PY'
import sys
import torch

print("=" * 60)
print("PYTORCH CUDA VALIDATION")
print("=" * 60)
print("Python executable :", sys.executable)
print("Python version    :", sys.version.split()[0])
print("PyTorch version   :", torch.__version__)
print("CUDA available    :", torch.cuda.is_available())
print("CUDA runtime      :", torch.version.cuda)
print("cuDNN version     :", torch.backends.cudnn.version())
print("GPU count         :", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError(
        "The Slurm job did not receive a usable CUDA GPU."
    )

print("GPU               :", torch.cuda.get_device_name(0))
print("=" * 60)
PY

python -m py_compile tests/case/test_training_speed.py

python -u tests/case/test_training_speed.py \
    --models b1 b2 b3 qsal \
    --batch-size 2 \
    --warmup 3 \
    --repeats 20 \
    --num-workers 0 \
    --no-amp \
    --output-dir "$RESULTS_DIR" \
    --run-name "case_all_models_job_${SLURM_JOB_ID:-local}"

echo
echo "============================================================"
echo "BENCHMARK COMPLETED"
echo "============================================================"
echo "Finished UTC: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo
echo "Generated report files:"
find "$RESULTS_DIR" -maxdepth 1 -type f \
    -name "case_all_models_job_${SLURM_JOB_ID:-local}_*" \
    -printf '%f\n' \
    | sort
