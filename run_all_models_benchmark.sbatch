#!/usr/bin/env bash
#SBATCH --job-name=all_models_bench
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=all_models_bench_%j.out
#SBATCH --error=all_models_bench_%j.err

set -euo pipefail

PROJECT_DIR="$HOME/projects/Hybrid-Quantum-Classical-Multimodal-Continuous-Emotion-Recognition-and-Trajectory-Forecasting"
RESULTS_DIR="$PROJECT_DIR/benchmark_results/all_models"

cd "$PROJECT_DIR"

if [[ ! -f ".venv/bin/activate" ]]; then
    echo "ERROR: virtual environment not found:"
    echo "$PROJECT_DIR/.venv/bin/activate"
    exit 1
fi

source .venv/bin/activate
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "CASE ALL-CONFIGURATION TRAINING-TIME BENCHMARK"
echo "============================================================"
echo "Started UTC         : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Host                : $(hostname)"
echo "Working directory   : $(pwd)"
echo "Python              : $(command -v python)"
echo "Slurm job ID        : ${SLURM_JOB_ID:-not-set}"
echo "Slurm job name      : ${SLURM_JOB_NAME:-not-set}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Results directory   : $RESULTS_DIR"
echo "============================================================"

echo
echo "GPU allocation:"
nvidia-smi

echo
echo "PyTorch CUDA validation:"
python - <<'PY'
import sys
import torch

print("Python executable :", sys.executable)
print("Python version    :", sys.version.split()[0])
print("PyTorch version   :", torch.__version__)
print("CUDA available    :", torch.cuda.is_available())
print("CUDA runtime      :", torch.version.cuda)
print("cuDNN version     :", torch.backends.cudnn.version())
print("GPU count         :", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("The Slurm job did not receive a usable CUDA GPU.")

print("GPU               :", torch.cuda.get_device_name(0))
print("GPU capability    :", torch.cuda.get_device_capability(0))
PY

echo
echo "Compiling benchmark code:"
python -m py_compile \
    tests/benchmark_all_models.py \
    training/benchmark_adapter.py \
    training/losses.py \
    utils/config.py

echo
echo "Running focused benchmark tests:"
python -m pytest -q \
    tests/test_benchmark_adapter.py \
    tests/test_config_validation.py \
    tests/test_benchmark_matrix.py

echo
echo "Validating benchmark factory:"
python - <<'PY'
from training.benchmark_adapter import build_benchmark
print("Factory import OK:", build_benchmark)
PY

echo
echo "Running full 10-configuration preflight:"
python -u tests/benchmark_all_models.py \
    --factory training.benchmark_adapter:build_benchmark \
    --device cuda \
    --batch-size 2 \
    --workers 0 \
    --no-amp \
    --preflight-only

echo
echo "Validating allocated CUDA device:"
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("The Slurm allocation does not provide usable CUDA.")
print("CUDA device:", torch.cuda.get_device_name(0))
PY

echo
echo "Starting direct 5-warm-up / 50-measured-step benchmark:"
python -u tests/benchmark_all_models.py \
    --factory training.benchmark_adapter:build_benchmark \
    --device cuda \
    --batch-size 2 \
    --workers 0 \
    --no-amp \
    --warmup 5 \
    --steps 50 \
    --output-dir "$RESULTS_DIR"

echo
echo "============================================================"
echo "BENCHMARK COMPLETED"
echo "============================================================"
echo "Finished UTC: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

echo
echo "Generated files:"
find "$RESULTS_DIR" \
    -maxdepth 1 \
    -type f \
    -printf '%TY-%Tm-%Td %TH:%TM:%TS %f\n' \
    | sort
