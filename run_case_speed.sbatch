#!/usr/bin/env bash
#SBATCH --job-name=case_speed_b2
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=case_speed_b2_%j.out
#SBATCH --error=case_speed_b2_%j.err

set -euo pipefail

PROJECT_DIR="$HOME/projects/Hybrid-Quantum-Classical-Multimodal-Continuous-Emotion-Recognition-and-Trajectory-Forecasting"

cd "$PROJECT_DIR"
source .venv/bin/activate

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"

nvidia-smi

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("Slurm job did not receive a usable CUDA GPU")

print("GPU:", torch.cuda.get_device_name(0))
PY

python -u tests/case/test_training_speed.py \
    --models b2 \
    --batch-size 2 \
    --warmup 3 \
    --repeats 20 \
    --num-workers 0 \
    --no-amp
