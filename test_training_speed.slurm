#!/bin/bash
#SBATCH --job-name=case_speed
#SBATCH --output=case_speed_%j.out
#SBATCH --error=case_speed_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

cd "$HOME/projects/Hybrid-Quantum-Classical-Multimodal-Continuous-Emotion-Recognition-and-Trajectory-Forecasting"
source .venv/bin/activate

echo "Host: $(hostname)"
echo "Python: $(which python)"
python --version
nvidia-smi

python tests/case/test_training_speed.py \
  --models b1 b2 \
  --batch-size 8 \
  --warmup 2 \
  --repeats 20 \
  --num-workers 4 \
  --pin-memory

python tests/case/test_training_speed.py \
  --models b3 qsal \
  --batch-size 2 \
  --warmup 1 \
  --repeats 3 \
  --num-workers 0