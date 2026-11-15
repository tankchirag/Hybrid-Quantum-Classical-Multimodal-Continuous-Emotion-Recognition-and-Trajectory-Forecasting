#!/bin/bash
#SBATCH --job-name=case_qsal_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=qsal_smoke_%j.out
#SBATCH --error=qsal_smoke_%j.err

set -euo pipefail

cd "$HOME/projects/Hybrid-Quantum-Classical-Multimodal-Continuous-Emotion-Recognition-and-Trajectory-Forecasting"
source .venv/bin/activate

echo "host=$(hostname)"
echo "date=$(date)"
echo "python=$(which python)"
nvidia-smi

python -u -m training.train_case_qsal
