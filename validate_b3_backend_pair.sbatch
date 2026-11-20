#!/bin/bash
#SBATCH --job-name=b3_backend_pair
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/b3_backend_pair_%j.out
#SBATCH --error=slurm/b3_backend_pair_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrid-qemotion

export OMP_NUM_THREADS=4

python -u -m tools.diagnostics.case_backend_pair_validation \
  --model b3 \
  --batch-size 2
