#!/bin/bash
#SBATCH --job-name=case_backend_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/case_candidate_%j.out
#SBATCH --error=slurm/case_candidate_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrid-qemotion

export OMP_NUM_THREADS=4

MODEL="${1:?Specify b3 or qsal}"

case "$MODEL" in
    b3)
        python -u -m training.train_case_b3 \
          --config configs/experiments/case_b3_torch_backprop.yaml
        ;;
    qsal)
        python -u -m training.train_case_qsal \
          --config configs/experiments/case_qsal_torch_backprop.yaml
        ;;
    *)
        echo "Unsupported model: $MODEL" >&2
        exit 2
        ;;
esac
