#!/usr/bin/env bash
#SBATCH --job-name=case_4m_cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/case_4m_cv_%j.out
#SBATCH --error=slurm/case_4m_cv_%j.err
set -Eeuo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
: "${METHOD:?Supply METHOD=b1|b2|b3|qsal in --export}"
: "${FOLD:?Supply FOLD=1..5 in --export}"
if [[ "$METHOD" != "b1" && "$METHOD" != "b2" && "$METHOD" != "b3" && "$METHOD" != "qsal" ]]; then echo 'Invalid METHOD' >&2; exit 2; fi
if [[ ! "$FOLD" =~ ^[1-5]$ ]]; then echo 'Invalid FOLD' >&2; exit 2; fi
# Full training is deliberately not submitted until the four smoke tests are inspected.
srun python -u -m training.train_case_four_method_cv --method "$METHOD" --fold "$FOLD" --train --epochs 3 --batch-size 8 --lr 0.0001 --clip 0.5 --weight-decay 0.0001
