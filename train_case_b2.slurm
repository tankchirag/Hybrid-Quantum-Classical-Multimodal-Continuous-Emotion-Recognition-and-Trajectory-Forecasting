#!/bin/bash
#SBATCH --job-name=case_b2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=case_b2_%j.out
#SBATCH --error=case_b2_%j.err

cd $SLURM_SUBMIT_DIR

source .venv/bin/activate

python -m training.train_case_b2
