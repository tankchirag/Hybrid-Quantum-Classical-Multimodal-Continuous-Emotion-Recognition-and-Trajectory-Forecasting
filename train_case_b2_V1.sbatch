#!/usr/bin/env bash

#SBATCH --job-name=case_b2_V1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=case_b2_full_%j.out
#SBATCH --error=case_b2_full_%j.err

set -euo pipefail

PROJECT_DIR="$HOME/projects/Hybrid-Quantum-Classical-Multimodal-Continuous-Emotion-Recognition-and-Trajectory-Forecasting"
RUN_DIR="$PROJECT_DIR/runs/case_b2"

cd "$PROJECT_DIR"

echo "============================================================"
echo "CASE B2 FULL TRAINING"
echo "============================================================"
echo "Started UTC         : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Host                : $(hostname)"
echo "Working directory   : $(pwd)"
echo "Slurm job ID        : ${SLURM_JOB_ID:-not-set}"
echo "Slurm job name      : ${SLURM_JOB_NAME:-not-set}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Run directory       : $RUN_DIR"
echo "============================================================"

if [[ ! -f ".venv/bin/activate" ]]; then
    echo "ERROR: virtual environment not found:"
    echo "$PROJECT_DIR/.venv/bin/activate"
    exit 1
fi

source .venv/bin/activate

# Make root-level repository packages importable from scripts in subdirectories.
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RUN_DIR"
mkdir -p "$RUN_DIR/job_metadata"

echo
echo "Python environment:"
echo "Python executable: $(command -v python)"
python --version

echo
echo "GPU allocation:"
nvidia-smi

echo
echo "PyTorch CUDA validation:"
python - <<'PY'
import os
import platform
import sys

import torch

print("Python executable :", sys.executable)
print("Python version    :", sys.version.split()[0])
print("Platform          :", platform.platform())
print("PyTorch version   :", torch.__version__)
print("CUDA available    :", torch.cuda.is_available())
print("CUDA runtime      :", torch.version.cuda)
print("cuDNN version     :", torch.backends.cudnn.version())
print("GPU count         :", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

if not torch.cuda.is_available():
    raise RuntimeError(
        "The Slurm allocation does not provide a usable CUDA GPU."
    )

print("GPU name          :", torch.cuda.get_device_name(0))
print("GPU capability    :", torch.cuda.get_device_capability(0))
print(
    "GPU memory GiB    :",
    round(
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3),
        2,
    ),
)
PY

echo
echo "Saving job and environment metadata:"

METADATA_FILE="$RUN_DIR/job_metadata/job_${SLURM_JOB_ID:-manual}.txt"
GIT_STATUS_FILE="$RUN_DIR/job_metadata/git_status_${SLURM_JOB_ID:-manual}.txt"
REQUIREMENTS_FILE="$RUN_DIR/job_metadata/requirements_${SLURM_JOB_ID:-manual}.txt"

{
    echo "started_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "host=$(hostname)"
    echo "slurm_job_id=${SLURM_JOB_ID:-not-set}"
    echo "slurm_job_name=${SLURM_JOB_NAME:-not-set}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-not-set}"

    if command -v git >/dev/null 2>&1 &&
       git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        GIT_EXECUTABLE="$(command -v git || true)"
        GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
        GIT_BRANCH="$(git branch --show-current 2>/dev/null || true)"
        echo "git_executable=${GIT_EXECUTABLE:-not-available}"
        echo "git_commit=${GIT_COMMIT:-not-recorded}"
        echo "git_branch=${GIT_BRANCH:-not-recorded}"
    else
        echo "git_executable=not-available"
        echo "git_commit=not-recorded"
        echo "git_branch=not-recorded"
    fi
} > "$METADATA_FILE"

if command -v git >/dev/null 2>&1 &&
   git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if ! git status --short > "$GIT_STATUS_FILE" 2>&1; then
        echo "Git status could not be recorded." > "$GIT_STATUS_FILE"
    fi
else
    echo "Git unavailable or working directory is not a Git repository." \
        > "$GIT_STATUS_FILE"
fi

if ! python -m pip freeze > "$REQUIREMENTS_FILE" 2>&1; then
    echo "Python package versions could not be recorded." \
        > "$REQUIREMENTS_FILE"
fi

if ! cp configs/case_b2.yaml \
    "$RUN_DIR/job_metadata/case_b2_${SLURM_JOB_ID:-manual}.yaml"; then
    echo "WARNING: could not copy configs/case_b2.yaml to job metadata." >&2
fi

echo
echo "Compiling B2 training modules:"

python -m py_compile \
    training/train_case_b2.py \
    training/base_training_script.py \
    models/wrappers/baseline_b2.py \
    utils/config.py

echo
echo "Validating configuration:"

python - <<'PY'
from utils.config import load_config

config = load_config("configs/case_b2.yaml")

print("Experiment :", config["experiment"]["name"])
print("Batch size :", config["training"]["batch_size"])
print("Epochs     :", config["training"]["epochs"])
print("Patience   :", config["training"]["patience"])
print("Learning rate:", config["optimizer"]["lr"])
print("Weight decay:", config["optimizer"]["weight_decay"])
print("History length:", config["history"]["history_length"])
print(
    "Forecast horizon:",
    config["forecasting"]["forecast_horizon"],
)
PY

echo
echo "Checking repository import path:"

python - <<'PY'
import os
import sys

print("Working directory:", os.getcwd())
print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))
print("sys.path:")
for path in sys.path:
    print("  ", path)

from utils.config import load_config
from datasets.wrappers.case_training_dataset import CaseTrainingDataset
from models.wrappers.baseline_b2 import BaselineB2

print("Repository imports: OK")
PY

echo
echo "Running focused B2 smoke tests:"

python -u tests/case/test_b2_case.py
python -u tests/case/test_b2_training_batch.py
python -m pytest -q \
    tests/test_config_validation.py

echo
echo "Starting CASE B2 full training:"
echo "============================================================"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

set +e
srun python -u -m training.train_case_b2
TRAIN_EXIT_CODE=$?
set -e

{
    echo "finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "exit_code=$TRAIN_EXIT_CODE"
} >> "$RUN_DIR/job_metadata/job_${SLURM_JOB_ID:-manual}.txt"


echo
echo "============================================================"
echo "CASE B2 TRAINING FINISHED"
echo "============================================================"
echo "Finished UTC : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Exit code    : $TRAIN_EXIT_CODE"

echo
echo "Generated run artifacts:"
find "$RUN_DIR" \
    -maxdepth 3 \
    -type f \
    -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' \
    | sort

exit "$TRAIN_EXIT_CODE"
