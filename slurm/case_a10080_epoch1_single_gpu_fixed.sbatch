#!/usr/bin/env bash
#SBATCH --job-name=case_a10080_epoch1_final
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclude=zits-slurmlogin1,zits-slurmworker4
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/case_a10080_epoch1_final_%j.out
#SBATCH --error=slurm/case_a10080_epoch1_final_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON="$HOME/miniforge3/envs/hybrid-qemotion/bin/python"
RESULTS="benchmark_results/case_a10080_epoch1_final/job_${SLURM_JOB_ID}"

mkdir -p "$RESULTS"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "=== EXPERIMENT INFORMATION ==="
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Git commit: 0425b741b5b78abbf513f6f586dd986ae7bc7aab"
echo "Start: $(date --iso-8601=seconds)"

echo "=== GPU INFORMATION ==="
nvidia-smi
nvidia-smi -L > "$RESULTS/gpu_information.txt"

"$PYTHON" - "$RESULTS" <<'PY'
import json
import sys
import torch
import yaml
from pathlib import Path

results = Path(sys.argv[1])
gpu = torch.cuda.get_device_name(0)

print("Selected GPU:", gpu, flush=True)

if "A100" not in gpu or "80GB" not in gpu.upper().replace(" ", ""):
    raise RuntimeError(f"Expected full A100 80GB GPU, received {gpu}")

for model in ("b3", "qsal"):
    path = Path(f"configs/experiments/case_{model}_a10080_epoch1_retry2.yaml")
    cfg = yaml.safe_load(path.read_text())
    t = cfg["training"]
    q = cfg["pqc" if model == "b3" else "qsal"]

    assert t["epochs"] == 1
    assert t["batch_size"] == 8
    assert t["max_train_batches"] is None
    assert t["max_val_batches"] is None
    assert q["backend"] == "default.qubit"
    assert q["diff_method"] == "backprop"

    print(f"{model.upper()}: CONFIGURATION PASS", flush=True)

    if Path(cfg["logging"]["run_dir"]).exists():
        raise RuntimeError("Run directory already exists")

    (results / path.name).write_bytes(path.read_bytes())

(results / "hardware.json").write_text(
    json.dumps({
        "gpu": gpu,
        "gpu_memory_gib": round(
            torch.cuda.get_device_properties(0).total_memory / 2**30, 3
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda
    }, indent=2) + "\n"
)
PY

echo "=== B3: REAL COMPLETE EPOCH ==="
B3_START=$(date +%s)

"$PYTHON" -u -m training.train_case_b3 \
  --config configs/experiments/case_b3_a10080_epoch1_retry2.yaml

B3_END=$(date +%s)
echo "B3 entrypoint elapsed: $((B3_END-B3_START)) seconds"

echo "=== QSAL: REAL COMPLETE EPOCH ==="
QSAL_START=$(date +%s)

"$PYTHON" -u -m training.train_case_qsal \
  --config configs/experiments/case_qsal_a10080_epoch1_retry2.yaml

QSAL_END=$(date +%s)
echo "QSAL entrypoint elapsed: $((QSAL_END-QSAL_START)) seconds"

"$PYTHON" - "$RESULTS" "$B3_START" "$B3_END" "$QSAL_START" "$QSAL_END" <<'PY'
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

output = Path(sys.argv[1])
b3s, b3e, qs, qe = map(int, sys.argv[2:])

rows = [
    {"model": "B3", "elapsed_seconds": b3e-b3s},
    {"model": "QSAL", "elapsed_seconds": qe-qs},
]

with (output / "epoch_duration.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "node": os.environ.get("SLURMD_NODENAME"),
    "git_commit": '0425b741b5b78abbf513f6f586dd986ae7bc7aab',
    "hardware": json.loads((output / "hardware.json").read_text()),
    "backend": "default.qubit",
    "diff_method": "backprop",
    "batch_size": 8,
    "training_windows": 28560,
    "validation_windows": 7140,
    "epochs_measured": 1,
    "timing_scope": "complete training entrypoint including initialization, training, validation, checkpoints and finalization",
    "results": rows,
}

(output / "epoch_duration.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)

print(json.dumps(summary, indent=2))
PY

echo "=== ALL MEASUREMENTS COMPLETED ==="
echo "Results: $RESULTS"
echo "End: $(date --iso-8601=seconds)"
