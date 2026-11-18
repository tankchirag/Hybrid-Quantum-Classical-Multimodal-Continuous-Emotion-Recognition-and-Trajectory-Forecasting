# Environment setup

Use Python 3.11 for the pinned dependencies in `requirements.txt`. The project metadata allows Python 3.10 and 3.11, but rejects Python 3.12 and later.

## Find Python 3.11 on the cluster

Start from a shell with the cluster's environment-modules system initialized:

```sh
deactivate 2>/dev/null || true
rm -rf .venv

echo "=== Available Python modules ==="
module spider Python 2>&1 | sed -n '1,160p'

echo
echo "=== Direct Python executables ==="
command -v python3.11 || true
command -v python3.10 || true
```

If `python3.11` is available directly, use it. Otherwise load one of the Python 3.11 modules reported by `module spider`, substituting its exact name:

```sh
module load <Python-3.11-module-name>
python3 --version
command -v python3
```

Confirm that the version is Python 3.11.x before proceeding. Do not continue with Python 3.13.

## Create and install the environment

From the repository root:

```sh
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

python --version
python -m pip --version
python -m pip install --upgrade pip setuptools wheel
```

Install the CUDA 12.1 PyTorch wheels separately from the PyTorch wheel index:

```sh
python -m pip install \
  torch==2.1.0+cu121 \
  torchvision==0.16.0+cu121 \
  torchaudio==2.1.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121 \
  2>&1 | tee install_torch.log
```

Check the installed build. CUDA availability can be false on a login node without an allocated GPU:

```sh
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available on current node:", torch.cuda.is_available())
PY
```

Install the other pinned requirements and the editable project:

```sh
grep -vE '^(torch|torchvision|torchaudio)==' requirements.txt > requirements-no-torch.txt
python -m pip install -r requirements-no-torch.txt 2>&1 | tee install_requirements.log
python -m pip install -e . 2>&1 | tee install_project.log
```

## Verify the environment

```sh
python - <<'PY'
import sys
import numpy
import pandas
import sklearn
import torch
import pennylane
import transformers
import yaml

print("Python:", sys.version)
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("PennyLane:", pennylane.__version__)
print("Transformers:", transformers.__version__)
print("PyYAML:", yaml.__version__)
print("CUDA visible:", torch.cuda.is_available())
PY

python -m pip check
```

Expected core versions are NumPy 1.26.4, pandas 2.1.4, scikit-learn 1.3.2, PyTorch 2.1.0+cu121, PennyLane 0.36.x, and Transformers 4.40.0.

## Run focused B3/QSAL tests

Start with configuration and model construction:

```sh
python -m pytest -q \
  tests/case/test_config_audit.py \
  tests/history/test_b3_build.py \
  tests/case/test_b3_case.py \
  tests/case/test_case_qsal_batch.py \
  tests/history/test_b3_training_step.py \
  tests/history/test_qsal_single_train_step.py \
  2>&1 | tee smoke_tests.log
```

If those pass, run the wider B3/QSAL checks:

```sh
python -m pytest -q \
  tests/case/test_b3_training_batch.py \
  tests/history/test_qsal_dataloader_forward.py \
  tests/history/test_qsal_mini_training.py \
  tests/test_qsal.py \
  2>&1 | tee b3_qsal_tests.log
```
