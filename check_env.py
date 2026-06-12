import torch
import pennylane as qml
import numpy as np
import yaml
import sklearn
import pandas as pd
import matplotlib
import transformers

print("==============================================")
print(" PYTHON VERSION")
print("==============================================")
import sys
print(sys.version)

print("\n==============================================")
print(" GPU / CUDA STATUS")
print("==============================================")
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
print("Torch version:", torch.__version__)
if torch.cuda.is_available():
    print("Torch device:", torch.cuda.get_device_name(0))

print("\n==============================================")
print(" PENNYLANE + LIGHTNING.GPU")
print("==============================================")
print("PennyLane version:", qml.__version__)

try:
    dev = qml.device("lightning.gpu", wires=2)
    print("lightning.gpu backend: AVAILABLE ✔")
except Exception as e:
    print("lightning.gpu backend: NOT AVAILABLE ✘")
    print("Error:", e)

print("\n==============================================")
print(" SCIENTIFIC STACK VERSIONS")
print("==============================================")
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("matplotlib:", matplotlib.__version__)
print("scikit-learn:", sklearn.__version__)
print("transformers:", transformers.__version__)
print("pyyaml:", yaml.__version__)

print("\n==============================================")
print(" GPU VISIBILITY (nvidia-smi)")
print("==============================================")
import subprocess
try:
    out = subprocess.check_output(["nvidia-smi"]).decode()
    print(out)
except Exception as e:
    print("nvidia-smi not available:", e)

print("\n==============================================")
print(" ALL CHECKS COMPLETE")
print("==============================================")
