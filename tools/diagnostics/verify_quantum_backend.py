import importlib.metadata
import importlib.util
import sys
import time

import pennylane as qml
import torch

print("=== SOFTWARE ===")
print("Python:", sys.version.replace("\n", " "))
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("PennyLane:", qml.__version__)

for package in (
    "pennylane",
    "pennylane-lightning",
    "pennylane-lightning-gpu",
):
    try:
        print(f"{package}: {importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")

print("\n=== MODULE SPECS ===")
for module in (
    "pennylane_lightning",
    "pennylane_lightning.lightning_qubit",
    "pennylane_lightning.lightning_gpu",
):
    spec = importlib.util.find_spec(module)
    if spec is None:
        print(f"{module}: NOT FOUND")
    else:
        print(
            f"{module}: "
            f"origin={spec.origin}, "
            f"locations={list(spec.submodule_search_locations or [])}"
        )

print("\n=== TORCH CUDA ===")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
else:
    print("No CUDA device visible to PyTorch")

print("\n=== PENNYLANE DEVICE TESTS ===")

def test_device(name):
    try:
        dev = qml.device(name, wires=4)

        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def circuit(x, weights):
            qml.AngleEmbedding(x, wires=range(4), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(4))
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]

        x = torch.tensor(
            [0.1, 0.2, 0.3, 0.4],
            dtype=torch.float64,
            requires_grad=True,
        )
        weights = torch.randn(
            (3, 4, 3),
            dtype=torch.float64,
            requires_grad=True,
        )

        start = time.perf_counter()
        output = torch.stack(circuit(x, weights))
        loss = output.square().mean()
        loss.backward()
        elapsed = time.perf_counter() - start

        print(f"{name}: PASS")
        print("  device:", dev)
        print("  class:", type(dev))
        print("  module:", type(dev).__module__)
        print("  output:", output.detach().cpu().tolist())
        print("  loss:", float(loss.detach()))
        print("  input_grad_finite:", bool(torch.isfinite(x.grad).all()))
        print("  weight_grad_finite:", bool(torch.isfinite(weights.grad).all()))
        print("  elapsed_seconds:", elapsed)

    except Exception as exc:
        print(f"{name}: FAIL")
        print(f"  {type(exc).__name__}: {exc}")

for device_name in (
    "lightning.qubit",
    "lightning.gpu",
    "default.qubit",
):
    test_device(device_name)
