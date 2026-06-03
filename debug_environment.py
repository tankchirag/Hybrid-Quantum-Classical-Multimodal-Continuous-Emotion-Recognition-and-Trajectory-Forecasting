"""
Comprehensive Environment Diagnostic Report

Creates:

debug_report_YYYYMMDD_HHMMSS.txt
debug_report_YYYYMMDD_HHMMSS.json

Checks:

✓ Python
✓ OS
✓ Virtual Environment
✓ CPU
✓ RAM
✓ NumPy
✓ Pandas
✓ Matplotlib
✓ Scikit-Learn
✓ Torch
✓ TorchVision
✓ TorchAudio
✓ Transformers
✓ PennyLane
✓ Autograd
✓ Autoray
✓ CUDA
✓ cuDNN
✓ GPU
✓ NVIDIA-SMI
✓ Environment Variables
✓ Pip Packages
✓ Import Times
"""

import os
import sys
import json
import time
import platform
import subprocess
import importlib
from datetime import datetime

report_lines = []
report_json = {}


# ==========================================================
# Utilities
# ==========================================================

def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


def section(title):
    line = "=" * 100

    log("")
    log(line)
    log(title)
    log(line)


def save_reports():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    txt_file = f"debug_report_{timestamp}.txt"
    json_file = f"debug_report_{timestamp}.json"

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=4)

    print("\n")
    print("=" * 100)
    print("REPORTS SAVED")
    print("=" * 100)
    print(txt_file)
    print(json_file)


def run_command(cmd):

    try:

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# ==========================================================
# Import checker
# ==========================================================

def safe_import(module_name):

    start = time.time()

    try:

        module = importlib.import_module(module_name)

        elapsed = round(time.time() - start, 3)

        version = getattr(
            module,
            "__version__",
            "Unknown"
        )

        return {
            "status": "OK",
            "version": version,
            "import_time_sec": elapsed
        }

    except Exception as e:

        elapsed = round(time.time() - start, 3)

        return {
            "status": "FAILED",
            "import_time_sec": elapsed,
            "error": str(e)
        }


# ==========================================================
# Main
# ==========================================================

def main():

    # ------------------------------------------------------
    # Python
    # ------------------------------------------------------

    section("PYTHON")

    python_info = {
        "version": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    report_json["python"] = python_info

    for k, v in python_info.items():
        log(f"{k}: {v}")

    # ------------------------------------------------------
    # Environment
    # ------------------------------------------------------

    section("ENVIRONMENT")

    env_info = {
        "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
        "CONDA_PREFIX": os.environ.get("CONDA_PREFIX"),
        "CUDA_PATH": os.environ.get("CUDA_PATH"),
        "CUDA_HOME": os.environ.get("CUDA_HOME"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cwd": os.getcwd(),
    }

    report_json["environment"] = env_info

    for k, v in env_info.items():
        log(f"{k}: {v}")

    # ------------------------------------------------------
    # Module imports
    # ------------------------------------------------------

    section("MODULE IMPORT TESTS")

    modules = [
        "numpy",
        "pandas",
        "matplotlib",
        "sklearn",
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "pennylane",
        "autograd",
        "autoray",
    ]

    module_results = {}

    for module in modules:

        result = safe_import(module)

        module_results[module] = result

        log(
            f"{module:<15} "
            f"{result['status']:<8} "
            f"{result.get('version','')} "
            f"{result['import_time_sec']} sec"
        )

    report_json["modules"] = module_results

    # ------------------------------------------------------
    # Torch details
    # ------------------------------------------------------

    section("TORCH DETAILS")

    try:

        import torch

        torch_info = {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_version": torch.backends.cudnn.version(),
        }

        if torch.cuda.is_available():

            torch_info["gpu_name"] = torch.cuda.get_device_name(0)
            torch_info["compute_capability"] = \
                str(torch.cuda.get_device_capability(0))

        report_json["torch"] = torch_info

        for k, v in torch_info.items():
            log(f"{k}: {v}")

    except Exception as e:

        report_json["torch"] = str(e)

        log(str(e))

    # ------------------------------------------------------
    # NVIDIA-SMI
    # ------------------------------------------------------

    section("NVIDIA-SMI")

    nvidia = run_command(["nvidia-smi"])

    report_json["nvidia_smi"] = nvidia

    if "stdout" in nvidia:
        log(nvidia["stdout"])

    if "stderr" in nvidia:
        log(nvidia["stderr"])

    # ------------------------------------------------------
    # pip list
    # ------------------------------------------------------

    section("PIP LIST")

    pip_list = run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "list"
        ]
    )

    report_json["pip_list"] = pip_list

    if "stdout" in pip_list:
        log(pip_list["stdout"])

    # ------------------------------------------------------
    # pip show important packages
    # ------------------------------------------------------

    section("PACKAGE DETAILS")

    packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "pennylane",
        "autograd",
        "autoray"
    ]

    package_info = {}

    for package in packages:

        result = run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "show",
                package
            ]
        )

        package_info[package] = result

    report_json["package_details"] = package_info

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    save_reports()


if __name__ == "__main__":
    main()