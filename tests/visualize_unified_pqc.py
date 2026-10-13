"""
tests.visualize_unified_pqc
===========================

UnifiedPQC Visualization & Validation Tool

Purpose
-------
Visualize and validate the actual quantum circuit
generated from configuration files.

Outputs
-------
reports/

    unified_pqc_report.csv

circuits/

    unified_pqc_diagram.txt

    unified_pqc_visualization.png

Validation
----------
✓ qubit count
✓ depth
✓ backend
✓ trainable parameters
✓ circuit generation
✓ thesis-ready visualization
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pennylane as qml
import torch

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

# ==========================================================
# Imports
# ==========================================================

from utils.config import load_config
from quantum.unified_pqc import UnifiedPQC


# ==========================================================
# Helpers
# ==========================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print("\n" + "=" * 70)
    print("UNIFIED PQC VISUALIZATION")
    print("=" * 70)

    # ------------------------------------------------------
    # Choose config
    # ------------------------------------------------------

    CONFIG_PATH = (
        "configs/qsal.yaml"
    )

    cfg = load_config(
        CONFIG_PATH
    )

    # ------------------------------------------------------
    # Determine section
    # ------------------------------------------------------

    if "qsal" in cfg:

        quantum_cfg = cfg["qsal"]

        depth_key = (
            "circuit_depth"
        )

    elif "pqc" in cfg:

        quantum_cfg = cfg["pqc"]

        depth_key = (
            "depth"
        )

    else:

        raise ValueError(
            "No quantum config found."
        )

    num_qubits = quantum_cfg[
        "num_qubits"
    ]

    depth = quantum_cfg[
        depth_key
    ]

    backend = quantum_cfg.get(
        "backend",
        "lightning.qubit",
    )

    # ------------------------------------------------------
    # Build Actual Circuit
    # ------------------------------------------------------

    pqc = UnifiedPQC(
        num_qubits=num_qubits,
        depth=depth,
        device_name=backend,
    )

    print(
        f"Qubits   : {num_qubits}"
    )

    print(
        f"Depth    : {depth}"
    )

    print(
        f"Backend  : {backend}"
    )

    print(
        f"Params   : "
        f"{count_parameters(pqc):,}"
    )

    # ------------------------------------------------------
    # Sample Input
    # ------------------------------------------------------

    angles = torch.zeros(
        num_qubits
    )

    # ------------------------------------------------------
    # Text Diagram
    # ------------------------------------------------------

    text_diagram = qml.draw(
        pqc.qnode
    )(
        angles,
        pqc.theta,
        pqc.phi,
    )

    circuits_dir = Path(
        "circuits"
    )

    reports_dir = Path(
        "reports"
    )

    circuits_dir.mkdir(
        exist_ok=True
    )

    reports_dir.mkdir(
        exist_ok=True
    )

    text_file = (
        circuits_dir /
        "unified_pqc_diagram.txt"
    )

    text_file.write_text(
        text_diagram,
        encoding="utf-8",
    )

    print(
        "\nText diagram saved:"
    )

    print(text_file)

    # ------------------------------------------------------
    # Matplotlib Circuit Figure
    # ------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(16, 8)
    )

    qml.draw_mpl(
        pqc.qnode
    )(
        angles,
        pqc.theta,
        pqc.phi,
    )

    image_file = (
        circuits_dir /
        "unified_pqc_visualization.png"
    )

    plt.tight_layout()

    plt.savefig(
        image_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Circuit figure saved:"
    )

    print(
        image_file
    )

    # ------------------------------------------------------
    # CSV Report
    # ------------------------------------------------------

    csv_file = (
        reports_dir /
        "unified_pqc_report.csv"
    )

    with open(
        csv_file,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "metric",
                "value",
            ]
        )

        writer.writerow(
            [
                "config",
                CONFIG_PATH,
            ]
        )

        writer.writerow(
            [
                "num_qubits",
                num_qubits,
            ]
        )

        writer.writerow(
            [
                "depth",
                depth,
            ]
        )

        writer.writerow(
            [
                "backend",
                backend,
            ]
        )

        writer.writerow(
            [
                "trainable_parameters",
                count_parameters(
                    pqc
                ),
            ]
        )

    print(
        "\nCSV report saved:"
    )

    print(
        csv_file
    )

    print("\n" + "=" * 70)
    print(
        "VISUALIZATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()