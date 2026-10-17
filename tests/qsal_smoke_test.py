from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

# ==========================================================
# Add Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

# ==========================================================
# Imports
# ==========================================================

from utils.config import load_config
from models.wrappers.qsal_half import QSAL


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


def count_trainable_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    print("\n" + "=" * 70)
    print("QSAL VALIDATION / SMOKE TEST")
    print("=" * 70)

    # ======================================================
    # Load Config
    # ======================================================

    config = load_config(
        "configs/qsal.yaml"
    )

    model = QSAL(config)

    fusion = model.fusion

    print("\n[CONFIG CHECK]")
    print("-" * 50)

    print(
        f"num_qubits      : {fusion.num_qubits}"
    )

    print(
        f"circuit_depth   : {fusion.circuit_depth}"
    )

    print(
        f"backend         : {fusion.backend}"
    )

    print(
        f"pqc_input_dim   : {fusion.pqc_input_dim}"
    )

    print(
        f"pqc_output_dim  : {fusion.pqc_output_dim}"
    )

    print(
        f"transformer_dim : {fusion.model_dim}"
    )

    # ======================================================
    # Linear Projection Validation
    # ======================================================

    print("\n[PROJECTION CHECK]")
    print("-" * 50)

    linear_out = (
        fusion.to_pqc_input.out_features
    )

    print(
        fusion.to_pqc_input
    )

    print(
        f"projection_out_features={linear_out}"
    )

    # ======================================================
    # PQC Validation
    # ======================================================

    print("\n[PQC CHECK]")
    print("-" * 50)

    print(
        f"Q PQC qubits : "
        f"{fusion.q_pqc.num_qubits}"
    )

    print(
        f"K PQC qubits : "
        f"{fusion.k_pqc.num_qubits}"
    )

    print(
        f"V PQC qubits : "
        f"{fusion.v_pqc.num_qubits}"
    )

    # ======================================================
    # Active Modalities
    # ======================================================

    print("\n[MODALITIES]")
    print("-" * 50)

    active = []

    modalities_cfg = config["modalities"]

    for m, enabled in modalities_cfg.items():
        if enabled:
            active.append(m)

    print(
        f"Active modalities = {len(active)}"
    )

    for m in active:
        print(f"  {m}")

    # ======================================================
    # Synthetic Fusion Test
    # ======================================================

    print("\n[FUSION SHAPE CHECK]")
    print("-" * 50)

    B = 2
    T = 10

    embeddings = {}

    for m in active:

        embeddings[m] = torch.randn(
            B,
            T,
            fusion.emb_dim,
        )

    with torch.no_grad():

        fused = fusion(
            embeddings
        )

    print(
        f"Fusion shape = "
        f"{tuple(fused.shape)}"
    )

    # ======================================================
    # Parameter Report
    # ======================================================

    total_params = count_parameters(
        model
    )

    trainable_params = (
        count_trainable_parameters(
            model
        )
    )

    fusion_params = count_parameters(
        fusion
    )

    print("\n[PARAMETERS]")
    print("-" * 50)

    print(
        f"Total      : {total_params:,}"
    )

    print(
        f"Trainable  : {trainable_params:,}"
    )

    print(
        f"Fusion     : {fusion_params:,}"
    )

    # ======================================================
    # Save CSV
    # ======================================================

    report_file = Path(
        "qsal_validation_report.csv"
    )

    with open(
        report_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            ["metric", "value"]
        )

        writer.writerow(
            ["num_qubits", fusion.num_qubits]
        )

        writer.writerow(
            ["circuit_depth", fusion.circuit_depth]
        )

        writer.writerow(
            ["backend", fusion.backend]
        )

        writer.writerow(
            ["projection_out_features", linear_out]
        )

        writer.writerow(
            ["q_qubits", fusion.q_pqc.num_qubits]
        )

        writer.writerow(
            ["k_qubits", fusion.k_pqc.num_qubits]
        )

        writer.writerow(
            ["v_qubits", fusion.v_pqc.num_qubits]
        )

        writer.writerow(
            ["active_modalities", len(active)]
        )

        writer.writerow(
            ["fusion_shape", str(tuple(fused.shape))]
        )

        writer.writerow(
            ["fusion_params", fusion_params]
        )

        writer.writerow(
            ["total_params", total_params]
        )

    print("\n[REPORT]")
    print("-" * 50)

    print(
        f"Saved: {report_file.resolve()}"
    )

    print("\n" + "=" * 70)

    print(config["qsal"]["num_qubits"])

if __name__ == "__main__":
    try:
        
        main()
    
    except Exception as e:
            
        print("\n" + "=" * 70)
        print("QSAL VALIDATION FAILED")
        print("=" * 70)
        
        print(type(e).__name__)
        print(str(e))
        
        raise