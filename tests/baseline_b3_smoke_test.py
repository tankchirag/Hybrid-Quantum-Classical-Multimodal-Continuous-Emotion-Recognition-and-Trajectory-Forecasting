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
from models.wrappers.baseline_b3 import BaselineB3


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
    print("BASELINE B3 VALIDATION / SMOKE TEST")
    print("=" * 70)

    # ======================================================
    # Load Config
    # ======================================================

    config = load_config(
        "configs/baseline_b3.yaml"
    )

    model = BaselineB3(
        config
    )

    fusion = model.fusion

    # ======================================================
    # PQC CONFIG
    # ======================================================

    print("\n[PQC CONFIG]")
    print("-" * 50)

    if hasattr(
        fusion,
        "num_qubits",
    ):
        print(
            f"num_qubits      : {fusion.num_qubits}"
        )

    if hasattr(
        fusion,
        "pqc_depth",
    ):
        print(
            f"pqc_depth       : {fusion.pqc_depth}"
        )

    if hasattr(
        fusion,
        "backend",
    ):
        print(
            f"backend         : {fusion.backend}"
        )

    # ======================================================
    # PQC INSTANCE DETAILS
    # ======================================================

    print("\n[PQC INSTANCE]")
    print("-" * 50)

    pqc = None

    if hasattr(
        fusion,
        "pqc",
    ):
        pqc = fusion.pqc

    elif hasattr(
        fusion,
        "quantum_fusion",
    ):
        pqc = fusion.quantum_fusion

    elif hasattr(
        fusion,
        "unified_pqc",
    ):
        pqc = fusion.unified_pqc

    if pqc is not None:

        if hasattr(
            pqc,
            "num_qubits",
        ):
            print(
                f"PQC qubits      : "
                f"{pqc.num_qubits}"
            )

        if hasattr(
            pqc,
            "depth",
        ):
            print(
                f"PQC depth       : "
                f"{pqc.depth}"
            )

    else:

        print(
            "Could not automatically locate "
            "PQC instance."
        )

    # ======================================================
    # Modalities
    # ======================================================

    print("\n[MODALITIES]")
    print("-" * 50)

    active = []

    modalities_cfg = config["modalities"]

    for m, enabled in modalities_cfg.items():

        if enabled:

            active.append(m)

    print(
        f"Active modalities = "
        f"{len(active)}"
    )

    for m in active:

        print(f"  {m}")

    # ======================================================
    # Synthetic Fusion Test
    # ======================================================

    print("\n[FUSION TEST]")
    print("-" * 50)

    B = 2
    T = 10
    D = 128

    embeddings = {}

    for m in active:

        embeddings[m] = torch.randn(
            B,
            T,
            D,
        )

    # ------------------------------------------------------
    # Try direct fusion call
    # ------------------------------------------------------

    fusion_success = False

    try:

        with torch.no_grad():

            fused = fusion(
                embeddings
            )

        print(
            f"Fusion shape = "
            f"{tuple(fused.shape)}"
        )

        fusion_success = True

    except Exception as e:

        print(
            f"Fusion forward skipped "
            f"({type(e).__name__})"
        )

        fused = None

    # ======================================================
    # Parameters
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
        f"Total      : "
        f"{total_params:,}"
    )

    print(
        f"Trainable  : "
        f"{trainable_params:,}"
    )

    print(
        f"Fusion     : "
        f"{fusion_params:,}"
    )

    # ======================================================
    # Save CSV
    # ======================================================

    report_file = Path(
        "baseline_b3_validation_report.csv"
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

        if hasattr(
            fusion,
            "num_qubits",
        ):
            writer.writerow(
                [
                    "num_qubits",
                    fusion.num_qubits,
                ]
            )

        if hasattr(
            fusion,
            "pqc_depth",
        ):
            writer.writerow(
                [
                    "pqc_depth",
                    fusion.pqc_depth,
                ]
            )

        writer.writerow(
            [
                "active_modalities",
                len(active),
            ]
        )

        writer.writerow(
            [
                "total_params",
                total_params,
            ]
        )

        writer.writerow(
            [
                "fusion_params",
                fusion_params,
            ]
        )

        if fusion_success:

            writer.writerow(
                [
                    "fusion_shape",
                    str(
                        tuple(
                            fused.shape
                        )
                    ),
                ]
            )

    print("\n[REPORT]")
    print("-" * 50)

    print(
        f"Saved: "
        f"{report_file.resolve()}"
    )

    print("\n" + "=" * 70)
    print("BASELINE B3 VALIDATION COMPLETED")
    print("=" * 70)
    print(config["pqc"]["num_qubits"])


if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        print("\n" + "=" * 70)
        print("BASELINE B3 VALIDATION FAILED")
        print("=" * 70)

        print(
            type(e).__name__
        )

        print(
            str(e)
        )

        raise