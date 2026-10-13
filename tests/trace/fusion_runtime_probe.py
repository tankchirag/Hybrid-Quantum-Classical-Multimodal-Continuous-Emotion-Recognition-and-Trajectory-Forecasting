"""
tests/trace/fusion_runtime_probe.py

Deep QSAL Fusion Runtime Probe

Purpose
-------
Captures the complete internal execution path
of QSALFusion and verifies:

Theory
=
Implementation
=
Runtime

Outputs:

trace_reports/qsal/qsal_fusion_probe.txt
trace_reports/qsal/qsal_fusion_probe.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from utils.config import load_config
from models.wrappers.qsal import QSAL
from training.base_training_script import (
    build_dataset,
    to_namespace,
)
from datasets.loaders.collate import collate


# ==========================================================
# Utilities
# ==========================================================

REPORT_LINES = []


def log(text=""):
    print(text)
    REPORT_LINES.append(text)


def shape(x):

    if isinstance(x, torch.Tensor):
        return tuple(x.shape)

    return str(type(x))


# ==========================================================
# Main
# ==========================================================

def main():

    config = load_config(
        "configs/qsal.yaml"
    )

    config = to_namespace(
        config
    )

    dataset = build_dataset(
        config,
        split="train",
    )

    sample = dataset[0]

    batch = collate(
        [sample]
    )

    model = QSAL(config)

    model.eval()

    fusion = model.fusion

    embeddings = model.extractors(
        batch["modalities"]
    )

    # ======================================================
    # Stage 1
    # ======================================================

    log("=" * 80)
    log("QSAL INTERNAL FUSION PROBE")
    log("=" * 80)

    log("")
    log("STAGE 1")
    log("ENCODER OUTPUTS")
    log("-" * 80)

    for name, emb in embeddings.items():

        log(
            f"{name:20}"
            f"{tuple(emb.shape)}"
        )

    # ======================================================
    # Stage 2
    # ======================================================

    x = fusion._stack_sources(
        embeddings
    )

    B, T, N, D = x.shape

    log("")
    log("STAGE 2")
    log("SOURCE STACK")
    log("-" * 80)

    log(
        f"Theory : (B,T,N,128)"
    )

    log(
        f"Runtime: {tuple(x.shape)}"
    )

    # ======================================================
    # Stage 3
    # ======================================================

    x_proj = fusion.to_pqc_input(
        x
    )

    log("")
    log("STAGE 3")
    log("QUANTUM PROJECTION")
    log("-" * 80)

    log(
        "Theory : (B,T,N,Q)"
    )

    log(
        f"Runtime: {tuple(x_proj.shape)}"
    )

    # ======================================================
    # Stage 4
    # ======================================================

    x_flat = x_proj.reshape(
        B * T * N,
        fusion.pqc_input_dim,
    )

    x_pqc = x_flat.unsqueeze(1)

    log("")
    log("STAGE 4")
    log("RESHAPE FOR PQC")
    log("-" * 80)

    log(
        "Theory : (B*T*N,1,Q)"
    )

    log(
        f"Runtime: {tuple(x_pqc.shape)}"
    )

    # ======================================================
    # Stage 5
    # ======================================================

    q_flat = (
        fusion.q_pqc(
            x_pqc
        )
        .squeeze(1)
    )

    log("")
    log("STAGE 5")
    log("Q-PQC")
    log("-" * 80)

    log(
        f"{tuple(q_flat.shape)}"
    )

    # ======================================================
    # Stage 6
    # ======================================================

    k_flat = (
        fusion.k_pqc(
            x_pqc
        )
        .squeeze(1)
    )

    log("")
    log("STAGE 6")
    log("K-PQC")
    log("-" * 80)

    log(
        f"{tuple(k_flat.shape)}"
    )

    # ======================================================
    # Stage 7
    # ======================================================

    v_flat = (
        fusion.v_pqc(
            x_pqc
        )
        .squeeze(1)
    )

    log("")
    log("STAGE 7")
    log("V-PQC")
    log("-" * 80)

    log(
        f"{tuple(v_flat.shape)}"
    )

    # ======================================================
    # Stage 8
    # ======================================================

    q = q_flat.view(
        B * T,
        N,
        fusion.pqc_output_dim,
    )

    k = k_flat.view(
        B * T,
        N,
        fusion.pqc_output_dim,
    )

    v = v_flat.view(
        B * T,
        N,
        fusion.pqc_output_dim,
    )

    log("")
    log("STAGE 8")
    log("RESTORE SOURCE STRUCTURE")
    log("-" * 80)

    log(
        f"Q : {tuple(q.shape)}"
    )

    log(
        f"K : {tuple(k.shape)}"
    )

    log(
        f"V : {tuple(v.shape)}"
    )

    # ======================================================
    # Stage 9
    # ======================================================

    out_attn = fusion.attention(
        q=q,
        k=k,
        v=v,
        mask=None,
    )

    log("")
    log("STAGE 9")
    log("GAUSSIAN ATTENTION")
    log("-" * 80)

    log(
        f"{tuple(out_attn.shape)}"
    )

    # ======================================================
    # Stage 10
    # ======================================================

    fused_pqc = out_attn.mean(
        dim=1
    )

    log("")
    log("STAGE 10")
    log("SOURCE POOLING")
    log("-" * 80)

    log(
        f"{tuple(fused_pqc.shape)}"
    )

    # ======================================================
    # Stage 11
    # ======================================================

    fused = fusion.pqc_to_fused(
        fused_pqc
    )

    log("")
    log("STAGE 11")
    log("Q → FUSED")
    log("-" * 80)

    log(
        f"{tuple(fused.shape)}"
    )

    # ======================================================
    # Stage 12
    # ======================================================

    fused_model = (
        fusion.fused_to_model(
            fused
        )
    )

    log("")
    log("STAGE 12")
    log("FUSED → MODEL")
    log("-" * 80)

    log(
        f"{tuple(fused_model.shape)}"
    )

    # ======================================================
    # Stage 13
    # ======================================================

    fused_model = fused_model.view(
        B,
        T,
        fusion.model_dim,
    )

    log("")
    log("STAGE 13")
    log("RESTORE TEMPORAL SHAPE")
    log("-" * 80)

    log(
        f"{tuple(fused_model.shape)}"
    )

    # ======================================================
    # Save
    # ======================================================

    report_dir = (
        PROJECT_ROOT
        / "trace_reports"
        / "qsal"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    txt_file = (
        report_dir
        / "qsal_fusion_probe.txt"
    )

    txt_file.write_text(
        "\n".join(REPORT_LINES),
        encoding="utf-8",
    )

    print("")
    print("=" * 80)
    print(
        f"Saved: {txt_file}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()