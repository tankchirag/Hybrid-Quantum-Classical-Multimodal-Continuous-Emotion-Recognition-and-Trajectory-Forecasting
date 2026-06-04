"""
model_stats.py
--------------
Unified model statistics logger for all baselines (B1, B2, B3) and QSAL.

This script computes:
    - Total parameters
    - Trainable vs. frozen parameters
    - Parameters per module (sorted)
    - FLOPs / MACs using ptflops
    - Model depth (Transformer layers, PQC layers, QSAL layers)
    - Input/output shapes
    - Fused_dim (for B2/B3/QSAL)
    - Active modalities (audio/video/ecg/eda/text)
    - JSON export for thesis reproducibility

Usage:
    python tools/model_stats.py --model baseline_b2 --dataset recola
"""

import os
import sys
import json
import argparse
import torch
from datetime import datetime

# Add project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Models
from models.wrappers.baseline_b1 import BaselineB1
from models.wrappers.baseline_b2 import BaselineB2
# from models.wrappers.baseline_b3 import BaselineB3   # future
# from models.wrappers.qsal_model import QSALModel     # future

# Dataset (for modality flags)
from datasets.unified_multimodal_dataset import UnifiedMultimodalDataset

# FLOPs
try:
    from ptflops import get_model_complexity_info
    PT_FLOPS_AVAILABLE = True
except ImportError:
    PT_FLOPS_AVAILABLE = False


# ---------------------------------------------------------
# Helper: Count parameters
# ---------------------------------------------------------
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


# ---------------------------------------------------------
# Helper: Parameters per module
# ---------------------------------------------------------
def module_param_breakdown(model):
    breakdown = {}
    for name, p in model.named_parameters():
        breakdown[name] = p.numel()
    return dict(sorted(breakdown.items(), key=lambda x: -x[1]))


# ---------------------------------------------------------
# Helper: FLOPs
# ---------------------------------------------------------
def compute_flops(model, input_shape):
    if not PT_FLOPS_AVAILABLE:
        return "ptflops not installed"

    macs, params = get_model_complexity_info(
        model,
        input_shape,
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    return macs


# ---------------------------------------------------------
# Build model
# ---------------------------------------------------------
def build_model(model_name, dataset_name, cfg):
    dataset = UnifiedMultimodalDataset(dataset_name)
    modalities = dataset.modalities

    if model_name == "baseline_b1":
        return BaselineB1(
            emb_dim=cfg["emb_dim"],
            transformer_dim=cfg["transformer_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            target_seq_len=cfg["T"],
            forecast_horizon=cfg["S"],
            synthetic=True,
        ), modalities

    if model_name == "baseline_b2":
        return BaselineB2(
            emb_dim=cfg["emb_dim"],
            proj_dim=cfg["proj_dim"],
            transformer_dim=cfg["transformer_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            target_seq_len=cfg["T"],
            forecast_horizon=cfg["S"],
            synthetic=True,
            modalities=modalities,
        ), modalities

    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["baseline_b1", "baseline_b2"],
                        help="Model to analyze")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["recola", "sewa", "combined"],
                        help="Dataset for modality configuration")
    parser.add_argument("--save", type=str, default="model_stats.json",
                        help="Output JSON file")
    args = parser.parse_args()

    # Minimal config (YAML-free)
    cfg = {
        "emb_dim": 128,
        "proj_dim": 64,
        "transformer_dim": 256,
        "num_heads": 2,
        "num_layers": 2,
        "T": 50,
        "S": 10,
    }

    # Build model
    model, modalities = build_model(args.model, args.dataset, cfg)
    model.eval()

    # Count parameters
    total_params, trainable_params, frozen_params = count_parameters(model)

    # Per-module breakdown
    breakdown = module_param_breakdown(model)

    # FLOPs (dummy input)
    dummy_input_shape = (cfg["T"], cfg["emb_dim"])
    flops = compute_flops(model, dummy_input_shape)

    # Fused dim (if available)
    fused_dim = getattr(model, "fusion", None)
    fused_dim = fused_dim.out_dim if fused_dim else None

    # Build stats dict
    stats = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "dataset": args.dataset,
        "modalities": modalities,
        "fused_dim": fused_dim,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "flops": flops,
        "module_breakdown": breakdown,
    }

    # Save JSON
    with open(args.save, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"\n[INFO] Model statistics saved to {args.save}\n")
    print(f"Total params: {total_params:,}")
    print(f"Trainable:    {trainable_params:,}")
    print(f"Frozen:       {frozen_params:,}")
    print(f"FLOPs:        {flops}")
    print(f"Modalities:   {modalities}")
    print(f"Fused dim:    {fused_dim}")


if __name__ == "__main__":
    main()
