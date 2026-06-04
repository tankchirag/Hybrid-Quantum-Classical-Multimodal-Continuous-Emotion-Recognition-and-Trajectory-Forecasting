"""
Training script for Baseline B2 (Unified Concatenation Fusion).

This script mirrors the Baseline B1 training pipeline but replaces the
fusion mechanism with the Baseline B2 fusion strategy:

    - Per-modality 128→64 projections
    - Concatenation into a fused multimodal token
    - Shared Transformer Encoder–Decoder for temporal modeling
    - Shared recognition and forecasting heads

Features:
    - YAML config loading
    - Robust device auto-selection (Windows-safe)
    - Unique run directory per experiment
    - tqdm progress bars (batch-level)
    - Epoch timing + ETA estimation
    - JSON logging
    - Safe checkpoint naming (no overwrites)
"""

import os
import sys
import json
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------
# Add project root to sys.path
# ---------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.config import load_config
from models.wrappers.baseline_b2 import BaselineB2
from training.trainer import Trainer

# Dataset + collate
from datasets.unified_multimodal_dataset import UnifiedMultimodalDataset
from datasets.collate_fn import multimodal_collate_fn


# ---------------------------------------------------------
# RUN DIRECTORY CREATION
# ---------------------------------------------------------
def make_run_folder(base_dir="runs/baseline_b2"):
    """
    Creates a unique run directory for Baseline B2 experiments.
    """
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ---------------------------------------------------------
# SAFE DEVICE SELECTION (Windows-friendly)
# ---------------------------------------------------------
def select_device(cfg_device: str):
    """
    Device selection logic that avoids Windows CUDA hangs.

    Modes:
        - "cpu"  → always CPU
        - "cuda" → force CUDA (if available)
        - "auto" → try CUDA, fallback to CPU safely
    """
    if cfg_device == "cpu":
        return torch.device("cpu")

    if cfg_device == "cuda":
        return torch.device("cuda")

    # AUTO MODE (safe)
    try:
        _ = torch.cuda.device_count()
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


# ---------------------------------------------------------
# MAIN TRAINING PIPELINE
# ---------------------------------------------------------
def main():

    # ---------------------------------------------------------
    # LOAD CONFIG
    # ---------------------------------------------------------
    cfg = load_config("configs/baseline_b2.yaml")

    # ---------------------------------------------------------
    # DEVICE SELECTION
    # ---------------------------------------------------------
    cfg_device = cfg["training"].get("device", "auto")
    device = select_device(cfg_device)

    print(f"[INFO] Using device: {device}")

    # ---------------------------------------------------------
    # CREATE RUN FOLDER
    # ---------------------------------------------------------
    run_dir = make_run_folder()
    ckpt_path = os.path.join(run_dir, "best_model.pt")
    log_path = os.path.join(run_dir, "training_log.json")

    print(f"[INFO] Run directory: {run_dir}")

    # ---------------------------------------------------------
    # DATASETS (REAL MODE)
    # ---------------------------------------------------------
    dataset_name = cfg["dataset"].get("name", "recola")

    train_set = UnifiedMultimodalDataset(dataset_name=dataset_name)
    val_set   = UnifiedMultimodalDataset(dataset_name=dataset_name)

    # Extract modality flags from dataset
    modalities = train_set.modalities

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=multimodal_collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=multimodal_collate_fn,
    )

    # ---------------------------------------------------------
    # MODEL (Baseline B2, modality-aware)
    # ---------------------------------------------------------
    model = BaselineB2(
        emb_dim=cfg["model"]["emb_dim"],
        proj_dim=cfg["model"]["proj_dim"],
        transformer_dim=cfg["model"]["transformer_dim"],
        num_heads=cfg["model"]["num_heads"],
        num_layers=cfg["model"]["num_layers"],
        target_seq_len=cfg["dataset"]["T"],
        forecast_horizon=cfg["dataset"]["S"],
        synthetic=False,                     # REAL MODE
        modalities=modalities,               # <-- NEW (automatic modality config)
    ).to(device)

    # ---------------------------------------------------------
    # OPTIMIZER
    # ---------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"]
    )

    # ---------------------------------------------------------
    # TRAINER
    # ---------------------------------------------------------
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        patience=cfg["training"]["patience"],
        save_path=ckpt_path,
        use_tqdm=True,
        log_batches=True,
    )

    # ---------------------------------------------------------
    # TRAIN LOOP WITH TIMING
    # ---------------------------------------------------------
    history = []
    total_epochs = cfg["training"]["epochs"]

    print("\n[INFO] Starting Baseline B2 training...\n")

    for epoch in range(1, total_epochs + 1):

        epoch_start = time.time()

        # -------------------------
        # TRAIN
        # -------------------------
        train_stats = trainer.train_one_epoch(
            train_loader=train_loader,
            epoch=epoch,
            total_epochs=total_epochs,
        )

        # -------------------------
        # VALIDATE
        # -------------------------
        val_stats = trainer.eval_one_epoch(
            val_loader=val_loader,
            epoch=epoch,
            total_epochs=total_epochs,
        )

        epoch_time = time.time() - epoch_start

        # -------------------------
        # LOGGING
        # -------------------------
        stats = {
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
            "epoch_time_sec": round(epoch_time, 2),
        }
        history.append(stats)

        print(
            f"[Epoch {epoch}/{total_epochs}] "
            f"train_loss={train_stats['loss']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} | "
            f"time={epoch_time:.2f}s"
        )

        with open(log_path, "w") as f:
            json.dump(history, f, indent=4)

    print("\n[INFO] Baseline B2 training complete.")
    print(f"[INFO] Best model saved to: {ckpt_path}")
    print(f"[INFO] Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
