# training/train_baseline_b3.py
"""
Baseline B3 Training Script
---------------------------

This script trains the Baseline B3 model using the unified PQC fusion
architecture. It is fully aligned with the updated unified modality
contract and the corrected BaselineB3 wrapper.

Pipeline:
    Step‑A: UnifiedMultimodalDataset (REAL mode)
    Step‑B: multimodal_collate_fn
    Step‑C: BaselineB3 wrapper (feature extractors + PQC fusion)
    Step‑D: Transformer Encoder–Decoder
    Step‑E: Recognition + Forecasting heads
    Step‑F: Trainer (losses, early stopping, checkpointing)

Outputs:
    • best_model.pt
    • training_log.json
    • config_used.json
"""

import os
import sys
import json
import time
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# Add project root to sys.path
# ---------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.config import load_config
from models.wrappers.baseline_b3 import BaselineB3
from training.trainer import Trainer
from datasets.unified_multimodal_dataset import UnifiedMultimodalDataset
from datasets.collate_fn import multimodal_collate_fn


# ---------------------------------------------------------
# RUN DIRECTORY CREATION
# ---------------------------------------------------------
def make_run_folder(base_dir="runs/baseline_b3"):
    """
    Creates a timestamped run directory:
        runs/baseline_b3/YYYY-MM-DD_HH-MM-SS/
    """
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ---------------------------------------------------------
# SAFE DEVICE SELECTION
# ---------------------------------------------------------
def select_device(cfg_device: str):
    """
    Selects CPU or CUDA device safely.
    PQC uses lightning.qubit → CPU backend, but CUDA is allowed
    for classical components (encoders, transformer).
    """
    if cfg_device == "cpu":
        return torch.device("cpu")
    if cfg_device == "cuda":
        return torch.device("cuda")
    try:
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


# ---------------------------------------------------------
# MAIN TRAINING PIPELINE
# ---------------------------------------------------------
def main():

    # ---------------------------------------------------------
    # 1. Load config
    # ---------------------------------------------------------
    cfg = load_config("configs/baseline_b3.yaml")

    # ---------------------------------------------------------
    # 2. Device
    # ---------------------------------------------------------
    device = select_device(cfg["training"]["device"])
    print(f"[INFO] Using device: {device}")

    # ---------------------------------------------------------
    # 3. Create run directory
    # ---------------------------------------------------------
    run_dir = make_run_folder(cfg["logging"]["checkpoint_dir"])
    ckpt_path = os.path.join(run_dir, "best_model.pt")
    log_path = os.path.join(run_dir, "training_log.json")

    print(f"[INFO] Run directory: {run_dir}")

    # Save config for reproducibility
    with open(os.path.join(run_dir, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=4)

    # ---------------------------------------------------------
    # 4. Dataset (REAL mode)
    # ---------------------------------------------------------
    dataset_name = cfg["dataset"]["name"]
    target_seq_len = cfg["dataset"]["target_seq_len"]
    forecast_horizon = cfg["dataset"]["forecast_horizon"]

    train_set = UnifiedMultimodalDataset(
        dataset_name=dataset_name,
        target_seq_len=target_seq_len,
        synthetic=False,
    )

    val_set = UnifiedMultimodalDataset(
        dataset_name=dataset_name,
        target_seq_len=target_seq_len,
        synthetic=False,
    )

    modalities = cfg["model"]["modalities"]

    # DataLoaders with correct collate_fn
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
    # 5. Model (Baseline B3 wrapper)
    # ---------------------------------------------------------
    model_cfg = cfg["model"]
    pqc_cfg = model_cfg["pqc"]

    model = BaselineB3(
        emb_dim=model_cfg["emb_dim"],
        proj_dim=model_cfg["proj_dim"],
        transformer_dim=model_cfg["transformer_dim"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        target_seq_len=target_seq_len,
        forecast_horizon=forecast_horizon,
        synthetic=False,
        modalities=modalities,
        pqc_cfg=pqc_cfg,
        dataset_name=dataset_name,
    ).to(device)

    # Log PQC architecture
    print("\n[INFO] PQC Configuration:")
    print(f"       Qubits      : {pqc_cfg.get('num_qubits', 4)}")
    print(f"       Depth       : {pqc_cfg.get('depth', 3)}")
    print(f"       Backend     : lightning.qubit (CPU)")
    print(f"       Fusion Out  : 256-dim\n")

    # ---------------------------------------------------------
    # 6. Optimizer
    # ---------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # ---------------------------------------------------------
    # 7. Trainer
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
    # 8. Training loop
    # ---------------------------------------------------------
    history = []
    total_epochs = cfg["training"]["epochs"]

    print("\n[INFO] Starting Baseline B3 training...\n")

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()

        # TRAIN
        train_stats = trainer.train_one_epoch(
            train_loader=train_loader,
            epoch=epoch,
            total_epochs=total_epochs,
        )

        # VALIDATE
        val_stats = trainer.eval_one_epoch(
            val_loader=val_loader,
            epoch=epoch,
            total_epochs=total_epochs,
        )

        epoch_time = time.time() - epoch_start

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

    print("\n[INFO] Baseline B3 training complete.")
    print(f"[INFO] Best model saved to: {ckpt_path}")
    print(f"[INFO] Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
