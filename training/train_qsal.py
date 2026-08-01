# training/train_qsal.py
"""
QSAL Training Script
--------------------

This script implements the complete QSAL (Quantum Self‑Attention Layer)
training pipeline, fully aligned with the unified modality contract:

    Step‑A: UnifiedMultimodalDataset (synthetic mode)
    Step‑B: MultimodalFeatureExtractors (synthetic encoders)
    Step‑C: ModalityTokenizer + QSALFusion
    Step‑D: Transformer Encoder–Decoder
    Step‑E: Recognition + Forecasting Heads

Key Features:
    • Synthetic dataset for QSAL debugging
    • Correct modality flags from config
    • Correct collate_fn for multimodal batching
    • Timestamped run directory
    • Training + validation loops
    • Best model checkpoint + training log
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
from models.wrappers.qsal import QSAL
from training.trainer import Trainer
from datasets.unified_multimodal_dataset import UnifiedMultimodalDataset
from datasets.collate_fn import multimodal_collate_fn


# ---------------------------------------------------------
# RUN DIRECTORY CREATION
# ---------------------------------------------------------
def make_run_folder(base_dir: str = "runs/qsal") -> str:
    """
    Creates a timestamped run directory:
        runs/qsal/YYYY-MM-DD_HH-MM-SS/
    """
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ---------------------------------------------------------
# SAFE DEVICE SELECTION
# ---------------------------------------------------------
def select_device(cfg_device: str) -> torch.device:
    """
    Selects CPU or CUDA device safely.
    QSAL uses lightning.qubit → CPU backend, but CUDA is allowed for
    classical parts (encoders, transformer).
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
# MODEL BUILDER (QSAL)
# ---------------------------------------------------------
def build_model(cfg, modalities) -> QSAL:
    """
    Builds QSAL wrapper model from configuration.

    Uses:
        • model.emb_dim
        • model.proj_dim
        • model.transformer_dim
        • model.num_heads
        • model.num_layers
        • model.target_seq_len
        • model.modalities (dict[str, bool])
        • model.qsal (quantum config)
        • dataset.forecast_horizon
        • dataset.synthetic
        • dataset.name
    """
    model_cfg = cfg["model"]
    qsal_cfg = model_cfg["qsal"]

    return QSAL(
        emb_dim=model_cfg["emb_dim"],
        proj_dim=model_cfg["proj_dim"],
        transformer_dim=model_cfg["transformer_dim"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        target_seq_len=model_cfg["target_seq_len"],
        forecast_horizon=cfg["dataset"]["forecast_horizon"],
        synthetic=cfg["dataset"]["synthetic"],   # QSAL uses synthetic=True
        modalities=model_cfg["modalities"],
        qsal_cfg=qsal_cfg,
        dataset_name=cfg["dataset"]["name"],
    )


# ---------------------------------------------------------
# MAIN TRAINING PIPELINE
# ---------------------------------------------------------
def main() -> None:
    """
    Full QSAL training pipeline:

        1. Load config
        2. Select device
        3. Create run directory
        4. Build datasets + dataloaders
        5. Build QSAL model
        6. Create optimizer + Trainer
        7. Run training loop
        8. Save best model + logs
    """

    # -----------------------------------------------------
    # 1. Load config
    # -----------------------------------------------------
    cfg = load_config("configs/qsal.yaml")

    # -----------------------------------------------------
    # 2. Device
    # -----------------------------------------------------
    device = select_device(cfg["training"]["device"])
    print(f"[INFO] Using device: {device}")

    # -----------------------------------------------------
    # 3. Create run directory
    # -----------------------------------------------------
    run_dir = make_run_folder(cfg["logging"]["run_dir"])
    ckpt_path = os.path.join(run_dir, "best_model.pt")
    log_path = os.path.join(run_dir, "training_log.json")

    print(f"[INFO] Run directory: {run_dir}")

    # Save config for reproducibility
    with open(os.path.join(run_dir, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=4)

    # -----------------------------------------------------
    # 4. Dataset (Step‑A)
    # -----------------------------------------------------
    dataset_name = cfg["dataset"]["name"]
    target_seq_len = cfg["dataset"]["target_seq_len"]
    num_samples = cfg["dataset"]["num_samples"]
    synthetic = cfg["dataset"]["synthetic"]

    train_set = UnifiedMultimodalDataset(
        dataset_name=dataset_name,
        target_seq_len=target_seq_len,
        synthetic=synthetic,
        num_samples=num_samples,
    )

    val_set = UnifiedMultimodalDataset(
        dataset_name=dataset_name,
        target_seq_len=target_seq_len,
        synthetic=synthetic,
        num_samples=num_samples,
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

    # -----------------------------------------------------
    # 5. Model (QSAL wrapper)
    # -----------------------------------------------------
    model = build_model(cfg, modalities).to(device)

    # -----------------------------------------------------
    # 6. Optimizer
    # -----------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # -----------------------------------------------------
    # 7. Trainer
    # -----------------------------------------------------
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        patience=cfg["training"]["patience"],
        save_path=ckpt_path,
        use_tqdm=True,
        log_batches=False,
    )

    # -----------------------------------------------------
    # 8. Training loop
    # -----------------------------------------------------
    history = []
    total_epochs = cfg["training"]["epochs"]

    print("\n[INFO] Starting QSAL training...\n")

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

    print("\n[INFO] QSAL training complete.")
    print(f"[INFO] Best model saved to: {ckpt_path}")
    print(f"[INFO] Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
