# training/train_baseline_b3.py
"""
Baseline B3 Training Script
---------------------------

This script trains the Baseline B3 model using the unified PQC fusion
architecture. It is now modality‑, dataset‑, and label‑agnostic, while
keeping the BaselineB3 architecture, quantum circuit, shapes, and
dimensions unchanged.

Pipeline:
    Step‑A: Dataset (UnifiedMultimodalDataset or SyntheticULMTSSTDataset)
    Step‑B: Synchronizer (raw → aligned T)
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
from datasets.ulm_tsst_synthetic import SyntheticULMTSSTDataset
from datasets.collate_fn import multimodal_collate_fn
from datasets.synchronizer import Synchronizer


# ---------------------------------------------------------
# RUN DIRECTORY CREATION
# ---------------------------------------------------------
def make_run_folder(base_dir: str = "runs/baseline_b3") -> str:
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
def select_device(cfg_device: str) -> torch.device:
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
# DATASET FACTORY
# ---------------------------------------------------------
def build_dataset(cfg, split: str):
    """
    Builds the appropriate dataset based on cfg["dataset"]["name"].

    Supported:
        - "recola" / "sewa" → UnifiedMultimodalDataset
        - "ulm_tsst_synthetic" → SyntheticULMTSSTDataset
    """
    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"]
    target_seq_len = ds_cfg["target_seq_len"]
    synthetic = ds_cfg["synthetic"]
    num_samples = ds_cfg.get("num_samples", 200)
    label_dim = ds_cfg.get("label_dim", 2)

    if name in ["recola", "sewa"]:
        return UnifiedMultimodalDataset(
            dataset_name=name,
            target_seq_len=target_seq_len,
            synthetic=synthetic,
            num_samples=num_samples,
        )

    if name == "ulm_tsst_synthetic":
        return SyntheticULMTSSTDataset(
            num_samples=num_samples,
            target_seq_len=target_seq_len,
            label_dim=label_dim,
            modalities=cfg["model"]["modalities"],
            device="cpu",
        )

    raise ValueError(f"Unsupported dataset name in Baseline B3 training: {name}")


# ---------------------------------------------------------
# MAIN TRAINING PIPELINE
# ---------------------------------------------------------
def main() -> None:
    """
    Full Baseline B3 training pipeline.
    """

    # 1. Load config
    cfg = load_config("configs/baseline_b3_ulm_tsst.yaml")

    # 2. Device
    device = select_device(cfg["training"]["device"])
    print(f"[INFO] Using device: {device}")

    # 3. Run directory
    run_dir = make_run_folder(cfg["logging"]["checkpoint_dir"])
    ckpt_path = os.path.join(run_dir, "best_model.pt")
    log_path = os.path.join(run_dir, "training_log.json")

    print(f"[INFO] Run directory: {run_dir}")

    with open(os.path.join(run_dir, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=4)

    # 4. Dataset
    train_set = build_dataset(cfg, split="train")
    val_set = build_dataset(cfg, split="val")

    dataset_name = cfg["dataset"]["name"]
    target_seq_len = cfg["dataset"]["target_seq_len"]
    forecast_horizon = cfg["dataset"]["forecast_horizon"]

    # Synchronizer (raw → aligned T)
    synchronizer = Synchronizer(
        target_seq_len=target_seq_len,
        device="cpu",
    )

    def sync_collate_fn(batch_list):
        """
        Applies Synchronizer to each sample before batching.
        """
        synced_batch_list = []
        for sample in batch_list:
            synced = synchronizer.synchronize(sample)
            synced_batch_list.append(synced)
        return multimodal_collate_fn(synced_batch_list)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=sync_collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=sync_collate_fn,
    )

    modalities = cfg["model"]["modalities"]
    model_cfg = cfg["model"]
    pqc_cfg = model_cfg["pqc"]

    # 5. Model (Baseline B3 wrapper)
    model = BaselineB3(
        emb_dim=model_cfg["emb_dim"],
        proj_dim=model_cfg["proj_dim"],
        transformer_dim=model_cfg["transformer_dim"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        target_seq_len=target_seq_len,
        forecast_horizon=forecast_horizon,
        synthetic=cfg["dataset"]["synthetic"],
        modalities=modalities,
        pqc_cfg=pqc_cfg,
        dataset_name=dataset_name,
    ).to(device)

    # Log PQC architecture (unchanged)
    print("\n[INFO] PQC Configuration:")
    print(f"       Qubits      : {pqc_cfg.get('num_qubits', 4)}")
    print(f"       Depth       : {pqc_cfg.get('depth', 3)}")
    print(f"       Backend     : lightning.qubit (CPU)")
    print(f"       Fusion Out  : 256-dim\n")

    # 6. Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        patience=cfg["training"]["patience"],
        save_path=ckpt_path,
        use_tqdm=True,
        log_batches=True,
    )

    # 8. Training loop
    history = []
    total_epochs = cfg["training"]["epochs"]

    print("\n[INFO] Starting Baseline B3 training...\n")

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()

        train_stats = trainer.train_one_epoch(
            train_loader=train_loader,
            epoch=epoch,
            total_epochs=total_epochs,
        )

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
