# training/train_qsal.py
"""
QSAL Training Script
--------------------

This script implements the complete QSAL (Quantum Self‑Attention Layer)
training pipeline, now fully modality‑, dataset‑, and label‑agnostic.

Pipeline:
    Step‑A: Dataset (UnifiedMultimodalDataset or SyntheticULMTSSTDataset)
    Step‑B: Synchronizer (raw → aligned T)
    Step‑C: MultimodalFeatureExtractors (encoders → (B, T, 128))
    Step‑D: ModalityTokenizer + QSALFusion
    Step‑E: Transformer Encoder–Decoder
    Step‑F: Recognition + Forecasting Heads

Key Features:
    • Supports "recola", "sewa", and "ulm_tsst_synthetic"
    • Uses new modality flags from config (audio/video/ecg/eda/eeg/emg/resp/bpm/eye/text)
    • Keeps QSAL architecture, quantum circuit, and dimensions unchanged
    • Uses Synchronizer for alignment when needed
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
from datasets.ulm_tsst_synthetic import SyntheticULMTSSTDataset
from datasets.collate_fn import multimodal_collate_fn
from datasets.synchronizer import Synchronizer


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

    QSAL uses a quantum backend internally (CPU),
    but classical parts (encoders, transformer) can use CUDA.
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

    raise ValueError(f"Unsupported dataset name in QSAL training: {name}")


# ---------------------------------------------------------
# MODEL BUILDER (QSAL)
# ---------------------------------------------------------
def build_model(cfg) -> QSAL:
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
        synthetic=cfg["dataset"]["synthetic"],
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

    # 1. Load config
    cfg = load_config("configs/qsal_ulm_tsst.yaml")

    # 2. Device
    device = select_device(cfg["training"]["device"])
    print(f"[INFO] Using device: {device}")

    # 3. Run directory
    run_dir = make_run_folder(cfg["logging"]["run_dir"])
    ckpt_path = os.path.join(run_dir, "best_model.pt")
    log_path = os.path.join(run_dir, "training_log.json")

    print(f"[INFO] Run directory: {run_dir}")

    with open(os.path.join(run_dir, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=4)

    # 4. Datasets
    train_set = build_dataset(cfg, split="train")
    val_set = build_dataset(cfg, split="val")

    # Synchronizer (alignment to T)
    synchronizer = Synchronizer(
        target_seq_len=cfg["dataset"]["target_seq_len"],
        device="cpu",
    )

    def sync_collate_fn(batch_list):
        """
        Wrapper around multimodal_collate_fn that also applies Synchronizer
        to each sample before batching.
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

    # 5. Model
    model = build_model(cfg).to(device)

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
        log_batches=False,
    )

    # 8. Training loop
    history = []
    total_epochs = cfg["training"]["epochs"]

    print("\n[INFO] Starting QSAL training...\n")

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

    print("\n[INFO] QSAL training complete.")
    print(f"[INFO] Best model saved to: {ckpt_path}")
    print(f"[INFO] Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
