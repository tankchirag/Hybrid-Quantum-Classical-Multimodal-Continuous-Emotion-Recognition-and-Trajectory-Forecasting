"""
Training script for Hybrid Quantum-Classical Multimodal Emotion Recognition Model.

This script:
    - Loads dataset
    - Initializes multimodal quantum model
    - Selects device based on config (cuda / cpu / auto)
    - Runs training loop via Trainer class
"""

import torch
from torch.utils.data import DataLoader

from utils.config import load_config
from datasets.synthetic import SyntheticMultimodalDataset
from models.wrappers.multimodal_qemotion_model import MultimodalQEmotionModel
from training.trainer import Trainer


def main():
    """
    Main training pipeline.
    """

    # ---------------------------------------------------------
    # LOAD CONFIG
    # ---------------------------------------------------------
    cfg = load_config("configs/base.yaml")

    # ---------------------------------------------------------
    # DEVICE SELECTION (CONFIG-CONTROLLED)
    # ---------------------------------------------------------
    cfg_device = cfg["training"].get("device", "auto")

    if cfg_device == "cuda":
        device = torch.device("cuda")
    elif cfg_device == "cpu":
        device = torch.device("cpu")
    else:  # auto
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")

    # ---------------------------------------------------------
    # DATASETS
    # ---------------------------------------------------------
    train_set = SyntheticMultimodalDataset(cfg, split="train")
    val_set = SyntheticMultimodalDataset(cfg, split="val")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg["training"]["batch_size"]
    )

    # ---------------------------------------------------------
    # MODEL (GPU ENABLED IF AVAILABLE)
    # ---------------------------------------------------------
    model = MultimodalQEmotionModel(cfg).to(device)

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
        patience=5,
    )

    # ---------------------------------------------------------
    # TRAIN LOOP
    # ---------------------------------------------------------
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["training"]["epochs"],
    )

    print("Training complete. Best model saved as best_model.pt")


if __name__ == "__main__":
    main()
