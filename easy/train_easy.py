"""
Stable Training Script for Hybrid Quantum Emotion Model
"""

import torch
from torch.utils.data import DataLoader

from utils.config import load_config
from datasets.synthetic import SyntheticMultimodalDataset
from models.wrappers.multimodal_qemotion_model import MultimodalQEmotionModel


def main():

    cfg = load_config("configs/base.yaml")

    # SAFE DEVICE
    device = torch.device("cpu")
    print("[INFO] Using CPU (stable quantum execution)")

    train_set = SyntheticMultimodalDataset(cfg, "train")
    val_set = SyntheticMultimodalDataset(cfg, "val")

    train_loader = DataLoader(train_set, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg["training"]["batch_size"])

    model = MultimodalQEmotionModel(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"]
    )

    loss_fn = torch.nn.MSELoss()

    for epoch in range(cfg["training"]["epochs"]):

        model.train()
        total_loss = 0

        for batch in train_loader:

            batch = {k: v.to(device) for k, v in batch.items()}

            pred = model(batch)

            target = batch["va_rec"]

            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "qgsal_model.pt")
    print("[INFO] Model saved")


if __name__ == "__main__":
    main()