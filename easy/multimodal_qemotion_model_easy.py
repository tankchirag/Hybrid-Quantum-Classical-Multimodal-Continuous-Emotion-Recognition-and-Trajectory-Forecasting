import torch
import torch.nn as nn
from quantum.qsal_layer import QGSALLayer


class MultimodalQEmotionModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        hidden = cfg["model"]["hidden_dim"]

        # FIX: correct input size = 4 modalities × 128 = 512
        self.input_proj = nn.Linear(512, hidden)

        self.qgsal = QGSALLayer(
            dim=hidden,
            n_heads=2,
            n_qubits=4
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, batch):

        x = torch.cat([
            batch["audio"],
            batch["video"],
            batch["ecg"],
            batch["eda"]
        ], dim=-1)   # (B, T, 512)

        x = x.mean(dim=1)  # (B, 512)

        x = self.input_proj(x)  # (B, hidden)

        x = self.qgsal(x.unsqueeze(1))  # (B, 1, hidden)

        x = x.squeeze(1)

        return self.regressor(x)