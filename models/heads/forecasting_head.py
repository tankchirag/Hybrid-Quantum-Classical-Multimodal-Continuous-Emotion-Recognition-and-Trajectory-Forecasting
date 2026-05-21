"""
Forecasting Head

Maps decoder outputs (B, S, 256) → (B, S, 2)
Predicts future valence–arousal trajectory.
"""

import torch.nn as nn


class ForecastingHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=2):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        x: (B, S, 256)
        returns: (B, S, 2)
        """
        return self.mlp(x)
