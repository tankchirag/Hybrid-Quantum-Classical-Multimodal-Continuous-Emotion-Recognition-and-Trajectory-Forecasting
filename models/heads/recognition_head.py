"""
Recognition Head

Maps encoder outputs (B, T, 256) → (B, T, 2)
Predicts valence–arousal at each time step.
"""

import torch.nn as nn


class RecognitionHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=2):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        x: (B, T, 256)
        returns: (B, T, 2)
        """
        return self.mlp(x)
