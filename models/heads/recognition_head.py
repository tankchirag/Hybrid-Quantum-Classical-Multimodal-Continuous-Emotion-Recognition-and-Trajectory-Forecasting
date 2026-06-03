"""
Recognition Head

Maps encoder outputs (B, T, 256) → (B, T, 2)
Predicts continuous valence–arousal at each time step.

This head is used by:
    - Baseline B1
    - Baseline B2 (QSAL)
    - Baseline B3 (Hybrid)
    - Any model requiring frame-level VA regression
"""

import torch.nn as nn


class RecognitionHead(nn.Module):
    """
    Lightweight MLP head for continuous VA regression.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 2,
        dropout: float = 0.1,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of encoder output (default: 256).
        hidden_dim : int
            Hidden layer size (default: 128).
        output_dim : int
            Output dimension (valence, arousal) = 2.
        dropout : float
            Dropout applied before final regression.
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            Shape (B, T, 256)

        Returns
        -------
        Tensor
            Shape (B, T, 2)
        """
        return self.mlp(x)
