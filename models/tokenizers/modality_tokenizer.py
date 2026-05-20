"""
Modality Tokenizer Module

This module converts raw modality embeddings (audio, video, ECG, EDA)
into modality-specific tokens suitable for fusion and QSAL processing.

Each modality:
    - Starts as a 128-dimensional embedding (from encoders)
    - Passes through a modality-specific linear projection
    - Receives a learned modality embedding (type embedding)
    - Produces a final 128-dimensional token

Output shape:
    (B, T, M, 128)
Where:
    B = batch size
    T = sequence length
    M = number of modalities (4)
"""

import torch
import torch.nn as nn


class ModalityTokenizer(nn.Module):
    """
    Tokenizes each modality into a unified 128-dimensional token space.
    """

    def __init__(self, modality_dim=128, num_modalities=4, dropout=0.1):
        """
        Parameters
        ----------
        modality_dim : int
            Dimensionality of each modality embedding (default: 128).
        num_modalities : int
            Number of modalities (audio, video, ECG, EDA).
        dropout : float
            Dropout applied after projection.
        """
        super().__init__()

        self.modality_dim = modality_dim
        self.num_modalities = num_modalities

        # Linear projection for each modality
        self.projections = nn.ModuleList([
            nn.Linear(modality_dim, modality_dim) for _ in range(num_modalities)
        ])

        # Modality type embeddings (learned)
        self.modality_embeddings = nn.Parameter(
            torch.randn(num_modalities, modality_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, audio, video, ecg, eda):
        """
        Tokenize all modalities.

        Parameters
        ----------
        audio : Tensor (B, T, 128)
        video : Tensor (B, T, 128)
        ecg   : Tensor (B, T, 128)
        eda   : Tensor (B, T, 128)

        Returns
        -------
        tokens : Tensor (B, T, 4, 128)
            Tokenized modalities ready for fusion/QSAL.
        """

        modalities = [audio, video, ecg, eda]
        token_list = []

        for i, x in enumerate(modalities):
            # Linear projection
            proj = self.projections[i](x)  # (B, T, 128)

            # Add modality embedding (broadcast over T)
            proj = proj + self.modality_embeddings[i]

            # Dropout
            proj = self.dropout(proj)

            token_list.append(proj)

        # Stack into (B, T, M, 128)
        tokens = torch.stack(token_list, dim=2)
        return tokens
