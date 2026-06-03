"""
Modality Tokenizer Module

This module converts raw modality embeddings (audio, video, ECG, EDA)
into modality-specific tokens suitable for fusion and QSAL processing.

Each modality:
    - Starts as a 128-dimensional embedding (from encoders)
    - Passes through a modality-specific linear projection
    - Receives a learned modality/type embedding
    - Produces a final 128-dimensional token

Output shape:
    tokens: (B, T, M, 128)
        B = batch size
        T = sequence length (history window)
        M = number of modalities (4: audio, video, ECG, EDA)
"""

import torch
import torch.nn as nn


class ModalityTokenizer(nn.Module):
    """
    Tokenizes each modality into a unified 128-dimensional token space.

    This module is the bridge between modality-specific encoders
    (wav2vec2, ResNet18, 1D CNNs) and the fusion/QSAL block.

    Responsibilities:
        - Apply a learnable linear projection per modality
        - Inject a learned modality/type embedding
        - Preserve temporal structure (B, T, ·)
        - Stack modalities along a dedicated modality axis (M)
    """

    def __init__(self, modality_dim: int = 128,
                 num_modalities: int = 4,
                 dropout: float = 0.1) -> None:
        """
        Parameters
        ----------
        modality_dim : int
            Dimensionality of each modality embedding (default: 128).
        num_modalities : int
            Number of modalities (audio, video, ECG, EDA).
        dropout : float
            Dropout applied after projection and type embedding.
        """
        super().__init__()

        self.modality_dim = modality_dim
        self.num_modalities = num_modalities

        # ------------------------------------------------------------------
        # 1. Linear projection for each modality
        #    - Each modality has its own learnable projection:
        #          R^{128} → R^{128}
        #    - This allows slight modality-specific adaptation while
        #      keeping a common dimensionality.
        # ------------------------------------------------------------------
        self.projections = nn.ModuleList([
            nn.Linear(modality_dim, modality_dim)
            for _ in range(num_modalities)
        ])

        # ------------------------------------------------------------------
        # 2. Modality type embeddings (learned)
        #    - Shape: (M, 128)
        #    - Each modality gets a learned embedding vector that acts
        #      as a "type token", similar to segment/type embeddings
        #      in BERT.
        # ------------------------------------------------------------------
        self.modality_embeddings = nn.Parameter(
            torch.randn(num_modalities, modality_dim)
        )

        # ------------------------------------------------------------------
        # 3. Dropout for regularization
        # ------------------------------------------------------------------
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                audio: torch.Tensor,
                video: torch.Tensor,
                ecg: torch.Tensor,
                eda: torch.Tensor) -> torch.Tensor:
        """
        Tokenize all modalities into a unified token tensor.

        Parameters
        ----------
        audio : Tensor
            Shape (B, T, 128) — audio embeddings.
        video : Tensor
            Shape (B, T, 128) — video embeddings.
        ecg : Tensor
            Shape (B, T, 128) — ECG embeddings.
        eda : Tensor
            Shape (B, T, 128) — EDA embeddings.

        Returns
        -------
        tokens : Tensor
            Shape (B, T, M, 128), where:
                B = batch size
                T = sequence length
                M = number of modalities (4)
            This tensor is ready to be consumed by the fusion/QSAL module.
        """
        # List of modality tensors in fixed order
        modalities = [audio, video, ecg, eda]
        token_list = []

        # ------------------------------------------------------------------
        # Process each modality independently:
        #   1. Apply modality-specific linear projection
        #   2. Add learned modality/type embedding
        #   3. Apply dropout
        # ------------------------------------------------------------------
        for i, x in enumerate(modalities):
            # x: (B, T, 128)
            proj = self.projections[i](x)  # (B, T, 128)

            # Modality/type embedding:
            #   self.modality_embeddings[i] has shape (128,)
            #   We reshape to (1, 1, 128) so it broadcasts correctly
            #   over both batch (B) and time (T) dimensions.
            type_emb = self.modality_embeddings[i].view(1, 1, -1)

            # Add type embedding → still (B, T, 128)
            proj = proj + type_emb

            # Regularization
            proj = self.dropout(proj)

            token_list.append(proj)

        # ------------------------------------------------------------------
        # Stack along a new modality axis:
        #   token_list: list of 4 tensors, each (B, T, 128)
        #   tokens: (B, T, M, 128)
        # ------------------------------------------------------------------
        tokens = torch.stack(token_list, dim=2)

        return tokens
