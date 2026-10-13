"""
Classical Self-Attention Fusion (Baseline B1)

Implements the baseline fusion mechanism described in the thesis:
- Input: modality tokens of shape (B, T, M, D)
    * B = batch size
    * T = temporal length (history window)
    * M = number of modalities (e.g., 4 or 5)
    * D = embedding dimension (e.g., 128)

Fusion pipeline:
    1. Project each modality token from D → fusion_dim (e.g., 64)
    2. Perform multi-head self-attention over the M modality tokens at each frame
    3. Pool the attended modality tokens into a single fused token (fusion_dim)
    4. Project fused token to Transformer model dimension (transformer_dim, e.g., 256)

Output:
    fused: (B, T, transformer_dim)

This module is used by:
    - Baseline B1 (classical fusion)
    - Can be reused by other baselines / QSAL variants that expect
      a fused temporal representation per frame.
"""

import torch
import torch.nn as nn


class ClassicalFusion(nn.Module):
    """
    Classical self-attention fusion over modality tokens.

    Input:
        tokens: (B, T, M, D)

    Output:
        fused: (B, T, transformer_dim)
    """

    def __init__(
        self,
        emb_dim: int = 128,
        fusion_dim: int = 64,
        transformer_dim: int = 256,
        num_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        emb_dim : int
            Input dimension of each modality token (default: 128).
        fusion_dim : int
            Intermediate fusion dimension after projection (default: 64).
        transformer_dim : int
            Output dimension for the temporal Transformer backbone (default: 256).
        num_heads : int
            Number of attention heads for modality self-attention.
        dropout : float
            Dropout applied after attention and projections.
        """
        super().__init__()

        self.emb_dim = emb_dim
        self.fusion_dim = fusion_dim
        self.transformer_dim = transformer_dim
        self.num_heads = num_heads

        # ------------------------------------------------------------------
        # 1. Linear projection for modality tokens
        #    - Shared across modalities:
        #          R^{emb_dim} → R^{fusion_dim}
        #    - This brings all modality tokens into a common fusion space.
        # ------------------------------------------------------------------
        self.input_proj = nn.Linear(emb_dim, fusion_dim)

        # ------------------------------------------------------------------
        # 2. Multi-head self-attention over modalities
        #    - Sequence length = M (number of modalities)
        #    - Operates independently for each (B, T) pair.
        # ------------------------------------------------------------------
        self.attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        # ------------------------------------------------------------------
        # 3. Normalization and dropout
        # ------------------------------------------------------------------
        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

        # ------------------------------------------------------------------
        # 4. Projection to Transformer model dimension
        #    - Fused token: fusion_dim → transformer_dim
        # ------------------------------------------------------------------
        self.to_transformer = nn.Linear(fusion_dim, transformer_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : Tensor
            Shape (B, T, M, D)
                B = batch size
                T = temporal length
                M = number of modalities
                D = emb_dim

        Returns
        -------
        fused : Tensor
            Shape (B, T, transformer_dim)
        """
        B, T, M, D = tokens.shape

        # ------------------------------------------------------------------
        # 1. Merge batch and time for modality-wise attention
        #    (B, T, M, D) → (B*T, M, D)
        # ------------------------------------------------------------------
        x = tokens.view(B * T, M, D)

        # ------------------------------------------------------------------
        # 2. Project to fusion_dim
        #    (B*T, M, D) → (B*T, M, fusion_dim)
        # ------------------------------------------------------------------
        x = self.input_proj(x)

        # ------------------------------------------------------------------
        # 3. Self-attention over modalities
        #    - Query, key, value are all modality tokens
        #    - Output: (B*T, M, fusion_dim)
        # ------------------------------------------------------------------
        attn_out, _ = self.attn(x, x, x)

        # Residual + normalization
        x = self.norm(x + self.dropout(attn_out))  # (B*T, M, fusion_dim)

        # ------------------------------------------------------------------
        # 4. Pool over modality dimension M
        #    - Mean pooling: (B*T, M, fusion_dim) → (B*T, fusion_dim)
        # ------------------------------------------------------------------
        fused = x.mean(dim=1)  # (B*T, fusion_dim)

        # ------------------------------------------------------------------
        # 5. Project to Transformer dimension
        #    (B*T, fusion_dim) → (B*T, transformer_dim)
        # ------------------------------------------------------------------
        fused = self.to_transformer(fused)
        fused = self.dropout(fused)

        # ------------------------------------------------------------------
        # 6. Reshape back to (B, T, transformer_dim)
        # ------------------------------------------------------------------
        fused = fused.view(B, T, self.transformer_dim)

        return fused
