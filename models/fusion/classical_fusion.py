"""
Classical Self-Attention Fusion (Baseline B1)

Implements the baseline fusion mechanism described in the thesis:
- Project each modality embedding (128-dim) to fusion dimension (64-dim)
- Perform multi-head self-attention over the 4 modality tokens at each frame
- Pool the attended modality tokens into a single fused token (64-dim)
- Project fused token to Transformer model dimension (256-dim)

Input:
    tokens: dict of modality embeddings
        audio: (B, T, 128)
        video: (B, T, 128)
        ecg:   (B, T, 128)
        eda:   (B, T, 128)

Output:
    fused: (B, T, 256)
"""

import torch
import torch.nn as nn


class ClassicalFusion(nn.Module):
    """
    Classical self-attention fusion over modality tokens.
    """

    def __init__(
        self,
        modality_dim=128,
        fusion_dim=64,
        num_modalities=4,
        num_heads=2,
        transformer_dim=256,
        dropout=0.1,
    ):
        """
        Parameters
        ----------
        modality_dim : int
            Input dimension of each modality embedding (default: 128).
        fusion_dim : int
            Dimension after projection for fusion (default: 64).
        num_modalities : int
            Number of modalities (default: 4).
        num_heads : int
            Number of attention heads for modality self-attention.
        transformer_dim : int
            Output dimension for Transformer input (default: 256).
        dropout : float
            Dropout applied after projections.
        """
        super().__init__()

        self.num_modalities = num_modalities
        self.fusion_dim = fusion_dim

        # 128 → 64 projection for each modality
        self.projections = nn.ModuleList([
            nn.Linear(modality_dim, fusion_dim) for _ in range(num_modalities)
        ])

        # Self-attention over modalities (sequence length = 4)
        self.attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 64 → 256 projection for Transformer
        self.to_transformer = nn.Linear(fusion_dim, transformer_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, audio, video, ecg, eda):
        """
        Parameters
        ----------
        audio, video, ecg, eda : Tensors (B, T, 128)

        Returns
        -------
        fused : Tensor (B, T, 256)
        """

        B, T, _ = audio.shape

        modalities = [audio, video, ecg, eda]
        proj_tokens = []

        # Project each modality: 128 → 64
        for i, x in enumerate(modalities):
            proj = self.projections[i](x)  # (B, T, 64)
            proj_tokens.append(proj)

        # Stack into (B, T, 4, 64)
        tokens = torch.stack(proj_tokens, dim=2)

        # Reshape for attention: (B*T, 4, 64)
        tokens = tokens.reshape(B * T, self.num_modalities, self.fusion_dim)

        # Self-attention over modalities
        attn_out, _ = self.attn(tokens, tokens, tokens)  # (B*T, 4, 64)

        # Mean pool modalities → (B*T, 64)
        fused = attn_out.mean(dim=1)

        # Reshape back to (B, T, 64)
        fused = fused.reshape(B, T, self.fusion_dim)

        # Project to Transformer dimension (256)
        fused = self.to_transformer(fused)
        fused = self.dropout(fused)

        return fused
