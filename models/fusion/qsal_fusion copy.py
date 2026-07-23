"""
QSAL Fusion Module (Quantum Self-Attention Fusion)

This module performs:
- 128 → 64 projection per modality
- Reshape into (B, T, M, H, head_dim)
- Apply QSALLayer (quantum multi-head attention)
- Concatenate heads → (B, T, 64)
- Project to Transformer dimension (256)

Input:
    audio, video, ecg, eda: (B, T, 128)

Output:
    fused: (B, T, 256)
"""

import torch
import torch.nn as nn
from quantum.qsal_layer import QSALLayer


class QSALFusion(nn.Module):
    """
    Quantum Self-Attention Fusion (QSAL)
    """

    def __init__(
        self,
        modality_dim=128,
        fusion_dim=64,
        num_modalities=4,
        num_heads=2,
        num_qubits=4,
        circuit_depth=2,
        transformer_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.num_modalities = num_modalities
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.head_dim = fusion_dim // num_heads

        # -----------------------------
        # 1. Classical projection 128 → 64
        # -----------------------------
        self.projections = nn.ModuleList([
            nn.Linear(modality_dim, fusion_dim) for _ in range(num_modalities)
        ])

        # -----------------------------
        # 2. Quantum attention layer
        # -----------------------------
        self.qsal = QSALLayer(
            num_heads=num_heads,
            head_dim=self.head_dim,
            num_modalities=num_modalities,
            num_qubits=num_qubits,
            circuit_depth=circuit_depth,
        )

        # -----------------------------
        # 3. Output projection 64 → 256
        # -----------------------------
        self.to_transformer = nn.Linear(fusion_dim, transformer_dim)
        self.dropout = nn.Dropout(dropout)

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------
    def forward(self, audio, video, ecg, eda):
        B, T, _ = audio.shape

        # -----------------------------
        # 1. Classical projection 128 → 64
        # -----------------------------
        modalities = [audio, video, ecg, eda]
        proj_tokens = []

        for i, x in enumerate(modalities):
            proj = self.projections[i](x)  # (B, T, 64)
            proj_tokens.append(proj)

        # Stack → (B, T, 4, 64)
        tokens = torch.stack(proj_tokens, dim=2)

        # -----------------------------
        # 2. Reshape into heads
        # -----------------------------
        # (B, T, 4, 2, 32)
        tokens = tokens.reshape(
            B, T, self.num_modalities, self.num_heads, self.head_dim
        )

        # Merge heads into last dimension for QSALLayer:
        # (B, T, 4, 64)
        tokens = tokens.reshape(B, T, self.num_modalities, self.fusion_dim)

        # -----------------------------
        # 3. Apply QSAL Layer
        # -----------------------------
        # QSALLayer expects (B, T, M, head_dim_per_head)
        # So reshape back into heads:
        tokens = tokens.reshape(B, T, self.num_modalities, self.num_heads, self.head_dim)

        # QSALLayer output: (B, T, num_heads * num_qubits)
        fused_q = self.qsal(tokens)

        # fused_q shape = (B, T, 2 * 4) = (B, T, 8)
        # But we want (B, T, 64), so we expand:
        fused = fused_q.repeat(1, 1, self.fusion_dim // fused_q.shape[-1])

        # -----------------------------
        # 4. Project to Transformer dim (256)
        # -----------------------------
        fused = self.to_transformer(fused)
        fused = self.dropout(fused)

        return fused
