"""
QSAL Fusion Layer
-----------------

Quantum Self-Attention fusion block for multimodal embeddings.

Input (from wrapper):
    emb_dict: dict of modality embeddings
        {
            "audio": (B, T, D) or None,
            "video": (B, T, D) or None,
            "ecg":   (B, T, D) or None,
            "eda":   (B, T, D) or None,
            "text":  (B, T, D) or None,
        }

Internal representation:
    x: (B, T, M, D_in)
        B: batch size
        T: time steps
        M: number of active modalities
        D_in: per-modality embedding dimension

Output:
    fused sequence of shape (B, T, model_dim),
    ready for the temporal Transformer.

IMPORTANT:
    Quantum circuit outputs are ALWAYS float64 (Double).
    Linear layers expect float32 (Float).
    Therefore, PQC outputs MUST be cast to float32 before projection.
"""

from __future__ import annotations
from typing import List, Optional, Dict

import torch
import torch.nn as nn

from quantum.qsal_pqc import QSALPQC
from models.attention.gaussian_attention import GaussianAttention


class QSALFusion(nn.Module):
    """
    Quantum Self-Attention Fusion Layer.

    Parameters
    ----------
    emb_dim : int
        Per-modality embedding dimension D_in.
    model_dim : int
        Target model dimension for the temporal Transformer (e.g., 256).
    fused_dim : int
        Intermediate fused dimension before projection to model_dim.
    modalities : List[str]
        List of active modalities, e.g. ["audio", "video", "ecg", "eda"].
    init_sigma : float
        Initial value for GaussianAttention sigma.
    learnable_sigma : bool
        Whether sigma is trainable.
    """

    def __init__(
        self,
        emb_dim: int,
        model_dim: int = 256,
        fused_dim: int = 64,
        modalities: Optional[List[str]] = None,
        init_sigma: float = 1.0,
        learnable_sigma: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = emb_dim
        self.model_dim = model_dim
        self.fused_dim = fused_dim
        self.modalities = modalities or ["audio", "video", "ecg", "eda"]

        # PQC input/output dimensions
        self.pqc_input_dim = 4
        self.pqc_output_dim = 4

        # ------------------------------------------------------------------
        #  Linear projections
        # ------------------------------------------------------------------
        self.to_pqc_input = nn.Linear(self.input_dim, self.pqc_input_dim)
        self.pqc_out_to_fused = nn.Linear(self.pqc_output_dim, self.fused_dim)
        self.fused_to_model = nn.Linear(self.fused_dim, self.model_dim)

        # ------------------------------------------------------------------
        #  Quantum circuits (Q, K, V)
        # ------------------------------------------------------------------
        self.q_pqc = QSALPQC(n_qubits=4, depth=2)
        self.k_pqc = QSALPQC(n_qubits=4, depth=2)
        self.v_pqc = QSALPQC(n_qubits=4, depth=2)

        # ------------------------------------------------------------------
        #  Gaussian attention over modalities
        # ------------------------------------------------------------------
        self.attention = GaussianAttention(
            init_sigma=init_sigma,
            learnable_sigma=learnable_sigma,
        )

    # ----------------------------------------------------------------------
    #  Stack modality embeddings → (B, T, M, D_in)
    # ----------------------------------------------------------------------
    def _stack_modalities(self, emb_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        tensors = []
        for m in self.modalities:
            if m in emb_dict and emb_dict[m] is not None:
                tensors.append(emb_dict[m])

        if len(tensors) == 0:
            raise ValueError("QSALFusion received no active modality embeddings.")

        return torch.stack(tensors, dim=2)  # (B, T, M, D_in)

    # ----------------------------------------------------------------------
    #  Forward pass
    # ----------------------------------------------------------------------
    def forward(
        self,
        emb_dict: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # 1) Stack modalities → (B, T, M, D_in)
        x = self._stack_modalities(emb_dict)

        if x.dim() != 4:
            raise ValueError(
                "QSALFusion expects stacked input of shape (B, T, M, D_in). "
                f"Got shape {x.shape}."
            )

        B, T, M, D_in = x.shape
        if D_in != self.input_dim:
            raise ValueError(
                f"QSALFusion input_dim={self.input_dim}, but got D_in={D_in}."
            )

        # 2) Project modality tokens → PQC input (B, T, M, 4)
        x_proj = self.to_pqc_input(x)  # (B, T, M, 4)
        x_flat = x_proj.reshape(B * T * M, self.pqc_input_dim)  # (B*T*M, 4)

        # 3) Apply Q/K/V PQCs (these return float64)
        q_flat = self.q_pqc(x_flat)  # (B*T*M, 4)
        k_flat = self.k_pqc(x_flat)  # (B*T*M, 4)
        v_flat = self.v_pqc(x_flat)  # (B*T*M, 4)

        # 4) Reshape PQC outputs
        q = q_flat.view(B * T, M, self.pqc_output_dim)
        k = k_flat.view(B * T, M, self.pqc_output_dim)
        v = v_flat.view(B * T, M, self.pqc_output_dim)

        # 5) Gaussian attention over modalities
        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                mask_bt = mask.unsqueeze(1).expand(B, T, M).reshape(B * T, M)
                attn_mask = mask_bt
            elif mask.dim() == 3:
                attn_mask = mask.reshape(B * T, M)
            else:
                raise ValueError(
                    f"Unsupported mask shape {mask.shape}. "
                    "Expected (B, M) or (B, T, M)."
                )

        out_attn = self.attention(q=q, k=k, v=v, mask=attn_mask)  # (B*T, M, 4)

        # 6) Pool over modalities → fused PQC output (B*T, 4)
        fused_pqc = out_attn.mean(dim=1)

        # ------------------------------------------------------------------
        #  CRITICAL FIX: PQC outputs are float64 → cast to float32
        # ------------------------------------------------------------------
        fused_pqc = fused_pqc.to(torch.float32)

        # 7) Project fused PQC output → fused_dim → model_dim
        fused = self.pqc_out_to_fused(fused_pqc)      # (B*T, fused_dim)
        fused_model = self.fused_to_model(fused)      # (B*T, model_dim)

        # 8) Reshape back to (B, T, model_dim)
        fused_model = fused_model.view(B, T, self.model_dim)

        return fused_model
