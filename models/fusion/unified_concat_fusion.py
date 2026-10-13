"""
models.fusion.unified_concat_fusion
===================================

Baseline B2

Projection + Concatenation Fusion

No attention.
No token interaction.
No quantum processing.

The Transformer itself must learn
cross-modal relationships.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UnifiedConcatFusion(
    nn.Module,
):
    """
    Baseline B2.

    Projection + Concatenation Fusion.

    Input
    -----

        (B,T,N,D)

    Output
    ------

        (B,T,transformer_dim)
    """

    def __init__(
        self,
        emb_dim: int = 128,
        proj_dim: int = 64,
        transformer_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.emb_dim = emb_dim

        self.proj_dim = proj_dim

        self.transformer_dim = (
            transformer_dim
        )

        # ----------------------------------
        # Shared Projection
        #
        # 128 → 64
        # ----------------------------------

        self.token_projection = (
            nn.Linear(
                emb_dim,
                proj_dim,
            )
        )

        # ----------------------------------
        # Lazy Alignment
        #
        # N*64 → 256
        # ----------------------------------

        self.to_transformer = None

        self.dropout = (
            nn.Dropout(
                dropout
            )
        )

    # ======================================
    # Utilities
    # ======================================

    def _build_projection(
        self,
        concat_dim: int,
        device: torch.device,
    ):

        self.to_transformer = (
            nn.Linear(
                concat_dim,
                self.transformer_dim,
            ).to(device)
        )

    # ======================================
    # Forward
    # ======================================

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:

        if tokens.ndim != 4:

            raise ValueError(

                "Expected token tensor "

                "(B,T,N,D), received "

                f"{tuple(tokens.shape)}"
            )

        B, T, N, D = tokens.shape

        # ----------------------------------
        # Token Projection
        #
        # D → proj_dim
        # ----------------------------------

        projected = (
            self.token_projection(
                tokens
            )
        )

        # ----------------------------------
        # Concatenate Tokens
        #
        # (B,T,N,64)
        #
        # →
        #
        # (B,T,N*64)
        # ----------------------------------

        fused = projected.reshape(

            B,

            T,

            N * self.proj_dim,
        )

        concat_dim = (
            N
            *
            self.proj_dim
        )

        # ----------------------------------
        # Build Projection Lazily
        # ----------------------------------

        if (
            self.to_transformer
            is None
        ):

            self._build_projection(

                concat_dim,

                fused.device,
            )

        # ----------------------------------
        # Capacity Matching
        #
        # N*64 → 256
        # ----------------------------------

        fused = (
            self.to_transformer(
                fused
            )
        )

        fused = (
            self.dropout(
                fused
            )
        )

        return fused

    # ======================================
    # Metadata
    # ======================================

    def summary(
        self,
    ):

        return {

            "fusion_type":
                "concat",

            "projection_dim":
                self.proj_dim,

            "transformer_dim":
                self.transformer_dim,
        }

    def extra_repr(
        self,
    ) -> str:

        return (

            f"emb_dim={self.emb_dim}, "

            f"proj_dim={self.proj_dim}, "

            f"transformer_dim={self.transformer_dim}"
        )