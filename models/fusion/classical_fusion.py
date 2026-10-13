"""
models.fusion.classical_fusion
=============================

Classical Token Fusion (Baseline B1)

Purpose
-------

Fuse multiple token streams into a single
temporal representation suitable for:

    • Continuous Emotion Recognition

    • Discrete Emotion Classification

    • Stress Recognition

    • Workload Recognition

    • Alertness / Drowsiness Recognition

    • Trajectory Forecasting

    • Classical Transformer Models

The mathematical formulation remains identical
to the original thesis baseline while the
interface is generalized for the unified
multimodal framework.

Design Philosophy
-----------------

This module operates on tokens rather than
modalities.

The fusion layer does NOT know whether a token
originates from:

    ECG
    EDA
    EEG
    EMG
    RESP
    BPM

    Eye Tracking

    Audio

    Video

    Text

    Continuous Label History

    Discrete Label History

    Future Task Tokens

    CLS Tokens

Any source capable of producing token embeddings
may participate in fusion.

Input
-----

tokens

    Shape:

        (B, T, N, D)

where

    B = Batch Size

    T = Temporal Length

    N = Number of Active Tokens

    D = Token Embedding Dimension

Default

    D = 128

Fusion Pipeline
---------------

1. Token Projection

        D
            ↓
        fusion_dim

2. Multi-Head Self Attention

        Across N tokens

3. Residual Connection

4. Layer Normalization

5. Token Pooling

        Mean Pooling

6. Transformer Projection

        fusion_dim
            ↓
        transformer_dim

Output
------

fused

    Shape:

        (B, T, transformer_dim)

Default

    transformer_dim = 256

Design Goals
------------

✓ Dataset Agnostic

✓ Label Agnostic

✓ Continuous Labels

✓ Discrete Labels

✓ Variable Token Count

✓ Recognition Compatible

✓ Forecasting Compatible

✓ Missing Token Compatible

✓ Future Proof

Used By
-------

Baseline B1

Potentially reusable for:

    B2

    B3

    QSAL Variants

    Future Hybrid Architectures
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassicalFusion(nn.Module):
    """
    Classical self-attention token fusion.

    Input:

        (B, T, N, D)

    Output:

        (B, T, transformer_dim)
    """

    def __init__(
        self,
        emb_dim: int = 128,
        fusion_dim: int = 64,
        transformer_dim: int = 256,
        num_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.emb_dim = emb_dim
        self.fusion_dim = fusion_dim
        self.transformer_dim = transformer_dim
        self.num_heads = num_heads

        # --------------------------------------------------
        # Input Projection
        #
        # D -> fusion_dim
        # --------------------------------------------------

        self.input_proj = nn.Linear(
            emb_dim,
            fusion_dim,
        )

        # --------------------------------------------------
        # Token Self-Attention
        #
        # Operates across token dimension N.
        # --------------------------------------------------

        self.attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        # --------------------------------------------------
        # Residual Normalization
        # --------------------------------------------------

        self.norm = nn.LayerNorm(
            fusion_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # --------------------------------------------------
        # Projection to Transformer Space
        #
        # fusion_dim -> transformer_dim
        # --------------------------------------------------

        self.to_transformer = nn.Linear(
            fusion_dim,
            transformer_dim,
        )

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : Tensor

            Shape:

                (B,T,N,D)

        Returns
        -------
        Tensor

            Shape:

                (B,T,transformer_dim)
        """

        if tokens.ndim != 4:
            raise ValueError(
                "Expected token tensor with shape "
                "(B,T,N,D), received "
                f"{tuple(tokens.shape)}"
            )

        B, T, N, D = tokens.shape

        # ----------------------------------------------
        # Merge Batch and Time
        #
        # (B,T,N,D)
        #
        # ->
        #
        # (B*T,N,D)
        # ----------------------------------------------

        x = tokens.reshape(
            B * T,
            N,
            D,
        )

        # ----------------------------------------------
        # Project Tokens
        #
        # D -> fusion_dim
        # ----------------------------------------------

        x = self.input_proj(
            x
        )

        # ----------------------------------------------
        # Token Self-Attention
        # ----------------------------------------------

        attn_out, _ = self.attn(
            x,
            x,
            x,
        )

        # ----------------------------------------------
        # Residual + LayerNorm
        # ----------------------------------------------

        x = self.norm(
            x +
            self.dropout(
                attn_out
            )
        )

        # ----------------------------------------------
        # Token Pooling
        #
        # (B*T,N,F)
        #
        # ->
        #
        # (B*T,F)
        # ----------------------------------------------

        fused = x.mean(
            dim=1
        )

        # ----------------------------------------------
        # Transformer Projection
        # ----------------------------------------------

        fused = self.to_transformer(
            fused
        )

        fused = self.dropout(
            fused
        )

        # ----------------------------------------------
        # Restore Temporal Structure
        # ----------------------------------------------

        fused = fused.view(
            B,
            T,
            self.transformer_dim,
        )

        return fused

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

    def summary(
        self,
    ) -> dict:
        """
        Return fusion metadata.
        """

        return {
            "fusion_type":
                "classical_attention",

            "token_dim":
                self.emb_dim,

            "fusion_dim":
                self.fusion_dim,

            "transformer_dim":
                self.transformer_dim,

            "num_heads":
                self.num_heads,
        }

    def extra_repr(
        self,
    ) -> str:

        return (
            f"emb_dim={self.emb_dim}, "
            f"fusion_dim={self.fusion_dim}, "
            f"transformer_dim={self.transformer_dim}, "
            f"num_heads={self.num_heads}"
        )