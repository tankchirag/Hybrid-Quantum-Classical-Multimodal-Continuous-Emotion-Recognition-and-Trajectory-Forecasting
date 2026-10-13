"""
models.fusion.unified_concat_fusion
===================================

Unified Concatenation Fusion

Baseline B2

Purpose
-------

Implements the explicit Baseline B2 strategy
defined in the thesis.

Unlike Baseline B1:

    Token Attention Fusion

and unlike:

    PQC Fusion
    QSAL Fusion

Baseline B2 performs:

    Per-Source Projection
        +
    Direct Concatenation

No attention-based fusion is used.

No token interactions are modeled.

No quantum processing is used.

Scientific Purpose
------------------

Baseline B2 is designed to answer:

    Can a sufficiently expressive Transformer
    learn cross-source interactions without
    an explicit fusion mechanism?

This baseline therefore serves as a
capacity-matched control condition.

Design Goals
------------

✓ Dataset Agnostic

✓ Label Agnostic

✓ Continuous Labels

✓ Discrete Labels

✓ Recognition Compatible

✓ Forecasting Compatible

✓ Variable Source Count

✓ Future Proof

Input
-----

embeddings

{
    source_name:

        (B,T,128)
}

Examples

    ecg
    eda
    eeg
    emg
    resp
    bpm

    eye

    audio
    video
    text

Future Sources

    continuous_labels

    discrete_labels

Fusion Pipeline
---------------

Each source embedding:

    128

        ↓

     64

via a dedicated source-specific projection.

All projected sources are concatenated:

    K × 64

where

    K = number of active sources

The concatenated representation is then
mapped into the shared Transformer space:

    K×64
        ↓
      256

This guarantees that all baselines operate
with the same Transformer dimensionality.

Output
------

fused

    (B,T,256)

Transformer Compatibility
-------------------------

Output dimensionality remains fixed:

    256

regardless of the number of active sources.

This preserves fairness across:

    B1
    B2
    B3
    QSAL
"""

from __future__ import annotations

from typing import Dict
from typing import List

import torch
import torch.nn as nn


class UnifiedConcatFusion(
    nn.Module,
):
    """
    Baseline B2.

    Projection + Concatenation Fusion.
    """

    DEFAULT_SOURCE_ORDER = [

        # Physiological

        "ecg",
        "eda",
        "eeg",
        "emg",
        "resp",
        "bpm",

        # Behavioral

        "eye",
        "audio",
        "video",
        "text",

        # Future

        "continuous_labels",
        "discrete_labels",
    ]

    def __init__(
        self,
        emb_dim: int = 128,
        proj_dim: int = 64,
        transformer_dim: int = 256,
        source_order: List[str] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.transformer_dim = transformer_dim

        self.source_order = (
            source_order
            or self.DEFAULT_SOURCE_ORDER
        )

        # ------------------------------------
        # Per-Source Projection
        #
        # 128 -> 64
        # ------------------------------------

        self.projections = nn.ModuleDict(
            {
                source_name:
                nn.Linear(
                    emb_dim,
                    proj_dim,
                )
                for source_name
                in self.source_order
            }
        )

        # ------------------------------------
        # Final alignment projection
        #
        # K*64 -> 256
        #
        # created lazily because
        # K depends on active sources
        # ------------------------------------

        self.output_projection = None

        self.dropout = nn.Dropout(
            dropout
        )

    # ----------------------------------------
    # Utilities
    # ----------------------------------------

    def _build_output_projection(
        self,
        input_dim: int,
        device: torch.device,
    ) -> None:
        """
        Create output projection when
        active source count becomes known.
        """

        self.output_projection = nn.Linear(
            input_dim,
            self.transformer_dim,
        ).to(device)

    # ----------------------------------------
    # Forward
    # ----------------------------------------

    def forward(
        self,
        embeddings: Dict[
            str,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings

            {
                source_name:
                    (B,T,128)
            }

        Returns
        -------
        Tensor

            (B,T,256)
        """

        projected_sources = []

        active_sources = []

        for source_name in self.source_order:

            x = embeddings.get(
                source_name,
                None,
            )

            if x is None:
                continue

            x = self.projections[
                source_name
            ](x)

            projected_sources.append(
                x
            )

            active_sources.append(
                source_name
            )

        if len(projected_sources) == 0:

            raise RuntimeError(
                "UnifiedConcatFusion received "
                "no active source embeddings."
            )

        # ------------------------------------
        # Concatenation
        #
        # (B,T,K*64)
        # ------------------------------------

        fused = torch.cat(
            projected_sources,
            dim=-1,
        )

        B, T, concat_dim = fused.shape

        # ------------------------------------
        # Build projection lazily
        # ------------------------------------

        if (
            self.output_projection
            is None
        ):

            self._build_output_projection(
                concat_dim,
                fused.device,
            )

        # ------------------------------------
        # Capacity Matching
        #
        # K*64 -> 256
        # ------------------------------------

        fused = self.output_projection(
            fused
        )

        fused = self.dropout(
            fused
        )

        return fused

    # ----------------------------------------
    # Metadata
    # ----------------------------------------

    def summary(
        self,
    ) -> dict:

        return {

            "fusion_type":
                "concat",

            "projection_dim":
                self.proj_dim,

            "transformer_dim":
                self.transformer_dim,

            "supported_sources":
                self.source_order,
        }

    def extra_repr(
        self,
    ) -> str:

        return (
            f"emb_dim={self.emb_dim}, "
            f"proj_dim={self.proj_dim}, "
            f"transformer_dim={self.transformer_dim}"
        )