"""
models.history.injection
========================

History Injection Module

Purpose
-------
Inject compressed history-state information
into fused multimodal representations.

Design
------

Fusion Output

    (B,T,D)

History State

    (B,H)

↓

Broadcast

↓

(B,T,H)

↓

Concatenate

↓

(B,T,D+H)

↓

Projection

↓

(B,T,D)

Supports
--------

✓ Dataset Agnostic

✓ Modality Agnostic

✓ Label Agnostic

✓ B1 Compatible

✓ B2 Compatible

✓ B3 Compatible

✓ QSAL Compatible

✓ Recognition Compatible

✓ Forecasting Compatible
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HistoryInjection(
    nn.Module,
):
    """
    Post-fusion history conditioning.
    """

    def __init__(
        self,
        model_dim: int = 256,
        history_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.model_dim = model_dim
        self.history_dim = history_dim

        self.projection = nn.Sequential(

            nn.Linear(
                model_dim +
                history_dim,

                model_dim,
            ),

            nn.LayerNorm(
                model_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout,
            ),
        )

    # ---------------------------------------
    # Forward
    # ---------------------------------------

    def forward(
        self,
        fused: torch.Tensor,
        history_state:
        torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------

        fused

            (B,T,D)

        history_state

            (B,H)

        Returns
        -------

        conditioned

            (B,T,D)
        """

        if fused.ndim != 3:
            raise ValueError(
                f"Expected fused "
                f"(B,T,D), got "
                f"{tuple(fused.shape)}"
            )

        if history_state.ndim != 2:
            raise ValueError(
                f"Expected history_state "
                f"(B,H), got "
                f"{tuple(history_state.shape)}"
            )

        B, T, _ = fused.shape

        history = (
            history_state
            .unsqueeze(1)
            .expand(
                B,
                T,
                -1,
            )
        )

        conditioned = torch.cat(
            [
                fused,
                history,
            ],
            dim=-1,
        )

        conditioned = (
            self.projection(
                conditioned
            )
        )

        return conditioned

    # ---------------------------------------
    # Metadata
    # ---------------------------------------

    def summary(
        self,
    ):
        return {

            "type":
                "history_injection",

            "model_dim":
                self.model_dim,

            "history_dim":
                self.history_dim,
        }