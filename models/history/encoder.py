"""
models.history.encoder
======================

History Trajectory Encoder

Purpose
-------
Encode temporal state trajectories into a
compact latent history representation.

Input
-----

(B,K,H)

Example

(B,50,64)

Output
------

(B,H)

Example

(B,64)

Design Goals
------------

✓ Dataset Agnostic

✓ Label Agnostic

✓ Continuous Compatible

✓ Discrete Compatible

✓ Mixed Labels Compatible

✓ Recognition Compatible

✓ Forecasting Compatible

✓ Future Memory Compatible
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HistoryEncoder(
    nn.Module,
):
    """
    GRU-based trajectory encoder.
    """

    def __init__(
        self,
        history_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        history_dropout: float = 0.20,
    ):
        super().__init__()

        self.history_dim = (
            history_dim
        )

        self.history_dropout_p = (
            history_dropout
        )

        self.gru = nn.GRU(
            input_size=
                history_dim,

            hidden_size=
                history_dim,

            num_layers=
                num_layers,

            batch_first=True,

            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.norm = (
            nn.LayerNorm(
                history_dim
            )
        )

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(
        self,
        history_representation:
        torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        history_representation

            (B,K,H)

        Returns
        -------
        history_state

            (B,H)
        """

        _, h_n = self.gru(
            history_representation
        )

        history_state = (
            h_n[-1]
        )

        history_state = (
            self.norm(
                history_state
            )
        )

        if (
            self.training
            and
            self.history_dropout_p > 0
        ):

            keep_mask = (
                torch.rand(
                    history_state.size(0),
                    1,
                    device=
                    history_state.device,
                )
                >
                self.history_dropout_p
            ).float()

            history_state = (
                history_state
                *
                keep_mask
            )

        return history_state