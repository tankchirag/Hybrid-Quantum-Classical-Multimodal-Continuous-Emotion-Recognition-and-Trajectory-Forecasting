"""
models.temporal.transformer_ed
==============================

Unified Transformer Encoder-Decoder

Purpose
-------

Temporal modeling backbone shared by:

    • Baseline B1
    • Baseline B2
    • Baseline B3
    • QSAL

The module performs:

    Historical Context Modeling

        +

    Future Trajectory Modeling

using a Transformer Encoder-Decoder architecture.

Unlike earlier implementations, the temporal model
supports arbitrary label-history dimensionalities.

Design Goals
------------

✓ Dataset Agnostic

✓ Label Agnostic

✓ Continuous Labels

✓ Discrete Labels

✓ Recognition Compatible

✓ Forecasting Compatible

✓ Variable History Dimension

✓ Future Proof

Input
-----

fused

    (B,T,d_model)

Optional

label_history

    (B,T,L)

where

    L = number of historical targets

Examples

    L = 0

        no history

    L = 2

        valence
        arousal

    L = 5

        stress
        workload
        fatigue
        valence
        arousal

Output
------

enc_out

    (B,T,d_model)

dec_out

    (B,S,d_model)

where

    S = forecasting horizon
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(
    nn.Module,
):
    """
    Standard sinusoidal positional encoding.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
    ):
        super().__init__()

        pe = torch.zeros(
            max_len,
            d_model,
        )

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            *
            (
                -math.log(10000.0)
                / d_model
            )
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        seq_len = x.size(1)

        return (
            x
            +
            self.pe[:, :seq_len]
        )


# ============================================================
# Transformer Encoder Decoder
# ============================================================

class TransformerEncoderDecoder(
    nn.Module,
):
    """
    Unified temporal backbone.

    Input

        fused

            (B,T,256)

        optional label_history

            (B,T,L)

    Output

        enc_out

            (B,T,256)

        dec_out

            (B,S,256)
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        forecast_horizon: int = 10,
        history_dim: int = 0,
        max_seq_len: int = 5000,
    ):
        super().__init__()

        self.d_model = d_model
        self.history_dim = history_dim
        self.forecast_horizon = forecast_horizon

        # ----------------------------------
        # History Integration
        # ----------------------------------

        if history_dim > 0:

            self.history_projection = (
                nn.Linear(
                    d_model + history_dim,
                    d_model,
                )
            )

        else:

            self.history_projection = (
                nn.Identity()
            )

        # ----------------------------------
        # Positional Encoding
        # ----------------------------------

        self.positional_encoding = (
            PositionalEncoding(
                d_model=d_model,
                max_len=max_seq_len,
            )
        )

        # ----------------------------------
        # Encoder
        # ----------------------------------

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_encoder_layers,
            )
        )

        # ----------------------------------
        # Decoder
        # ----------------------------------

        decoder_layer = (
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
        )

        self.decoder = (
            nn.TransformerDecoder(
                decoder_layer,
                num_layers=num_decoder_layers,
            )
        )

        # ----------------------------------
        # Forecast Queries
        # ----------------------------------

        self.query_tokens = (
            nn.Parameter(
                torch.randn(
                    forecast_horizon,
                    d_model,
                )
            )
        )

    # --------------------------------------------------------
    # History Integration
    # --------------------------------------------------------

    def _integrate_history(
        self,
        fused: torch.Tensor,
        label_history: torch.Tensor | None,
    ) -> torch.Tensor:

        if self.history_dim == 0:
            return fused

        if label_history is None:

            raise ValueError(
                "history_dim > 0 but "
                "label_history was not provided."
            )

        x = torch.cat(
            [
                fused,
                label_history,
            ],
            dim=-1,
        )

        return self.history_projection(
            x
        )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        fused: torch.Tensor,
        label_history: torch.Tensor | None = None,
    ):
        """
        Parameters
        ----------
        fused

            (B,T,d_model)

        label_history

            Optional

            (B,T,L)

        Returns
        -------
        enc_out

            (B,T,d_model)

        dec_out

            (B,S,d_model)
        """

        B, T, D = fused.shape

        if D != self.d_model:

            raise ValueError(
                f"Expected d_model="
                f"{self.d_model}, "
                f"received {D}."
            )

        # ----------------------------------
        # History Integration
        # ----------------------------------

        x = self._integrate_history(
            fused,
            label_history,
        )

        # ----------------------------------
        # Positional Encoding
        # ----------------------------------

        x = self.positional_encoding(
            x
        )

        # ----------------------------------
        # Encoder
        # ----------------------------------

        enc_out = self.encoder(
            x
        )

        # ----------------------------------
        # Decoder Queries
        # ----------------------------------

        queries = (
            self.query_tokens
            .unsqueeze(0)
            .expand(
                B,
                -1,
                -1,
            )
        )

        # ----------------------------------
        # Decoder
        # ----------------------------------

        dec_out = self.decoder(
            tgt=queries,
            memory=enc_out,
        )

        return enc_out, dec_out

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    def summary(
        self,
    ) -> dict:

        return {
            "d_model":
                self.d_model,

            "history_dim":
                self.history_dim,

            "forecast_horizon":
                self.forecast_horizon,

            "architecture":
                "transformer_encoder_decoder",
        }

    def extra_repr(
        self,
    ) -> str:

        return (
            f"d_model={self.d_model}, "
            f"history_dim={self.history_dim}, "
            f"forecast_horizon={self.forecast_horizon}"
        )