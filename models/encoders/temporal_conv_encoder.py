"""
models.encoders.temporal_conv_encoder
=====================================

Universal Temporal Convolutional Encoder

Purpose
-------

Convert synchronized time-series signals

    (B,T,C)

into modality embeddings

    (B,T,D)

This module acts as the common backbone
for most physiological and behavioral
signals in the framework.

Supported Modalities
--------------------

ECG

EDA

RESP

BPM

EEG

EMG

Eye Tracking

Future Biosignals

Design Goals
------------

✓ Dataset Agnostic

✓ Label Agnostic

✓ Continuous Labels

✓ Discrete Labels

✓ Recognition Compatible

✓ Forecasting Compatible

✓ Sensor Agnostic

✓ Future Proof
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base_encoder import BaseEncoder


class TemporalConvEncoder(BaseEncoder):
    """
    Universal temporal encoder.

    Input
    -----

    (B,T,C)

    Output
    ------

    (B,T,D)
    """

    def __init__(
        self,
        input_channels: int,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
        modality_name: str = "generic",
    ):
        super().__init__(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            modality_name=modality_name,
        )

        layers = []

        in_channels = input_channels

        for _ in range(num_layers):

            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=hidden_dim,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),

                    nn.BatchNorm1d(
                        hidden_dim
                    ),

                    nn.GELU(),

                    nn.Dropout(
                        dropout
                    ),
                ]
            )

            in_channels = hidden_dim

        self.backbone = nn.Sequential(
            *layers
        )

        self.projection = nn.Linear(
            hidden_dim,
            embedding_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        self.validate_input(
            x
        )

        # ------------------------
        # (B,T,C)
        #
        # Conv1d needs:
        #
        # (B,C,T)
        # ------------------------

        x = x.permute(
            0,
            2,
            1,
        )

        feats = self.backbone(
            x
        )

        feats = feats.permute(
            0,
            2,
            1,
        )

        emb = self.projection(
            feats
        )

        return emb