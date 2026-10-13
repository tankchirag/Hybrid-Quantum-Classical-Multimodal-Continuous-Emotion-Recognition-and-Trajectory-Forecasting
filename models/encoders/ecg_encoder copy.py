"""
models.encoders.ecg_encoder
===========================

ECG Encoder

Purpose
-------

Transforms synchronized ECG signals into
latent temporal representations.

Input
-----

ECG Signal

    (B, T, 1)

Output
------

ECG Embedding

    (B, T, D)

Default

    D = 128

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

Supported Datasets
------------------

RECOLA

SEWA

ULM-TSST

Future Physiological Datasets

Notes
-----

This encoder assumes signals have already been:

    synchronized

and represented as

    (B,T,C)

by the Synchronizer.

The encoder therefore does NOT know anything about:

    sampling rate

    dataset

    labels

    forecasting

    classification

Its only responsibility is:

    Signal
        ↓
    Representation
"""

from __future__ import annotations

from .temporal_conv_encoder import (
    TemporalConvEncoder,
)


class ECGEncoder(
    TemporalConvEncoder,
):
    """
    ECG-specific temporal encoder.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__(
            input_channels=1,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            modality_name="ecg",
        )

    @property
    def modality(self) -> str:
        """
        Modality identifier.
        """

        return "ecg"