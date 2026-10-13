"""
models.encoders.bpm_encoder
===========================

BPM Encoder

Purpose
-------

Transforms synchronized BPM (Heart Rate)
signals into latent temporal embeddings.

BPM captures:

    • Cardiovascular Activity

    • Stress Responses

    • Arousal

    • Recovery Dynamics

    • Workload Effects

Input
-----

BPM Signal

    (B,T,1)

Output
------

BPM Embedding

    (B,T,D)

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

Future Datasets

Notes
-----

The encoder expects:

    heart rate trajectories

already synchronized by the data pipeline.

It therefore does not depend on:

    sampling rate

    dataset

    recognition labels

    forecasting labels

Its only responsibility is:

    BPM
        ↓
    Representation
"""

from __future__ import annotations

from .temporal_conv_encoder import (
    TemporalConvEncoder,
)


class BPMEncoder(
    TemporalConvEncoder,
):
    """
    BPM temporal encoder.
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
            modality_name="bpm",
        )

    @property
    def modality(self) -> str:
        return "bpm"