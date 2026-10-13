"""
models.encoders.resp_encoder
============================

Respiration Encoder

Purpose
-------

Transforms synchronized respiration signals
into latent temporal representations.

Respiration contains information related to:

    • Stress

    • Arousal

    • Anxiety

    • Fatigue

    • Cognitive Load

Input
-----

Respiration Signal

    (B,T,1)

Output
------

Respiration Embedding

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

Notes
-----

The encoder assumes the signal has already
been synchronized onto a common timeline.

It therefore remains independent of:

    sampling rates

    datasets

    label types

    tasks

Its only responsibility is:

    Respiration
        ↓
    Representation
"""

from __future__ import annotations

from .temporal_conv_encoder import (
    TemporalConvEncoder,
)


class RESPEncoder(
    TemporalConvEncoder,
):
    """
    Respiration temporal encoder.
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
            modality_name="resp",
        )

    @property
    def modality(self) -> str:
        return "resp"