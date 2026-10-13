"""
models.encoders.emg_encoder
===========================

EMG Encoder

Purpose
-------

Transforms synchronized Electromyography (EMG)
signals into latent temporal embeddings.

EMG captures:

    • Muscle Activation

    • Facial Tension

    • Motor Activity

    • Physiological Stress Responses

    • Fatigue

Input
-----

EMG Signal

    (B,T,1)

Output
------

EMG Embedding

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
"""

from __future__ import annotations

from .temporal_conv_encoder import (
    TemporalConvEncoder,
)


class EMGEncoder(
    TemporalConvEncoder,
):
    """
    EMG temporal encoder.
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
            modality_name="emg",
        )

    @property
    def modality(self) -> str:
        return "emg"