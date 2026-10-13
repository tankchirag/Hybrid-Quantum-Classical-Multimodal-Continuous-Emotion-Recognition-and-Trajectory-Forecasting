"""
models.encoders.eye_encoder
===========================

Eye Tracking Encoder

Purpose
-------

Transforms synchronized eye-tracking signals
into latent temporal representations.

Current Input Channels
----------------------

Channel 0

    Pupil Size

Channel 1

    Blink / Eye Activity

Future Extensions
-----------------

The architecture supports future channels:

    Gaze X

    Gaze Y

    Fixation Duration

    Saccade Velocity

without modifying downstream modules.

Input
-----

Eye Signal

    (B,T,2)

Output
------

Eye Embedding

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

Eye Tracking Importance
-----------------------

Eye behavior is highly correlated with:

    Stress

    Workload

    Fatigue

    Attention

    Engagement

making it an important modality for
human-state modeling.
"""

from __future__ import annotations

from .temporal_conv_encoder import (
    TemporalConvEncoder,
)


class EyeEncoder(
    TemporalConvEncoder,
):
    """
    Eye-tracking temporal encoder.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
        input_channels: int = 2,
    ):
        super().__init__(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            modality_name="eye",
        )

    @property
    def modality(self) -> str:
        return "eye"