"""
models.encoders.video_encoder
=============================

Video Encoder

Purpose
-------

Transforms synchronized video-derived features
into latent temporal representations.

The encoder is:

    • Dataset Agnostic
    • Label Agnostic
    • Task Agnostic

Input
-----

Video Signal

    (B,T,C)

Current ULM-TSST Channels
-------------------------

Channel 0

    Facial Motion

Channel 1

    Smile Intensity

Future Extensions
-----------------

Supports additional facial feature channels:

    Facial Action Units

    Face Embeddings

    Head Pose

    Eye Contact

    Expression Features

    MediaPipe Features

    OpenFace Features

without changing downstream modules.

Output
------

Video Embedding

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

Future Video Datasets

Notes
-----

The encoder assumes inputs have already
been synchronized onto a common timeline.

Therefore it remains independent of:

    frame rate

    dataset

    labels

    task type

Its only responsibility is:

    Video Features
        ↓
    Representation
"""

from __future__ import annotations

from .temporal_conv_encoder import (
    TemporalConvEncoder,
)


class VideoEncoder(
    TemporalConvEncoder,
):
    """
    Video temporal encoder.

    Input
    -----

        (B,T,C)

    Output
    ------

        (B,T,D)

    Default

        C = 2
        D = 128
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
            modality_name="video",
        )

    @property
    def modality(
        self,
    ) -> str:
        """
        Modality identifier.
        """

        return "video"