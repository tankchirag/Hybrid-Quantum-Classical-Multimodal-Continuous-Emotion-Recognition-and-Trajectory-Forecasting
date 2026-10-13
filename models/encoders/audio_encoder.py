"""
models.encoders.audio_encoder
=============================

Audio Encoder

Purpose
-------

Transforms synchronized audio features into
latent temporal representations.

The encoder is intentionally:

    • Dataset Agnostic
    • Label Agnostic
    • Task Agnostic

Input
-----

Audio Signal

    (B,T,C)

Current ULM-TSST channels:

    Channel 0:
        Pitch

    Channel 1:
        Energy

Future Extensions
-----------------

Supports additional channels without changing
the downstream architecture:

    MFCCs
    Wav2Vec Features
    Spectrogram Features
    Prosody Features
    Whisper Features
    OpenSMILE Features

Output
------

Audio Embedding

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

Future Multimodal Datasets

Notes
-----

The encoder operates on synchronized
time-series representations.

It is independent of:

    sampling rates

    datasets

    recognition labels

    forecasting labels

Its only responsibility is:

    Audio
        ↓
    Representation
"""

from __future__ import annotations

from .temporal_conv_encoder import (
    TemporalConvEncoder,
)


class AudioEncoder(
    TemporalConvEncoder,
):
    """
    Audio temporal encoder.

    Input
    -----

        (B,T,C)

    Output
    ------

        (B,T,D)

    Current default:

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
            modality_name="audio",
        )

    @property
    def modality(self) -> str:
        """
        Modality identifier.
        """

        return "audio"