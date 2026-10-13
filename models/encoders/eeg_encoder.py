"""
models.encoders.eeg_encoder
===========================

EEG Encoder

Purpose
-------

Transforms synchronized Electroencephalography
(EEG) signals into latent temporal embeddings.

EEG captures:

    • Cortical Activation

    • Cognitive Workload

    • Attention

    • Mental Effort

    • Fatigue

    • Stress Responses

Input
-----

EEG Signal

    (B,T,1)

Output
------

EEG Embedding

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

Future EEG Datasets

Notes
-----

This encoder assumes EEG signals have already
been synchronized onto a common timeline.

The encoder is therefore independent of:

    sampling rates

    dataset-specific formats

    label definitions

    task types

Its only responsibility is:

    EEG
        ↓
    Representation
"""

from __future__ import annotations

from .temporal_conv_encoder import (
    TemporalConvEncoder,
)


class EEGEncoder(
    TemporalConvEncoder,
):
    """
    EEG temporal encoder.

    Input
    -----

        (B,T,1)

    Output
    ------

        (B,T,D)
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
            modality_name="eeg",
        )

    @property
    def modality(
        self,
    ) -> str:
        """
        Modality identifier.
        """

        return "eeg"
