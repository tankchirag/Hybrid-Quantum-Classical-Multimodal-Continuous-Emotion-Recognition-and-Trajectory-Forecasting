"""
models.encoders.text_encoder
============================

Text Encoder

Purpose
-------

Transforms synchronized text-derived features
into latent temporal embeddings.

The encoder is intentionally:

    • Dataset Agnostic
    • Label Agnostic
    • Task Agnostic

Input
-----

Text Features

    (B,T,C)

Current ULM-TSST Channels
-------------------------

Channel 0

    Text Valence

Channel 1

    Text Stress

Future Extensions
-----------------

The architecture supports future channels without
changing any downstream component.

Examples

    Sentiment Features

    Emotion Features

    Stress Markers

    LIWC Features

    OpenSMILE Linguistic Features

    BERT Embeddings

    RoBERTa Embeddings

    LLM Features

Output
------

Text Embedding

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

✓ Modality Agnostic

✓ Future Proof

Supported Datasets
------------------

RECOLA

SEWA

ULM-TSST

Future Multimodal Datasets

Notes
-----

The encoder operates on synchronized temporal
text features.

The encoder does NOT depend on:

    labels

    dataset names

    forecasting tasks

    classification tasks

    continuous targets

    discrete targets

Its only responsibility is:

    Text Features
        ↓
    Representation
"""

from __future__ import annotations

from .temporal_conv_encoder import (
    TemporalConvEncoder,
)


class TextEncoder(
    TemporalConvEncoder,
):
    """
    Text temporal encoder.

    Input
    -----

        (B,T,C)

    Output
    ------

        (B,T,D)

    Current Default

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
            modality_name="text",
        )

    @property
    def modality(
        self,
    ) -> str:
        """
        Modality identifier.
        """

        return "text"