"""
models.encoders.eda_encoder
===========================

EDA Encoder

Purpose
-------

Transforms Electrodermal Activity (EDA)
signals into latent temporal embeddings.

EDA measures:

    • Skin Conductance

    • Sympathetic Activation

    • Physiological Stress

Input
-----

EDA Signal

    (B,T,1)

Output
------

EDA Embedding

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

This encoder assumes signals are already:

    synchronized

by the Synchronizer component.

The encoder therefore does NOT depend on:

    sampling rate

    dataset name

    label type

    forecasting horizon

    classification tasks

Its only responsibility is:

    EDA Signal
        ↓
    Representation
"""

from __future__ import annotations

from .temporal_conv_encoder import (
    TemporalConvEncoder,
)


class EDAEncoder(
    TemporalConvEncoder,
):
    """
    Electrodermal Activity encoder.

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
            modality_name="eda",
        )

    @property
    def modality(
        self,
    ) -> str:
        """
        Modality identifier.
        """

        return "eda"