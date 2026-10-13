"""
models.encoders.base_encoder
============================

Universal Encoder Interface

Purpose
-------

This module defines the common encoder API used
throughout the thesis framework.

The goal is that ALL modality encoders expose a
consistent, dataset-agnostic interface.

Supported Modalities
--------------------

Physiological

    ECG
    EDA
    EEG
    EMG
    RESP
    BPM

Behavioral

    Eye Tracking
    Audio
    Video
    Text

Design Principles
-----------------

✓ Dataset Agnostic

✓ Label Agnostic

✓ Continuous Labels

✓ Discrete Labels

✓ Recognition Compatible

✓ Forecasting Compatible

✓ Modality Agnostic

✓ Future Proof

Encoder Contract
----------------

Input

    (B, T, C)

Output

    (B, T, D)

where

    B = batch size
    T = temporal length
    C = input channels
    D = embedding dimension

Encoders NEVER operate on labels.

They only transform signals into
latent representations.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(
    nn.Module,
    ABC,
):
    """
    Universal encoder base class.

    All modality encoders must inherit
    from this interface.
    """

    def __init__(
        self,
        input_channels: int,
        embedding_dim: int = 128,
        modality_name: str = "unknown",
    ):
        super().__init__()

        self.input_channels = (
            input_channels
        )

        self.embedding_dim = (
            embedding_dim
        )

        self.modality_name = (
            modality_name
        )

    # --------------------------------------------------
    # Properties
    # --------------------------------------------------

    @property
    def output_dim(
        self,
    ) -> int:
        """
        Encoder output dimension.
        """

        return self.embedding_dim

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def validate_input(
        self,
        x: torch.Tensor,
    ) -> None:
        """
        Validate input format.

        Expected:

            (B, T, C)
        """

        if not isinstance(
            x,
            torch.Tensor,
        ):
            raise TypeError(
                f"{self.modality_name}: "
                f"expected Tensor, "
                f"received {type(x)}"
            )

        if x.ndim != 3:

            raise ValueError(
                f"{self.modality_name}: "
                f"expected shape (B,T,C), "
                f"received {tuple(x.shape)}"
            )

        channels = x.shape[-1]

        if (
            channels
            != self.input_channels
        ):
            raise ValueError(
                f"{self.modality_name}: "
                f"expected "
                f"{self.input_channels} channels "
                f"received {channels}"
            )

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

    def encoder_info(
        self,
    ) -> dict:
        """
        Encoder metadata.
        """

        return {

            "modality":
                self.modality_name,

            "input_channels":
                self.input_channels,

            "embedding_dim":
                self.embedding_dim,
        }

    # --------------------------------------------------
    # Abstract API
    # --------------------------------------------------

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x

            (B,T,C)

        Returns
        -------
        Tensor

            (B,T,D)
        """
        pass

    # --------------------------------------------------
    # String Representation
    # --------------------------------------------------

    def __repr__(
        self,
    ) -> str:

        return (

            f"{self.__class__.__name__}("
            f"modality={self.modality_name}, "
            f"channels={self.input_channels}, "
            f"embedding_dim={self.embedding_dim}"
            f")"

        )