"""
models.tokenizers.tokenizer
===========================

Universal Tokenizer

Purpose
-------

Convert arbitrary embedding streams into a
unified token representation suitable for:

    • Classical Fusion

    • Quantum Fusion

    • Transformers

    • Recognition

    • Forecasting

The tokenizer is intentionally independent of:

    • Dataset

    • Modalities

    • Labels

    • Tasks

Input
-----

Dictionary of embeddings.

Example

{
    "ecg": (B,T,D),

    "eda": (B,T,D),

    "audio": (B,T,D),

    "video": (B,T,D),

    "continuous_labels": (B,T,D),

    "discrete_labels": (B,T,D),
}

Output
------

tokens

    (B,T,N,D)

where

    B = batch size

    T = sequence length

    N = number of active tokens

    D = embedding dimension

Design Goals
------------

✓ Dataset Agnostic

✓ Label Agnostic

✓ Continuous Labels

✓ Discrete Labels

✓ Variable Modality Count

✓ Forecasting Compatible

✓ Recognition Compatible

✓ Missing Token Compatible

✓ Future Proof
"""

from __future__ import annotations

from typing import Dict
from typing import List

import torch
import torch.nn as nn


class Tokenizer(
    nn.Module,
):
    """
    Universal tokenizer.

    Converts arbitrary embedding streams into
    a unified token representation.
    """

    DEFAULT_TOKEN_ORDER = [

        # ----------------------------------
        # Physiological
        # ----------------------------------

        "ecg",
        "eda",
        "eeg",
        "emg",
        "resp",
        "bpm",

        # ----------------------------------
        # Behavioral
        # ----------------------------------

        "eye",
        "audio",
        "video",
        "text",

        # ----------------------------------
        # Label History
        # ----------------------------------

        "continuous_labels",
        "discrete_labels",

        # ----------------------------------
        # Future Extensions
        # ----------------------------------

        "task",
        "cls",
    ]

    def __init__(
        self,
        embedding_dim: int = 128,
        token_order: List[str] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = (
            embedding_dim
        )

        self.token_order = (
            token_order
            or self.DEFAULT_TOKEN_ORDER
        )

        # ----------------------------------
        # Token-Type Embeddings
        # ----------------------------------

        self.token_type_embeddings = (
            nn.Parameter(
                torch.randn(
                    len(self.token_order),
                    embedding_dim,
                )
            )
        )

        # ----------------------------------
        # Token Projections
        # ----------------------------------

        self.projections = (
            nn.ModuleDict(
                {
                    token_name:
                    nn.Linear(
                        embedding_dim,
                        embedding_dim,
                    )
                    for token_name
                    in self.token_order
                }
            )
        )

        self.dropout = (
            nn.Dropout(
                dropout
            )
        )

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(
        self,
        embedding_dict: Dict[
            str,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        embedding_dict

            {
                name:
                    (B,T,D)
            }

        Returns
        -------
        tokens

            (B,T,N,D)
        """

        token_list = []

        for idx, token_name in enumerate(
            self.token_order
        ):

            emb = (
                embedding_dict.get(
                    token_name,
                    None,
                )
            )

            if emb is None:
                continue

            token = (
                self.projections[
                    token_name
                ](
                    emb
                )
            )

            type_embedding = (
                self.token_type_embeddings[
                    idx
                ]
                .view(
                    1,
                    1,
                    -1,
                )
            )

            token = (
                token
                +
                type_embedding
            )

            token = self.dropout(
                token
            )

            token_list.append(
                token
            )

        if len(token_list) == 0:

            raise ValueError(
                "Tokenizer received "
                "no active embeddings."
            )

        tokens = torch.stack(
            token_list,
            dim=2,
        )

        return tokens

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

    @property
    def supported_tokens(
        self,
    ) -> List[str]:

        return list(
            self.token_order
        )

    @property
    def num_supported_tokens(
        self,
    ) -> int:

        return len(
            self.token_order
        )

    def summary(
        self,
    ) -> dict:

        return {

            "embedding_dim":
                self.embedding_dim,

            "supported_tokens":
                self.supported_tokens,

            "num_supported_tokens":
                self.num_supported_tokens,
        }