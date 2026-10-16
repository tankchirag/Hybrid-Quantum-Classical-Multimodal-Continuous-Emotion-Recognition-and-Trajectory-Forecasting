"""
models.history.representation
=============================

History Representation Layer

Purpose
-------
Transforms continuous and discrete history
trajectories into a unified latent history space.

Pipeline
--------

Continuous History

(B,K,Nc)

    ↓

Continuous Projection

    ↓

(B,K,H)

--------------------------------------------

Discrete History

(B,K,Nd)

    ↓

Per-Label Embeddings

    ↓

(B,K,H)

--------------------------------------------

Mixed

(E_cont, E_disc)

    ↓

Concat

(B,K,2H)

    ↓

Merge Projection

(B,K,H)

Supports
--------

✓ Continuous Only

✓ Discrete Only

✓ Mixed Labels

✓ Dataset Agnostic

✓ Label Agnostic

✓ Recognition Compatible

✓ Forecasting Compatible

✓ Future GRU Compatible
"""

from __future__ import annotations

from typing import Dict
from typing import Optional

import torch
import torch.nn as nn


class HistoryRepresentation(
    nn.Module,
):
    """
    Unified History Representation Layer.
    """

    def __init__(
        self,
        history_dim: int,
        discrete_cardinalities:
        Optional[
            Dict[str, int]
        ] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.history_dim = (
            history_dim
        )

        self.discrete_cardinalities = (
            discrete_cardinalities
            or {}
        )

        # ==================================
        # Continuous Branch
        # ==================================

        self.continuous_projection = (
            None
        )

        # Lazy initialization because
        # Nc depends on dataset.

        # ==================================
        # Discrete Branch
        # ==================================

        self.discrete_embeddings = (
            nn.ModuleDict()
        )

        for (
            label_name,
            cardinality,
        ) in (
            self.discrete_cardinalities
            .items()
        ):

            self.discrete_embeddings[
                label_name
            ] = nn.Embedding(
                num_embeddings=
                cardinality,

                embedding_dim=
                history_dim,
            )

        self.discrete_norm = (
            nn.LayerNorm(
                history_dim
            )
        )

        # ==================================
        # Mixed Merge Layer
        # ==================================

        self.merge = nn.Sequential(

            nn.Linear(
                history_dim * 2,
                history_dim,
            ),

            nn.LayerNorm(
                history_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout,
            ),
        )

    # ======================================
    # Continuous
    # ======================================

    def _encode_continuous(
        self,
        continuous: torch.Tensor,
    ):

        if (
            self.continuous_projection
            is None
        ):

            nc = continuous.shape[-1]

            self.continuous_projection = (
                nn.Sequential(

                    nn.Linear(
                        nc,
                        self.history_dim,
                    ),

                    nn.LayerNorm(
                        self.history_dim,
                    ),

                    nn.GELU(),
                )
            ).to(
                continuous.device
            )

        return (
            self.continuous_projection(
                continuous
            )
        )

    # ======================================
    # Discrete
    # ======================================

    def _encode_discrete(
        self,
        discrete: torch.Tensor,
        label_names,
    ):

        embeddings = []

        for idx, name in (
            enumerate(
                label_names
            )
        ):

            if (
                name
                not in
                self.discrete_embeddings
            ):
                continue

            emb = (
                self.discrete_embeddings[
                    name
                ](
                    discrete[
                        :,
                        :,
                        idx
                    ]
                )
            )

            embeddings.append(
                emb
            )

        if len(
            embeddings
        ) == 0:

            return None

        summed = torch.stack(
            embeddings,
            dim=0,
        ).sum(
            dim=0
        )

        return self.discrete_norm(
            summed
        )

    # ======================================
    # Forward
    # ======================================

    def forward(
        self,
        continuous=None,
        discrete=None,
        discrete_label_names=None,
    ):

        cont_repr = None
        disc_repr = None

        if continuous is not None:

            cont_repr = (
                self._encode_continuous(
                    continuous
                )
            )

        if (
            discrete is not None
            and
            discrete_label_names
            is not None
        ):

            disc_repr = (
                self._encode_discrete(
                    discrete,
                    discrete_label_names,
                )
            )

        # ----------------------------------
        # Continuous Only
        # ----------------------------------

        if (
            cont_repr is not None
            and
            disc_repr is None
        ):

            return cont_repr

        # ----------------------------------
        # Discrete Only
        # ----------------------------------

        if (
            disc_repr is not None
            and
            cont_repr is None
        ):

            return disc_repr

        # ----------------------------------
        # Mixed
        # ----------------------------------

        if (
            cont_repr is not None
            and
            disc_repr is not None
        ):

            combined = torch.cat(
                [
                    cont_repr,
                    disc_repr,
                ],
                dim=-1,
            )

            return self.merge(
                combined
            )

        raise ValueError(
            "HistoryRepresentation "
            "received neither "
            "continuous nor discrete "
            "history."
        )