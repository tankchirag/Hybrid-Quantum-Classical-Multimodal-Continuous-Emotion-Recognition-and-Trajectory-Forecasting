"""
models.history.adapter
======================

Dynamic History Adapter

Purpose
-------
Convert arbitrary continuous and/or
discrete label histories into a common
tensor representation.

Supports
--------
✓ Continuous Labels

✓ Discrete Labels

✓ Mixed Labels

✓ Dataset Agnostic

✓ Label Agnostic

✓ Recognition Compatible

✓ Forecasting Compatible

✓ Future GRU Compatible

Input
-----

continuous_labels

{
    label_name: (B,K)
}

discrete_labels

{
    label_name: (B,K)
}

Output
------

{
    "continuous": (B,K,Nc) or None,

    "discrete": (B,K,Nd) or None
}
"""

from __future__ import annotations

from typing import Dict
from typing import Optional

import torch
import torch.nn as nn


class HistoryAdapter(
    nn.Module,
):
    """
    Dynamic history tensor harmonizer.

    No learnable parameters.

    Converts dictionary-based histories
    into tensor-based histories.
    """

    def __init__(
        self,
    ):
        super().__init__()

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    @staticmethod
    def _stack_history(
        history_dict: Dict[
            str,
            torch.Tensor,
        ],
        dtype=None,
    ) -> Optional[
        torch.Tensor
    ]:

        if (
            history_dict is None
            or
            len(history_dict) == 0
        ):
            return None

        tensors = []

        for name in sorted(
            history_dict.keys()
        ):

            x = history_dict[name]

            if dtype is not None:
                x = x.to(dtype)

            tensors.append(
                x.unsqueeze(-1)
            )

        return torch.cat(
            tensors,
            dim=-1,
        )

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(
        self,
        continuous_labels=None,
        discrete_labels=None,
    ):

        continuous = (
            self._stack_history(
                continuous_labels,
                dtype=torch.float32,
            )
        )

        discrete = (
            self._stack_history(
                discrete_labels,
                dtype=torch.long,
            )
        )

        return {

            "continuous":
                continuous,

            "discrete":
                discrete,

            "num_continuous":
                (
                    0
                    if continuous is None
                    else continuous.shape[-1]
                ),

            "num_discrete":
                (
                    0
                    if discrete is None
                    else discrete.shape[-1]
                ),
        }