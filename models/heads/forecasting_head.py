"""
models.heads.forecasting_head
=============================

Unified Forecasting Head

Purpose
-------

Predict future trajectories from Transformer
decoder outputs.

Supports:

    • Continuous Forecasting

    • Discrete-State Forecasting

Target selection is entirely configuration driven.

Input
-----

Decoder Outputs

    (B,S,D)

Default

    D = 256

Output
------

{
    "continuous":
    {
        target_name:
            (B,S)
    },

    "discrete":
    {
        target_name:
            (B,S,C)
    }
}

Design Goals
------------

✓ Dataset Agnostic

✓ Label Agnostic

✓ Continuous Labels

✓ Discrete Labels

✓ Recognition Compatible

✓ Forecasting Compatible

✓ Future Proof
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ForecastingHead(nn.Module):
    """
    Unified forecasting head.
    """

    def __init__(
        self,
        continuous_targets: list[str] | None = None,
        discrete_targets: dict[str, int] | None = None,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.continuous_targets = (
            continuous_targets or []
        )

        self.discrete_targets = (
            discrete_targets or {}
        )

        # ----------------------------------
        # Continuous Forecasts
        # ----------------------------------

        self.continuous_heads = (
            nn.ModuleDict()
        )

        for target in self.continuous_targets:

            self.continuous_heads[target] = nn.Sequential(

                nn.Linear(
                    input_dim,
                    hidden_dim,
                ),

                nn.GELU(),

                nn.Dropout(
                    dropout,
                ),

                nn.Linear(
                    hidden_dim,
                    1,
                ),
            )

        # ----------------------------------
        # Discrete Forecasts
        # ----------------------------------

        self.discrete_heads = (
            nn.ModuleDict()
        )

        for (
            target,
            num_classes,
        ) in self.discrete_targets.items():

            self.discrete_heads[target] = nn.Sequential(

                nn.Linear(
                    input_dim,
                    hidden_dim,
                ),

                nn.GELU(),

                nn.Dropout(
                    dropout,
                ),

                nn.Linear(
                    hidden_dim,
                    num_classes,
                ),
            )

    def forward(
        self,
        decoder_outputs: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        decoder_outputs

            (B,S,256)

        Returns
        -------
        Dict
        """

        outputs = {

            "continuous": {},

            "discrete": {},
        }

        # ----------------------------------
        # Continuous Forecasts
        # ----------------------------------

        for (
            name,
            head,
        ) in self.continuous_heads.items():

            outputs[
                "continuous"
            ][name] = head(
                decoder_outputs
            ).squeeze(-1)

        # ----------------------------------
        # Discrete Forecasts
        # ----------------------------------

        for (
            name,
            head,
        ) in self.discrete_heads.items():

            outputs[
                "discrete"
            ][name] = head(
                decoder_outputs
            )

        return outputs

    def summary(
        self,
    ) -> dict:

        return {

            "continuous_targets":
                self.continuous_targets,

            "discrete_targets":
                self.discrete_targets,
        }