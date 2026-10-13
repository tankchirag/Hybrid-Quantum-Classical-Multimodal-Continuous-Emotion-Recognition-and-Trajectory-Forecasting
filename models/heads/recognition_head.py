"""
models.heads.recognition_head
=============================

Current-State Recognition Head

Input
-----

Encoder Outputs

    (B,T,D)

Output
------

Continuous

    (B)

Discrete

    (B,C)

Design

History Window
        ↓
Transformer Encoder
        ↓
Last Latent State
        ↓
Current-State Prediction
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class RecognitionHead(
    nn.Module,
):
    """
    Current-state recognition head.

    Predicts the label at the
    current timestep from the final
    encoder state.
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
        # Continuous
        # ----------------------------------

        self.continuous_heads = (
            nn.ModuleDict()
        )

        for target in self.continuous_targets:

            self.continuous_heads[target] = (
                nn.Sequential(

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
            )

        # ----------------------------------
        # Discrete
        # ----------------------------------

        self.discrete_heads = (
            nn.ModuleDict()
        )

        for (
            target,
            num_classes,
        ) in self.discrete_targets.items():

            self.discrete_heads[target] = (
                nn.Sequential(

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
            )

    def forward(
        self,
        encoder_outputs: torch.Tensor,
    ) -> Dict:

        outputs = {

            "continuous": {},
            "discrete": {},
        }

        # ==================================
        # Last Encoder State
        # ==================================

        current_state = (
            encoder_outputs[:, -1]
        )

        # Shape

        # (B,D)

        # ==================================
        # Continuous
        # ==================================

        for (
            name,
            head,
        ) in self.continuous_heads.items():

            outputs[
                "continuous"
            ][name] = (

                head(
                    current_state
                )

                .squeeze(-1)
            )

        # Result

        # (B)

        # ==================================
        # Discrete
        # ==================================

        for (
            name,
            head,
        ) in self.discrete_heads.items():

            outputs[
                "discrete"
            ][name] = head(
                current_state
            )

        # Result

        # (B,C)

        return outputs

    def summary(
        self,
    ):

        return {

            "recognition_type":
                "current_state",

            "continuous_targets":
                self.continuous_targets,

            "discrete_targets":
                self.discrete_targets,
        }