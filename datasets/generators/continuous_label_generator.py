"""
continuous_label_generator.py
=============================

Continuous Label Generator

Converts latent human-state trajectories into
dataset continuous labels.

This module provides a fully:

    ✓ label agnostic
    ✓ modality agnostic
    ✓ dataset agnostic
    ✓ configurable

continuous labeling system.

Motivation
----------

Different datasets may require different
continuous labels.

Example A:

    valence
    arousal

Example B:

    valence
    arousal
    stress

Example C:

    workload
    fatigue

Example D:

    any future custom trajectory

The generator enables all such combinations
without code changes.

Expected Workflow
-----------------

TSST Protocol
      ↓

Latent State Generator
      ↓

Continuous Label Generator
      ↓

Dataset Labels

Output
------

{
    "valence": Tensor(T,),
    "arousal": Tensor(T,),
    "stress": Tensor(T,),
    ...
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from datasets.generators.latent_state_generator import (
    LatentStateOutput,
)


@dataclass
class ContinuousLabelsOutput:
    """
    Container for generated labels.
    """

    labels: Dict[str, torch.Tensor]


class ContinuousLabelGenerator:
    """
    Generate configurable continuous labels.

    Parameters
    ----------
    label_config : dict

        Example:

        {
            "valence": True,
            "arousal": True,
            "stress": True,
            "workload": True,
            "fatigue": False,
            "dominance": False,
            "engagement": False
        }
    """

    # -----------------------------------------------------
    # Constructor
    # -----------------------------------------------------

    def __init__(
        self,
        label_config: Dict[str, bool],
    ) -> None:

        self.label_config = dict(
            label_config
        )

    # -----------------------------------------------------
    # Available labels
    # -----------------------------------------------------

    @staticmethod
    def _available_labels(
        latent: LatentStateOutput,
    ) -> Dict[str, torch.Tensor]:
        """
        Create mapping from latent trajectories
        to labels.
        """

        return {

            "valence":
                latent.valence,

            "arousal":
                latent.arousal,

            "stress":
                latent.stress,

            "workload":
                latent.workload,

            "fatigue":
                latent.fatigue,

            "dominance":
                latent.dominance,

            "engagement":
                latent.engagement,
        }

    # -----------------------------------------------------
    # Label generation
    # -----------------------------------------------------

    def generate(
        self,
        latent: LatentStateOutput,
    ) -> ContinuousLabelsOutput:
        """
        Generate continuous labels.

        Parameters
        ----------
        latent : LatentStateOutput

        Returns
        -------
        ContinuousLabelsOutput
        """

        source = self._available_labels(
            latent
        )

        output = {}

        for name, enabled in (
            self.label_config.items()
        ):

            if not enabled:
                continue

            if name not in source:
                continue

            output[name] = source[name]

        return ContinuousLabelsOutput(
            labels=output
        )

    # -----------------------------------------------------
    # Utility
    # -----------------------------------------------------

    def active_labels(self):
        """
        Return enabled labels.
        """

        return [

            name

            for name, enabled

            in self.label_config.items()

            if enabled
        ]