"""
discrete_label_generator.py
===========================

Discrete Label Generator

Converts latent human-state trajectories into
discrete classification labels.

This module provides:

    ✓ Emotion Classes

    ✓ Stress Classes

    ✓ Workload Classes

    ✓ Alertness Classes

    ✓ Drowsiness Classes

    ✓ Karolinska Sleepiness Scale (KSS)

The generator is:

    ✓ Label agnostic

    ✓ Dataset agnostic

    ✓ Future-proof

    ✓ Configurable

Motivation
----------

Many datasets contain both:

Continuous labels

    valence
    arousal
    stress

and

Discrete labels

    happy
    sad
    stress class
    drowsiness level

This generator derives discrete labels
from latent trajectories while maintaining
physiological consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from datasets.generators.latent_state_generator import (
    LatentStateOutput,
)


# ---------------------------------------------------------
# Output
# ---------------------------------------------------------

@dataclass
class DiscreteLabelsOutput:
    """
    Container for discrete labels.
    """

    labels: Dict[str, torch.Tensor]


# ---------------------------------------------------------
# Generator
# ---------------------------------------------------------

class DiscreteLabelGenerator:
    """
    Generate configurable discrete labels.

    Example config:

    {
        "emotion_class": True,
        "stress_class": True,
        "workload_class": True,
        "alertness_class": True,
        "drowsiness_class": True,
        "kss_score": True,
    }
    """

    def __init__(
        self,
        label_config: Dict[str, bool],
    ) -> None:

        self.label_config = dict(
            label_config
        )

    # -----------------------------------------------------
    # Emotion Mapping
    # -----------------------------------------------------

    @staticmethod
    def _emotion_class(
        valence: torch.Tensor,
        arousal: torch.Tensor,
    ) -> torch.Tensor:
        """
        VA → emotion class

        Classes

        0 = Neutral
        1 = Happy
        2 = Relaxed
        3 = Sad
        4 = Angry
        5 = Excited
        """

        emotion = torch.zeros_like(
            valence,
            dtype=torch.long,
        )

        happy = (
            (valence > 0.4)
            &
            (arousal > 0.2)
        )

        relaxed = (
            (valence > 0.4)
            &
            (arousal <= 0.2)
        )

        sad = (
            (valence < -0.4)
            &
            (arousal < 0.0)
        )

        angry = (
            (valence < -0.4)
            &
            (arousal > 0.2)
        )

        excited = (
            (valence > 0.4)
            &
            (arousal > 0.6)
        )

        emotion[happy] = 1
        emotion[relaxed] = 2
        emotion[sad] = 3
        emotion[angry] = 4
        emotion[excited] = 5

        return emotion

    # -----------------------------------------------------
    # Generic Low-Mid-High
    # -----------------------------------------------------

    @staticmethod
    def _three_class(
        signal: torch.Tensor,
        low: float = 0.33,
        high: float = 0.66,
    ) -> torch.Tensor:
        """
        Create 3-level label.

        0 = low
        1 = medium
        2 = high
        """

        out = torch.zeros_like(
            signal,
            dtype=torch.long,
        )

        out[
            (signal >= low)
            &
            (signal < high)
        ] = 1

        out[
            signal >= high
        ] = 2

        return out

    # -----------------------------------------------------
    # Drowsiness
    # -----------------------------------------------------

    @staticmethod
    def _drowsiness_class(
        fatigue: torch.Tensor,
    ) -> torch.Tensor:
        """
        0 Awake
        1 Mild
        2 Moderate
        3 Severe
        """

        out = torch.zeros_like(
            fatigue,
            dtype=torch.long,
        )

        out[
            fatigue > 0.30
        ] = 1

        out[
            fatigue > 0.60
        ] = 2

        out[
            fatigue > 0.85
        ] = 3

        return out

    # -----------------------------------------------------
    # KSS
    # -----------------------------------------------------

    @staticmethod
    def _kss_score(
        fatigue: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert fatigue → 1..9 KSS score.
        """

        score = (
            1 +
            8 * fatigue
        )

        score = torch.round(
            score
        )

        score = torch.clamp(
            score,
            1,
            9,
        )

        return score.long()

    # -----------------------------------------------------
    # Main
    # -----------------------------------------------------

    def generate(
        self,
        latent: LatentStateOutput,
    ) -> DiscreteLabelsOutput:
        """
        Generate enabled labels.
        """

        candidates = {

            "emotion_class":
                self._emotion_class(
                    latent.valence,
                    latent.arousal,
                ),

            "stress_class":
                self._three_class(
                    latent.stress,
                ),

            "workload_class":
                self._three_class(
                    latent.workload,
                ),

            "alertness_class":
                self._three_class(
                    1.0 - latent.fatigue,
                ),

            "drowsiness_class":
                self._drowsiness_class(
                    latent.fatigue,
                ),

            "kss_score":
                self._kss_score(
                    latent.fatigue,
                ),
        }

        output = {}

        for (
            name,
            enabled,
        ) in self.label_config.items():

            if (
                enabled
                and
                name in candidates
            ):
                output[name] = (
                    candidates[name]
                )

        return DiscreteLabelsOutput(
            labels=output
        )

    # -----------------------------------------------------

    def active_labels(
        self,
    ):
        """
        Return enabled labels.
        """

        return [

            name

            for name, enabled

            in self.label_config.items()

            if enabled

        ]