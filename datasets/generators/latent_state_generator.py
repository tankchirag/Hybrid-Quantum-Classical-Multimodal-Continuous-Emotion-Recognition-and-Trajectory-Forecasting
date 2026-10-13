"""
latent_state_generator.py
=========================

Latent Human State Generator

This module generates the hidden psychological state
of a synthetic human participant.

The latent state is not directly observed by sensors.

Instead it acts as the underlying cause of:

    • emotions
    • stress
    • workload
    • engagement
    • fatigue
    • physiology

which then influence multimodal sensor signals.

The generated trajectories are:

    valence(t)
    arousal(t)
    stress(t)
    workload(t)
    fatigue(t)
    dominance(t)
    engagement(t)

Design Goals
------------

✓ Continuous trajectories

✓ Forecastable dynamics

✓ Subject-specific variability

✓ TSST-driven stress response

✓ Realistic cross-label correlations

✓ Dataset agnostic

✓ Label agnostic

✓ Modality agnostic

✓ Long-duration clips

Notes
-----

Trajectories are modeled using:

    • Ornstein-Uhlenbeck dynamics
    • TSST phase effects
    • Subject profile effects
    • Slow emotion drift
    • Recovery dynamics

These trajectories will later generate:

    physiology
    sensor observations
    continuous labels
    discrete labels
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from datasets.generators.subject_model import (
    SubjectProfile,
)

from datasets.generators.tsst_protocol import (
    TSSTProtocolOutput,
)


# ---------------------------------------------------------
# Output Structure
# ---------------------------------------------------------

@dataclass
class LatentStateOutput:
    """
    Hidden human state trajectories.
    """

    valence: torch.Tensor
    arousal: torch.Tensor

    stress: torch.Tensor
    workload: torch.Tensor

    fatigue: torch.Tensor

    dominance: torch.Tensor
    engagement: torch.Tensor


# ---------------------------------------------------------
# Generator
# ---------------------------------------------------------

class LatentStateGenerator:
    """
    Generate continuous latent trajectories.

    Parameters
    ----------
    sampling_rate : int

        Label frequency.

    device : str
    """

    def __init__(
        self,
        sampling_rate: int = 25,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:

        self.sampling_rate = sampling_rate
        self.device = torch.device(device)

        self.rng = np.random.default_rng(seed)

    # -----------------------------------------------------
    # OU Process
    # -----------------------------------------------------

    def _ou_process(
        self,
        length: int,
        theta: float,
        mu: float,
        sigma: float,
    ) -> torch.Tensor:
        """
        Ornstein-Uhlenbeck process.

        Creates realistic continuous trajectories
        with temporal persistence.
        """

        x = np.zeros(length)

        dt = 1.0 / self.sampling_rate

        for t in range(1, length):

            dx = (
                theta *
                (mu - x[t - 1]) *
                dt
            )

            dx += (
                sigma *
                np.sqrt(dt) *
                self.rng.normal()
            )

            x[t] = x[t - 1] + dx

        return torch.tensor(
            x,
            dtype=torch.float32,
            device=self.device,
        )

    # -----------------------------------------------------
    # Utility
    # -----------------------------------------------------

    @staticmethod
    def _normalize(
        x: torch.Tensor,
        low: float = -1.0,
        high: float = 1.0,
    ) -> torch.Tensor:

        xmin = x.min()
        xmax = x.max()

        x = (
            (x - xmin)
            /
            (xmax - xmin + 1e-6)
        )

        x = low + (
            high - low
        ) * x

        return x

    # -----------------------------------------------------
    # Main
    # -----------------------------------------------------

    def generate(
        self,
        subject: SubjectProfile,
        protocol: TSSTProtocolOutput,
    ) -> LatentStateOutput:
        """
        Generate latent trajectories.

        Parameters
        ----------
        subject : SubjectProfile

        protocol : TSSTProtocolOutput

        Returns
        -------
        LatentStateOutput
        """

        N = protocol.stress_load.shape[0]

        # -------------------------------------------------
        # Base stochastic components
        # -------------------------------------------------

        v_base = self._ou_process(
            N,
            theta=0.2,
            mu=subject.baseline_valence,
            sigma=0.15,
        )

        a_base = self._ou_process(
            N,
            theta=0.25,
            mu=subject.baseline_arousal,
            sigma=0.20,
        )

        # -------------------------------------------------
        # Stress
        # -------------------------------------------------

        stress = (
            protocol.stress_load
            * subject.stress_gain
        )

        stress += (
            0.15 *
            self._ou_process(
                N,
                theta=0.30,
                mu=0.0,
                sigma=0.20,
            )
        )

        stress = self._normalize(
            stress,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Workload
        # -------------------------------------------------

        workload = (
            protocol.workload_load
            * subject.workload_gain
        )

        workload += (
            0.10 *
            stress
        )

        workload = self._normalize(
            workload,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Fatigue
        # -------------------------------------------------

        fatigue = (
            protocol.fatigue_load
            * subject.fatigue_gain
        )

        fatigue += 0.3 * workload

        fatigue = self._normalize(
            fatigue,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Arousal
        # -------------------------------------------------

        arousal = (
            a_base +
            0.5 * stress +
            0.2 * workload
        )

        arousal = torch.tanh(
            arousal
        )

        # -------------------------------------------------
        # Valence
        # -------------------------------------------------

        valence = (
            v_base
            - 0.6 * stress
            - 0.2 * fatigue
        )

        valence = torch.tanh(
            valence
        )

        # -------------------------------------------------
        # Dominance
        # -------------------------------------------------

        dominance = (
            0.8
            - 0.7 * stress
            - 0.3 * workload
        )

        dominance = torch.clamp(
            dominance,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Engagement
        # -------------------------------------------------

        engagement = (
            0.5 * arousal
            + 0.4 * workload
            - 0.3 * fatigue
        )

        engagement = torch.tanh(
            engagement
        )

        return LatentStateOutput(
            valence=valence,
            arousal=arousal,
            stress=stress,
            workload=workload,
            fatigue=fatigue,
            dominance=dominance,
            engagement=engagement,
        )