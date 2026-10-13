# datasets/generators/tsst_protocol.py

"""
TSST Protocol Generator
=======================

Synthetic Trier Social Stress Test (TSST)
timeline generator.

This module creates realistic stress-inducing
experimental phases commonly used in:

    - ULM-TSST
    - stress assessment studies
    - affective computing
    - workload estimation
    - physiological computing

The protocol itself does not generate emotions,
physiology or sensor signals.

Instead it produces latent protocol-driven loads:

    stress_load(t)
    workload_load(t)
    fatigue_load(t)

that are consumed by later generators.

Design Goals
------------

✓ Dataset agnostic

✓ Modality agnostic

✓ Label agnostic

✓ Time-series compatible

✓ Forecasting compatible

✓ Continuous trajectory generation

✓ Long duration recordings

Output
------

The protocol returns:

    phase_ids
    stress_load
    workload_load
    fatigue_load

which become the causal drivers
of the complete multimodal simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# ---------------------------------------------------------
# Phase Definitions
# ---------------------------------------------------------

PHASE_BASELINE = 0
PHASE_PREPARATION = 1
PHASE_SPEECH = 2
PHASE_ARITHMETIC = 3
PHASE_RECOVERY = 4


@dataclass
class TSSTProtocolOutput:
    """
    Protocol output trajectories.
    """

    time: torch.Tensor

    phase_ids: torch.Tensor

    stress_load: torch.Tensor

    workload_load: torch.Tensor

    fatigue_load: torch.Tensor


class TSSTProtocolGenerator:
    """
    Generate realistic TSST protocol trajectories.

    Parameters
    ----------
    duration_seconds : int

        Total simulation duration.

    sampling_rate : int

        Temporal label rate.

        Example:

            25 Hz

    device : str
    """

    def __init__(
        self,
        duration_seconds: int = 1200,
        sampling_rate: int = 25,
        device: str = "cpu",
    ) -> None:

        self.duration_seconds = duration_seconds
        self.sampling_rate = sampling_rate
        self.device = torch.device(device)

        self.num_steps = (
            duration_seconds *
            sampling_rate
        )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _phase_boundaries(self):
        """
        TSST phase proportions.

        20 min example:

            baseline      20%
            preparation   10%
            speech        25%
            arithmetic    25%
            recovery      20%
        """

        N = self.num_steps

        return (
            int(0.20 * N),
            int(0.30 * N),
            int(0.55 * N),
            int(0.80 * N),
            N,
        )

    # ---------------------------------------------------------

    def generate(
        self,
    ) -> TSSTProtocolOutput:
        """
        Generate protocol.
        """

        N = self.num_steps

        t = torch.linspace(
            0,
            self.duration_seconds,
            N,
            device=self.device,
        )

        phase_ids = torch.zeros(
            N,
            dtype=torch.long,
            device=self.device,
        )

        stress = torch.zeros(
            N,
            device=self.device,
        )

        workload = torch.zeros(
            N,
            device=self.device,
        )

        fatigue = torch.zeros(
            N,
            device=self.device,
        )

        (
            prep_start,
            speech_start,
            arithmetic_start,
            recovery_start,
            end_idx,
        ) = self._phase_boundaries()

        # -------------------------------------------------
        # Baseline
        # -------------------------------------------------

        phase_ids[:prep_start] = PHASE_BASELINE

        stress[:prep_start] = 0.15
        workload[:prep_start] = 0.10
        fatigue[:prep_start] = 0.10

        # -------------------------------------------------
        # Preparation
        # -------------------------------------------------

        phase_ids[
            prep_start:speech_start
        ] = PHASE_PREPARATION

        stress[
            prep_start:speech_start
        ] = torch.linspace(
            0.20,
            0.60,
            speech_start - prep_start,
            device=self.device,
        )

        workload[
            prep_start:speech_start
        ] = torch.linspace(
            0.20,
            0.50,
            speech_start - prep_start,
            device=self.device,
        )

        fatigue[
            prep_start:speech_start
        ] = torch.linspace(
            0.10,
            0.25,
            speech_start - prep_start,
            device=self.device,
        )

        # -------------------------------------------------
        # Speech
        # -------------------------------------------------

        phase_ids[
            speech_start:arithmetic_start
        ] = PHASE_SPEECH

        stress[
            speech_start:arithmetic_start
        ] = 0.80

        workload[
            speech_start:arithmetic_start
        ] = 0.65

        fatigue[
            speech_start:arithmetic_start
        ] = torch.linspace(
            0.25,
            0.55,
            arithmetic_start - speech_start,
            device=self.device,
        )

        # -------------------------------------------------
        # Arithmetic
        # -------------------------------------------------

        phase_ids[
            arithmetic_start:recovery_start
        ] = PHASE_ARITHMETIC

        stress[
            arithmetic_start:recovery_start
        ] = 0.95

        workload[
            arithmetic_start:recovery_start
        ] = 1.00

        fatigue[
            arithmetic_start:recovery_start
        ] = torch.linspace(
            0.55,
            0.85,
            recovery_start - arithmetic_start,
            device=self.device,
        )

        # -------------------------------------------------
        # Recovery
        # -------------------------------------------------

        phase_ids[
            recovery_start:
        ] = PHASE_RECOVERY

        stress[
            recovery_start:
        ] = torch.linspace(
            0.60,
            0.20,
            end_idx - recovery_start,
            device=self.device,
        )

        workload[
            recovery_start:
        ] = torch.linspace(
            0.40,
            0.10,
            end_idx - recovery_start,
            device=self.device,
        )

        fatigue[
            recovery_start:
        ] = torch.linspace(
            0.85,
            0.65,
            end_idx - recovery_start,
            device=self.device,
        )

        return TSSTProtocolOutput(
            time=t,
            phase_ids=phase_ids,
            stress_load=stress,
            workload_load=workload,
            fatigue_load=fatigue,
        )