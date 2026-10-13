"""
physiological_generator.py
==========================

Physiological State Generator

Transforms latent human-state trajectories into
hidden physiological trajectories.

Purpose
-------

The latent state is not directly observed.

People experience:

    stress
    arousal
    workload
    fatigue

which influence physiological systems:

    cardiovascular system
    autonomic nervous system
    respiratory system
    muscular activation
    ocular behavior
    cortical activity

Those physiological processes eventually create
the observable sensor signals.

Pipeline
--------

TSST
  ↓

Latent State
  ↓

Physiology
  ↓

Sensors

Generated Variables
-------------------

heart_rate

heart_rate_variability

sympathetic_activation

parasympathetic_activation

respiration_rate

muscle_tension

pupil_size

cortical_activation

Design Goals
------------

✓ Physiologically plausible

✓ Subject specific

✓ Correlated modalities

✓ Forecastable

✓ Continuous trajectories

✓ Dataset agnostic

✓ Modality agnostic

✓ Label agnostic
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from datasets.generators.subject_model import (
    SubjectProfile,
)

from datasets.generators.latent_state_generator import (
    LatentStateOutput,
)


# ==========================================================
# Output
# ==========================================================

@dataclass
class PhysiologicalStateOutput:
    """
    Hidden physiological trajectories.
    """

    heart_rate: torch.Tensor

    hrv: torch.Tensor

    sympathetic: torch.Tensor

    parasympathetic: torch.Tensor

    respiration_rate: torch.Tensor

    muscle_tension: torch.Tensor

    pupil_size: torch.Tensor

    cortical_activation: torch.Tensor


# ==========================================================
# Generator
# ==========================================================

class PhysiologicalGenerator:
    """
    Generate physiological state trajectories.
    """

    def __init__(
        self,
        device: str = "cpu",
    ):

        self.device = torch.device(device)

    # ------------------------------------------------------
    # Utilities
    # ------------------------------------------------------

    @staticmethod
    def _normalize_01(
        x: torch.Tensor,
    ) -> torch.Tensor:

        xmin = x.min()

        xmax = x.max()

        return (
            x - xmin
        ) / (
            xmax - xmin + 1e-6
        )

    # ------------------------------------------------------
    # Main
    # ------------------------------------------------------

    def generate(
        self,
        latent: LatentStateOutput,
        subject: SubjectProfile,
    ) -> PhysiologicalStateOutput:
        """
        Generate physiology.

        Parameters
        ----------
        latent : LatentStateOutput

        subject : SubjectProfile

        Returns
        -------
        PhysiologicalStateOutput
        """

        stress = latent.stress
        workload = latent.workload
        fatigue = latent.fatigue
        arousal = latent.arousal

        # --------------------------------------------------
        # Autonomic nervous system
        # --------------------------------------------------

        sympathetic = (
            0.6 * stress
            + 0.3 * workload
            + 0.2 * arousal
        )

        sympathetic = self._normalize_01(
            sympathetic
        )

        parasympathetic = (
            1.0
            - sympathetic
        )

        parasympathetic = torch.clamp(
            parasympathetic,
            0.0,
            1.0,
        )

        # --------------------------------------------------
        # Heart Rate
        # --------------------------------------------------

        heart_rate = (
            subject.resting_heart_rate
            +
            15.0 * stress
            +
            8.0 * workload
            +
            5.0 * arousal
        )

        heart_rate += (
            2.0 *
            torch.sin(
                torch.linspace(
                    0,
                    20,
                    len(stress),
                    device=stress.device,
                )
            )
        )

        heart_rate = torch.clamp(
            heart_rate,
            50.0,
            180.0,
        )

        # --------------------------------------------------
        # Heart Rate Variability
        # --------------------------------------------------

        hrv = (
            1.0
            - sympathetic
        )

        hrv = (
            30.0
            + 70.0 * hrv
        )

        # --------------------------------------------------
        # Respiration
        # --------------------------------------------------

        respiration_rate = (
            12
            +
            6 * stress
            +
            4 * arousal
        )

        respiration_rate = torch.clamp(
            respiration_rate,
            8,
            35,
        )

        # --------------------------------------------------
        # Muscle Tension
        # --------------------------------------------------

        muscle_tension = (
            0.5 * stress
            +
            0.4 * workload
        )

        muscle_tension = torch.clamp(
            muscle_tension,
            0.0,
            1.0,
        )

        # --------------------------------------------------
        # Pupil Dilation
        # --------------------------------------------------

        pupil_size = (
            3.5
            +
            2.0 * stress
            +
            1.0 * workload
        )

        pupil_size = torch.clamp(
            pupil_size,
            2.5,
            8.0,
        )

        # --------------------------------------------------
        # Cortical Activation
        # --------------------------------------------------

        cortical_activation = (
            0.5 * workload
            +
            0.3 * arousal
            -
            0.2 * fatigue
        )

        cortical_activation = torch.clamp(
            cortical_activation,
            0.0,
            1.0,
        )

        return PhysiologicalStateOutput(

            heart_rate=heart_rate,

            hrv=hrv,

            sympathetic=sympathetic,

            parasympathetic=parasympathetic,

            respiration_rate=respiration_rate,

            muscle_tension=muscle_tension,

            pupil_size=pupil_size,

            cortical_activation=cortical_activation,
        )