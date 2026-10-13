# datasets/generators/subject_model.py

"""
Subject Model Generator
=======================

Generates realistic subject-specific variability
for multimodal human-state simulation.

The goal is to emulate inter-subject differences
observed in affective computing datasets such as:

    RECOLA
    SEWA
    MuSe
    WESAD
    DEAP

Each subject receives a unique physiological,
cognitive and emotional profile.

These subject parameters are later used by:

    latent_state_generator.py

    physiological_generator.py

    sensor_generator.py

to create realistic subject-specific behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SubjectProfile:
    """
    Synthetic subject profile.
    """

    subject_id: int

    baseline_valence: float
    baseline_arousal: float

    stress_gain: float
    workload_gain: float
    fatigue_gain: float

    recovery_rate: float

    resting_heart_rate: float
    resting_eda: float

    eeg_alpha_gain: float
    eeg_beta_gain: float

    voice_pitch_base: float

    blink_rate_base: float


class SubjectModelGenerator:
    """
    Generator of synthetic subject profiles.
    """

    def __init__(
        self,
        seed: int = 42,
    ) -> None:

        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        subject_id: int,
    ) -> SubjectProfile:
        """
        Generate one realistic synthetic subject.
        """

        return SubjectProfile(

            subject_id=subject_id,

            baseline_valence=float(
                self.rng.normal(0.0, 0.25)
            ),

            baseline_arousal=float(
                self.rng.normal(0.3, 0.15)
            ),

            stress_gain=float(
                self.rng.uniform(0.8, 1.5)
            ),

            workload_gain=float(
                self.rng.uniform(0.8, 1.5)
            ),

            fatigue_gain=float(
                self.rng.uniform(0.8, 1.5)
            ),

            recovery_rate=float(
                self.rng.uniform(0.5, 1.3)
            ),

            resting_heart_rate=float(
                self.rng.normal(70, 5)
            ),

            resting_eda=float(
                self.rng.normal(0.30, 0.05)
            ),

            eeg_alpha_gain=float(
                self.rng.uniform(0.8, 1.3)
            ),

            eeg_beta_gain=float(
                self.rng.uniform(0.8, 1.3)
            ),

            voice_pitch_base=float(
                self.rng.normal(
                    140,
                    25,
                )
            ),

            blink_rate_base=float(
                self.rng.normal(
                    18,
                    4,
                )
            ),
        )