"""
emg_validation.py
=================

Electromyography (EMG) Validation Framework

Purpose
-------
Validate synthetic EMG signals and generate
supervisor-requested physiological validation
artifacts.

Validation Outputs
------------------

1. Raw EMG Signal

2. RMS Envelope

3. Emotional State Summary

4. EMG Statistics

5. Validation Figure

6. Validation JSON

Supervisor Motivation
---------------------

The EMG validation figure should answer:

    Does generated muscle activity reflect
    the generated emotional state?

Example Expectations
--------------------

High Stress
    -> Higher muscle tension
    -> Higher EMG amplitude
    -> Higher RMS envelope

Low Stress
    -> Lower muscle tension
    -> Lower RMS envelope

Architecture
------------

BaseSignalValidator
        ↑
        │
    EMGValidator

Responsibilities
----------------

This module contains ONLY EMG-specific logic.

Shared functionality comes from:

    base_signal_validator.py

Design Goals
------------

✓ Sampling-rate agnostic

✓ Dataset agnostic

✓ Configurable interval

✓ Publication-quality figures

✓ Minimal code duplication

✓ Reusable

Example
-------

validator = EMGValidator()

summary = validator.validate(
    sample=sample,
    output_dir="results/sample_000"
)

Outputs

    emg_validation.png

    emg_summary.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from datasets.evaluation.base_signal_validator import (
    BaseSignalValidator,
)


# ==========================================================
# EMG Validator
# ==========================================================

class EMGValidator(
    BaseSignalValidator
):
    """
    Electromyography validation utility.
    """

    MODALITY_NAME = "emg"

    # ------------------------------------------------------
    # Constructor
    # ------------------------------------------------------

    def __init__(
        self,
        sampling_rate: int = 1000,
        window_seconds: float = 60.0,
        envelope_window: int = 256,
        dpi: int = 300,
    ) -> None:
        """
        Parameters
        ----------
        sampling_rate : int

            EMG sampling frequency.

        window_seconds : float

            Validation interval duration.

        envelope_window : int

            RMS envelope window size.

        dpi : int

            Figure export resolution.
        """

        super().__init__(
            window_seconds=
            window_seconds
        )

        self.sampling_rate = (
            sampling_rate
        )

        self.envelope_window = (
            envelope_window
        )

        self.dpi = dpi

    # ======================================================
    # EMG Processing
    # ======================================================

    def compute_envelope(
        self,
        emg: np.ndarray,
    ) -> np.ndarray:
        """
        Compute RMS envelope.

        Uses implementation from
        BaseSignalValidator.
        """

        return self.rms_envelope(
            signal=emg,
            window=self.envelope_window,
        )

    # ======================================================
    # Summary
    # ======================================================

    def build_emg_summary(
        self,
        sample: Dict[str, Any],
        raw_emg: np.ndarray,
        envelope: np.ndarray,
        start_sec: float,
        end_sec: float,
    ) -> Dict[str, Any]:
        """
        Create EMG validation summary.
        """

        summary = self.build_summary(
            sample=sample,
            signal=raw_emg,
            start_sec=start_sec,
            end_sec=end_sec,
        )

        summary["envelope_statistics"] = (
            self.signal_statistics(
                envelope
            )
        )

        return summary

    # ======================================================
    # Figure Generation
    # ======================================================

    def generate_plot(
        self,
        time_axis: np.ndarray,
        raw_emg: np.ndarray,
        envelope: np.ndarray,
        summary: Dict[str, Any],
        output_file: Path,
    ) -> None:
        """
        Generate EMG validation figure.
        """

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(14, 8),
            sharex=True,
        )

        fig.suptitle(
            "EMG Validation",
            fontsize=14,
            fontweight="bold",
        )

        # --------------------------------------------------
        # Raw EMG
        # --------------------------------------------------

        axes[0].plot(
            time_axis,
            raw_emg,
            color="tab:blue",
            linewidth=0.5,
        )

        axes[0].set_title(
            "Raw EMG"
        )

        axes[0].set_ylabel(
            "Amplitude"
        )

        axes[0].grid(
            alpha=0.3
        )

        # --------------------------------------------------
        # RMS Envelope
        # --------------------------------------------------

        axes[1].plot(
            time_axis,
            envelope,
            color="tab:red",
            linewidth=1.0,
        )

        axes[1].set_title(
            (
                "RMS Envelope "
                f"(window={self.envelope_window})"
            )
        )

        axes[1].set_xlabel(
            "Time (seconds)"
        )

        axes[1].set_ylabel(
            "Amplitude"
        )

        axes[1].grid(
            alpha=0.3
        )

        # --------------------------------------------------
        # Emotion Summary
        # --------------------------------------------------

        emotion_text = "\n".join(
            [
                (
                    f"{k.replace('mean_', '')}: "
                    f"{v:.3f}"
                )
                for k, v in
                summary[
                    "emotion_summary"
                ].items()
            ]
        )

        fig.text(
            0.83,
            0.60,
            emotion_text,
            fontsize=8,
            bbox=dict(
                facecolor="white",
                alpha=0.95,
            ),
        )

        plt.tight_layout()

        plt.savefig(
            output_file,
            dpi=self.dpi,
            bbox_inches="tight",
        )

        plt.close()

    # ======================================================
    # Validation Pipeline
    # ======================================================

    def validate(
        self,
        sample: Dict[str, Any],
        output_dir: str | Path,
    ) -> Dict[str, Any]:
        """
        Generate EMG validation artifacts.

        Outputs
        -------

        emg_validation.png

        emg_summary.json
        """

        output_dir = (
            self.ensure_output_dir(
                output_dir
            )
        )

        window = self.extract_window(
            sample=sample,
            modality_name=
            self.MODALITY_NAME,
            sampling_rate=
            self.sampling_rate,
        )

        raw_emg = (
            window["signal"]
        )

        envelope = (
            self.compute_envelope(
                raw_emg
            )
        )

        summary = (
            self.build_emg_summary(
                sample=sample,
                raw_emg=raw_emg,
                envelope=envelope,
                start_sec=
                window["start_sec"],
                end_sec=
                window["end_sec"],
            )
        )

        self.generate_plot(
            time_axis=
            window["time_axis"],

            raw_emg=
            raw_emg,

            envelope=
            envelope,

            summary=
            summary,

            output_file=
            output_dir
            / "emg_validation.png",
        )

        self.save_json(
            summary,
            output_dir
            / "emg_summary.json",
        )

        return summary