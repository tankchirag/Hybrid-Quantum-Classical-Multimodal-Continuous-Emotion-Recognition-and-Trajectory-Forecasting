"""
ecg_validation.py
=================

ECG Validation Module

Purpose
-------
Visual validation of generated ECG signals.

Supervisor Requirements
-----------------------

✓ Raw ECG

✓ ECG trend

✓ Configurable validation interval

✓ Emotion-state summary

✓ Publication-quality figure

Architecture
------------

BaseSignalValidator
        ↑
        │
   ECGValidator

Notes
-----

This module intentionally contains ONLY ECG-
specific logic.

Generic functionality lives in:

    base_signal_validator.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from datasets.evaluation.base_signal_validator import (
    BaseSignalValidator
)


# ==========================================================
# ECG Validator
# ==========================================================

class ECGValidator(
    BaseSignalValidator
):
    """
    ECG validation utility.
    """

    MODALITY_NAME = "ecg"

    # ------------------------------------------------------
    # Constructor
    # ------------------------------------------------------

    def __init__(
        self,
        sampling_rate: int = 1000,
        window_seconds: float = 60.0,
        moving_average_window: int = 100,
        dpi: int = 300,
    ) -> None:

        super().__init__(
            window_seconds=
            window_seconds,

            moving_average_window=
            moving_average_window,
        )

        self.sampling_rate = (
            sampling_rate
        )

        self.dpi = dpi

    # ======================================================
    # ECG Processing
    # ======================================================

    def compute_trend(
        self,
        ecg: np.ndarray,
    ) -> np.ndarray:
        """
        ECG trend using moving average.

        Supervisor requested:

            Raw ECG
            +
            100-sample mean
        """

        return self.moving_average(
            ecg,
            self.moving_average_window,
        )

    # ======================================================
    # Plot
    # ======================================================

    def generate_plot(
        self,
        time_axis: np.ndarray,
        raw_ecg: np.ndarray,
        trend: np.ndarray,
        summary: Dict[str, Any],
        output_file: Path,
    ) -> None:
        """
        Generate ECG validation figure.
        """

        fig = plt.figure(
            figsize=(14, 8)
        )

        fig.suptitle(
            "ECG Validation",
            fontsize=14,
            fontweight="bold",
        )

        # ----------------------------------
        # Raw ECG
        # ----------------------------------

        ax1 = plt.subplot(
            2,
            1,
            1,
        )

        ax1.plot(
            time_axis,
            raw_ecg,
            linewidth=0.6,
            color="tab:blue",
        )

        ax1.set_title(
            "Raw ECG Signal"
        )

        ax1.set_ylabel(
            "Amplitude"
        )

        ax1.grid(
            alpha=0.3
        )

        # ----------------------------------
        # Trend
        # ----------------------------------

        ax2 = plt.subplot(
            2,
            1,
            2,
        )

        ax2.plot(
            time_axis,
            trend,
            linewidth=1.0,
            color="tab:orange",
        )

        ax2.set_title(
            (
                "ECG Moving Average "
                f"({self.moving_average_window})"
            )
        )

        ax2.set_xlabel(
            "Time (seconds)"
        )

        ax2.set_ylabel(
            "Amplitude"
        )

        ax2.grid(
            alpha=0.3
        )

        # ----------------------------------
        # Emotion Summary
        # ----------------------------------

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
    # Main API
    # ======================================================

    def validate(
        self,
        sample: Dict[str, Any],
        output_dir: str | Path,
    ) -> Dict[str, Any]:
        """
        Generate ECG validation artifacts.

        Outputs
        -------

        ecg_validation.png

        ecg_summary.json
        """

        output_dir = (
            self.ensure_output_dir(
                output_dir
            )
        )

        window = (
            self.extract_window(
                sample=sample,

                modality_name=
                self.MODALITY_NAME,

                sampling_rate=
                self.sampling_rate,
            )
        )

        raw_ecg = window["signal"]

        trend = (
            self.compute_trend(
                raw_ecg
            )
        )

        summary = (
            self.build_summary(
                sample=sample,

                signal=raw_ecg,

                start_sec=
                window["start_sec"],

                end_sec=
                window["end_sec"],
            )
        )

        self.generate_plot(
            time_axis=
            window["time_axis"],

            raw_ecg=
            raw_ecg,

            trend=
            trend,

            summary=
            summary,

            output_file=
            output_dir
            / "ecg_validation.png",
        )

        self.save_json(
            summary,
            output_dir
            / "ecg_summary.json",
        )

        return summary