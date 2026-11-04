"""
eda_validation.py
=================

Electrodermal Activity (EDA) Validation

Purpose
-------
Validate generated EDA signals and produce
physiological validation figures requested by
the supervisor.

Validation Outputs
------------------

1. Raw EDA

2. SCL
   (Skin Conductance Level)

3. SCR
   (Skin Conductance Response)

4. Emotional State Summary

5. JSON Statistics

Supervisor Requirements
-----------------------

Show:

    Raw EDA

    SCL

    SCR

for selected recording interval.

Architecture
------------

BaseSignalValidator
        ↑
        │
    EDAValidator

Notes
-----

Current dataset generator creates:

    tonic + phasic

EDA signal.

This validator decomposes EDA into:

    SCL = low-frequency tonic component

    SCR = residual phasic component
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from datasets.evaluation.base_signal_validator import (
    BaseSignalValidator,
)


# ==========================================================
# EDA Validator
# ==========================================================

class EDAValidator(
    BaseSignalValidator
):
    """
    Electrodermal activity validation.
    """

    MODALITY_NAME = "eda"

    # ------------------------------------------------------
    # Constructor
    # ------------------------------------------------------

    def __init__(
        self,
        sampling_rate: int = 128,
        window_seconds: float = 60.0,
        scl_window_seconds: float = 5.0,
        dpi: int = 300,
    ) -> None:

        super().__init__(
            window_seconds=
            window_seconds
        )

        self.sampling_rate = (
            sampling_rate
        )

        self.scl_window_seconds = (
            scl_window_seconds
        )

        self.dpi = dpi

    # ======================================================
    # EDA Decomposition
    # ======================================================

    def compute_scl(
        self,
        eda: np.ndarray,
    ) -> np.ndarray:
        """
        Compute tonic component.

        SCL = Skin Conductance Level
        """

        window = int(
            self.scl_window_seconds
            * self.sampling_rate
        )

        window = max(
            3,
            window,
        )

        return self.moving_average(
            eda,
            window,
        )

    # ------------------------------------------------------

    @staticmethod
    def compute_scr(
        eda: np.ndarray,
        scl: np.ndarray,
    ) -> np.ndarray:
        """
        Compute phasic component.

        SCR = Raw EDA - SCL
        """

        return eda - scl

    # ======================================================
    # Summary
    # ======================================================

    def build_eda_summary(
        self,
        sample: Dict[str, Any],
        raw_eda: np.ndarray,
        scl: np.ndarray,
        scr: np.ndarray,
        start_sec: float,
        end_sec: float,
    ) -> Dict[str, Any]:

        summary = self.build_summary(
            sample=sample,
            signal=raw_eda,
            start_sec=start_sec,
            end_sec=end_sec,
        )

        summary["scl_statistics"] = (
            self.signal_statistics(
                scl
            )
        )

        summary["scr_statistics"] = (
            self.signal_statistics(
                scr
            )
        )

        return summary

    # ======================================================
    # Visualization
    # ======================================================

    def generate_plot(
        self,
        time_axis: np.ndarray,
        raw_eda: np.ndarray,
        scl: np.ndarray,
        scr: np.ndarray,
        summary: Dict[str, Any],
        output_file: Path,
    ) -> None:
        """
        Generate EDA validation figure.
        """

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(14, 10),
            sharex=True,
        )

        fig.suptitle(
            "EDA Validation",
            fontsize=14,
            fontweight="bold",
        )

        # ---------------------------------
        # Raw EDA
        # ---------------------------------

        axes[0].plot(
            time_axis,
            raw_eda,
            color="tab:blue",
            linewidth=0.8,
        )

        axes[0].set_title(
            "Raw EDA"
        )

        axes[0].set_ylabel(
            "Amplitude"
        )

        axes[0].grid(
            alpha=0.3
        )

        # ---------------------------------
        # SCL
        # ---------------------------------

        axes[1].plot(
            time_axis,
            scl,
            color="tab:green",
            linewidth=1.2,
        )

        axes[1].set_title(
            "SCL (Tonic Component)"
        )

        axes[1].set_ylabel(
            "Conductance"
        )

        axes[1].grid(
            alpha=0.3
        )

        # ---------------------------------
        # SCR
        # ---------------------------------

        axes[2].plot(
            time_axis,
            scr,
            color="tab:red",
            linewidth=0.8,
        )

        axes[2].set_title(
            "SCR (Phasic Component)"
        )

        axes[2].set_xlabel(
            "Time (seconds)"
        )

        axes[2].set_ylabel(
            "Conductance"
        )

        axes[2].grid(
            alpha=0.3
        )

        # ---------------------------------
        # Emotion Summary
        # ---------------------------------

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
        Generate EDA validation artifacts.

        Outputs
        -------

        eda_validation.png

        eda_summary.json
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

        raw_eda = (
            window["signal"]
        )

        scl = self.compute_scl(
            raw_eda
        )

        scr = self.compute_scr(
            raw_eda,
            scl,
        )

        summary = (
            self.build_eda_summary(
                sample=sample,
                raw_eda=raw_eda,
                scl=scl,
                scr=scr,
                start_sec=
                window["start_sec"],
                end_sec=
                window["end_sec"],
            )
        )

        self.generate_plot(
            time_axis=
            window["time_axis"],

            raw_eda=
            raw_eda,

            scl=scl,

            scr=scr,

            summary=
            summary,

            output_file=
            output_dir
            / "eda_validation.png",
        )

        self.save_json(
            summary,
            output_dir
            / "eda_summary.json",
        )

        return summary