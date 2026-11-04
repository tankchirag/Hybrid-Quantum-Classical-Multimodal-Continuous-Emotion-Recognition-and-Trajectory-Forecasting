"""
base_signal_validator.py
========================

Base Signal Validation Framework

Purpose
-------
Provides shared infrastructure for all physiological
signal validation modules.

This class centralizes:

    • signal extraction
    • window selection
    • time-axis generation
    • moving averages
    • summary statistics
    • emotion-state extraction
    • JSON export
    • output directory handling

Derived Validators
------------------

    ECGValidator
    EDAValidator
    EMGValidator

Design Goals
------------

✓ Dataset agnostic

✓ Modality agnostic

✓ Duration agnostic

✓ Sampling-rate agnostic

✓ Thesis-grade

✓ Reusable

✓ Extensible

✓ Strong typing

✓ Minimal code duplication

Architecture
------------

BaseSignalValidator
        ↑
        │
 ┌──────┼──────┐
 │      │      │
ECG    EDA    EMG

Example
-------

class ECGValidator(
    BaseSignalValidator
):

    MODALITY_NAME = "ecg"

    def validate(...):
        ...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Tuple

import numpy as np
import torch


# ==========================================================
# Base Validator
# ==========================================================

class BaseSignalValidator:
    """
    Base physiological signal validator.
    """

    DEFAULT_TRACKED_LABELS = [

        "valence",
        "arousal",
        "dominance",

        "stress",
        "workload",
        "fatigue",

        "engagement",
    ]

    # ------------------------------------------------------
    # Constructor
    # ------------------------------------------------------

    def __init__(
        self,
        window_seconds: float = 60.0,
        moving_average_window: int = 100,
    ) -> None:
        """
        Parameters
        ----------
        window_seconds

            Visualization duration.

            Example:
                60 sec

        moving_average_window

            Smoothing window.

            Example:
                100 samples
        """

        self.window_seconds = (
            window_seconds
        )

        self.moving_average_window = (
            moving_average_window
        )

    # ======================================================
    # Tensor Utilities
    # ======================================================

    @staticmethod
    def to_numpy(
        x: torch.Tensor | np.ndarray,
    ) -> np.ndarray:
        """
        Convert signal to NumPy.
        """

        if isinstance(
            x,
            torch.Tensor,
        ):
            return (
                x.detach()
                .cpu()
                .numpy()
            )

        return np.asarray(x)

    # ======================================================
    # Time Utilities
    # ======================================================

    @staticmethod
    def create_time_axis(
        num_samples: int,
        sampling_rate: int,
    ) -> np.ndarray:
        """
        Build time axis in seconds.
        """

        return (
            np.arange(
                num_samples
            )
            /
            float(
                sampling_rate
            )
        )

    # ======================================================
    # Window Selection
    # ======================================================

    def get_middle_window(
        self,
        total_samples: int,
        sampling_rate: int,
    ) -> Tuple[int, int]:
        """
        Extract middle interval.

        Works for arbitrary duration clips.

        Returns
        -------
        (start_idx, end_idx)
        """

        window_size = int(
            self.window_seconds
            * sampling_rate
        )

        center = (
            total_samples // 2
        )

        start = max(
            0,
            center
            - window_size // 2,
        )

        end = min(
            total_samples,
            start + window_size,
        )

        return start, end

    # ======================================================
    # Signal Processing
    # ======================================================

    @staticmethod
    def moving_average(
        signal: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """
        Moving average smoother.
        """

        if len(signal) < window:
            return signal

        kernel = (
            np.ones(window)
            / float(window)
        )

        return np.convolve(
            signal,
            kernel,
            mode="same",
        )

    # ------------------------------------------------------

    @staticmethod
    def rms_envelope(
        signal: np.ndarray,
        window: int = 256,
    ) -> np.ndarray:
        """
        Root-mean-square envelope.

        Primarily used for EMG.
        """

        signal = np.square(signal)

        kernel = (
            np.ones(window)
            / float(window)
        )

        rms = np.sqrt(
            np.convolve(
                signal,
                kernel,
                mode="same",
            )
        )

        return rms

    # ======================================================
    # Statistics
    # ======================================================

    @staticmethod
    def signal_statistics(
        signal: np.ndarray,
    ) -> Dict[str, float]:
        """
        Common signal statistics.
        """

        return {

            "mean":
                float(
                    np.mean(signal)
                ),

            "std":
                float(
                    np.std(signal)
                ),

            "min":
                float(
                    np.min(signal)
                ),

            "max":
                float(
                    np.max(signal)
                ),

            "median":
                float(
                    np.median(signal)
                ),

            "range":
                float(
                    np.max(signal)
                    -
                    np.min(signal)
                ),
        }

    # ======================================================
    # Emotion Analysis
    # ======================================================

    def emotion_summary(
        self,
        sample: Dict[str, Any],
        start_sec: float,
        end_sec: float,
    ) -> Dict[str, float]:
        """
        Extract emotional state for
        selected signal interval.
        """

        labels = sample[
            "continuous_labels"
        ]

        metadata = sample[
            "metadata"
        ]

        duration = float(
            metadata[
                "duration_seconds"
            ]
        )

        summary = {}

        for label_name in (
            self
            .DEFAULT_TRACKED_LABELS
        ):

            if label_name not in labels:
                continue

            values = (
                self.to_numpy(
                    labels[
                        label_name
                    ]
                )
            )

            label_fs = (
                len(values)
                /
                duration
            )

            start_idx = int(
                start_sec
                * label_fs
            )

            end_idx = int(
                end_sec
                * label_fs
            )

            interval = (
                values[
                    start_idx:end_idx
                ]
            )

            summary[
                f"mean_{label_name}"
            ] = float(
                np.mean(interval)
            )

        return summary

    # ======================================================
    # Signal Extraction
    # ======================================================

    def extract_window(
        self,
        sample: Dict[str, Any],
        modality_name: str,
        sampling_rate: int,
    ) -> Dict[str, Any]:
        """
        Extract windowed signal segment.
        """

        signal = self.to_numpy(
            sample[
                "modalities"
            ][
                modality_name
            ]
        ).squeeze()

        start_idx, end_idx = (
            self.get_middle_window(
                len(signal),
                sampling_rate,
            )
        )

        segment = signal[
            start_idx:end_idx
        ]

        start_sec = (
            start_idx
            /
            sampling_rate
        )

        end_sec = (
            end_idx
            /
            sampling_rate
        )

        time_axis = (
            np.linspace(
                start_sec,
                end_sec,
                len(segment),
            )
        )

        return {

            "signal":
                segment,

            "time_axis":
                time_axis,

            "start_idx":
                start_idx,

            "end_idx":
                end_idx,

            "start_sec":
                start_sec,

            "end_sec":
                end_sec,
        }

    # ======================================================
    # Output
    # ======================================================

    @staticmethod
    def ensure_output_dir(
        output_dir: str | Path,
    ) -> Path:
        """
        Create output directory.
        """

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return output_dir

    # ------------------------------------------------------

    @staticmethod
    def save_json(
        data: Dict[str, Any],
        output_file:
        str | Path,
    ) -> None:
        """
        Save JSON artifact.
        """

        output_file = Path(
            output_file
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as fp:

            json.dump(
                data,
                fp,
                indent=4,
                ensure_ascii=False,
            )

    # ======================================================
    # Validation Summary
    # ======================================================

    def build_summary(
        self,
        sample: Dict[str, Any],
        signal: np.ndarray,
        start_sec: float,
        end_sec: float,
    ) -> Dict[str, Any]:
        """
        Generic validation summary.
        """

        return {

            "interval": {

                "start_seconds":
                    float(
                        start_sec
                    ),

                "end_seconds":
                    float(
                        end_sec
                    ),

                "duration_seconds":
                    float(
                        end_sec
                        - start_sec
                    ),
            },

            "signal_statistics":
                self.signal_statistics(
                    signal
                ),

            "emotion_summary":
                self.emotion_summary(
                    sample,
                    start_sec,
                    end_sec,
                ),
        }
