"""
multimodal_alignment.py
=======================

Multimodal Alignment Visualization

Purpose
-------
Visualize synchronized relationships between:

    Emotional State

            ↓

    Physiological State

            ↓

    Sensor Signals

for the same time interval.

Motivation
----------

Supervisor Question:

    Did generated physiology respond
    to generated emotion?

This module produces a synchronized
multimodal timeline allowing rapid
visual inspection.

Example Relationships
---------------------

Stress ↑

    -> BPM ↑

    -> EDA ↑

    -> EMG ↑

Valence ↓

    -> Increased physiological stress

Outputs
-------

multimodal_alignment.png

Design Goals
------------

✓ Configurable window

✓ Dataset agnostic

✓ Signal agnostic

✓ Supervisor friendly

✓ Publication quality

✓ Fully synchronized

✓ No hardcoded clip duration
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
from typing import Any
from typing import List

import numpy as np
import torch
import matplotlib.pyplot as plt


# ==========================================================
# Multimodal Alignment Plotter
# ==========================================================

class MultimodalAlignment:
    """
    Synchronized multimodal visualization.
    """

    DEFAULT_EMOTIONS = [

        "valence",
        "arousal",
        "stress",
        "workload",

        "dominance",
        "engagement",
    ]

    DEFAULT_SIGNALS = [

        "bpm",
        "eda",
        "emg",
        "resp",
    ]

    # ------------------------------------------------------
    # Constructor
    # ------------------------------------------------------

    def __init__(
        self,
        window_seconds: float = 60.0,
        dpi: int = 300,
        figsize: tuple = (16, 14),
    ) -> None:

        self.window_seconds = (
            window_seconds
        )

        self.dpi = dpi

        self.figsize = figsize

    # ======================================================
    # Utilities
    # ======================================================

    @staticmethod
    def to_numpy(
        x,
    ) -> np.ndarray:

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

    # ------------------------------------------------------

    @staticmethod
    def normalize(
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize to [0,1].

        Allows different modalities
        to be visually comparable.
        """

        minimum = np.min(signal)
        maximum = np.max(signal)

        denom = (
            maximum - minimum
        )

        if denom < 1e-8:
            return np.zeros_like(
                signal
            )

        return (
            signal - minimum
        ) / denom

    # ======================================================
    # Window Extraction
    # ======================================================

    def _middle_window(
        self,
        signal: np.ndarray,
        sampling_rate: float,
    ):

        total_samples = len(
            signal
        )

        window_samples = int(
            self.window_seconds
            * sampling_rate
        )

        centre = (
            total_samples // 2
        )

        start = max(
            0,
            centre
            - window_samples // 2,
        )

        end = min(
            total_samples,
            start
            + window_samples,
        )

        return signal[
            start:end
        ]

    # ======================================================
    # Label Extraction
    # ======================================================

    def extract_labels(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:

        output = {}

        labels = sample[
            "continuous_labels"
        ]

        duration = (
            sample["metadata"]
            ["duration_seconds"]
        )

        for name in (
            self.DEFAULT_EMOTIONS
        ):

            if name not in labels:
                continue

            values = (
                self.to_numpy(
                    labels[name]
                )
            )

            fs = (
                len(values)
                / duration
            )

            output[name] = (
                self._middle_window(
                    values,
                    fs,
                )
            )

        return output

    # ======================================================
    # Signal Extraction
    # ======================================================

    def extract_signals(
        self,
        sample: Dict[str, Any],
        sampling_rates:
        Dict[str, int],
    ) -> Dict[str, np.ndarray]:

        output = {}

        modalities = sample[
            "modalities"
        ]

        for signal_name in (
            self.DEFAULT_SIGNALS
        ):

            if signal_name \
            not in modalities:

                continue

            values = (
                self.to_numpy(
                    modalities[
                        signal_name
                    ]
                )
            ).squeeze()

            output[
                signal_name
            ] = (
                self._middle_window(
                    values,
                    sampling_rates[
                        signal_name
                    ],
                )
            )

        return output

    # ======================================================
    # Plot
    # ======================================================

    def plot(
        self,
        sample: Dict[str, Any],
        sampling_rates:
        Dict[str, int],
        output_file:
        str | Path,
    ) -> None:

        emotions = (
            self.extract_labels(
                sample
            )
        )

        signals = (
            self.extract_signals(
                sample,
                sampling_rates,
            )
        )

        total_axes = (
            len(emotions)
            +
            len(signals)
        )

        fig, axes = (
            plt.subplots(
                total_axes,
                1,
                figsize=
                self.figsize,
                sharex=False,
            )
        )

        fig.suptitle(
            (
                "Multimodal Alignment\n"
                "Emotion ↔ Physiology"
            ),
            fontsize=16,
            fontweight="bold",
        )

        row = 0

        # ----------------------------------
        # Emotion Labels
        # ----------------------------------

        for label, values in (
            emotions.items()
        ):

            t = np.linspace(

                0,

                self.window_seconds,

                len(values)
            )

            axes[row].plot(
                t,
                self.normalize(
                    values
                ),
                linewidth=1.4,
            )

            axes[row].set_ylabel(
                label
            )

            axes[row].grid(
                alpha=0.3
            )

            row += 1

        # ----------------------------------
        # Signals
        # ----------------------------------

        for signal, values in (
            signals.items()
        ):

            t = np.linspace(

                0,

                self.window_seconds,

                len(values)
            )

            axes[row].plot(
                t,
                self.normalize(
                    values
                ),
                linewidth=1.0,
            )

            axes[row].set_ylabel(
                signal.upper()
            )

            axes[row].grid(
                alpha=0.3
            )

            row += 1

        axes[-1].set_xlabel(
            "Time (Seconds)"
        )

        plt.tight_layout()

        output_file = Path(
            output_file
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            output_file,
            dpi=self.dpi,
            bbox_inches="tight",
        )

        plt.close()

    # ======================================================
    # Convenience API
    # ======================================================

    def generate(
        self,
        sample: Dict[str, Any],
        sampling_rates:
        Dict[str, int],
        output_dir:
        str | Path,
    ) -> Path:

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            output_dir
            /
            "multimodal_alignment.png"
        )

        self.plot(
            sample=sample,
            sampling_rates=
            sampling_rates,
            output_file=
            output_file,
        )

        return output_file