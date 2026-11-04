"""
emotion_timeline.py
===================

Emotion Timeline Visualization

Purpose
-------
Visualize generated emotional trajectories for a
synthetic multimodal recording.

This module creates thesis-quality figures showing:

    • Valence
    • Arousal
    • Dominance
    • Stress
    • Workload
    • Fatigue
    • Engagement

across the full recording duration.

Motivation
----------
Supervisor Requirement:

    Targeted Emotional State
            vs
    Generated Emotional State

Before validating ECG, EDA, and EMG signals,
we must first inspect the generated emotional
trajectories that drive physiology and sensor
generation.

Responsibilities
----------------
✓ Create emotional timeline figure
✓ Support arbitrary continuous labels
✓ Save PNG output
✓ Publication/thesis quality figure

Does NOT
--------
✗ Generate reports
✗ Validate ECG
✗ Validate EDA
✗ Validate EMG

Example
-------
plotter = EmotionTimelinePlotter()

plotter.plot(
    sample=sample,
    output_file="emotion_timeline.png"
)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt


# ==========================================================
# Plotter
# ==========================================================

class EmotionTimelinePlotter:
    """
    Emotional trajectory visualization utility.
    """

    DEFAULT_LABELS = [
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
        figsize: tuple[int, int] = (16, 12),
        dpi: int = 300,
    ) -> None:

        self.figsize = figsize
        self.dpi = dpi

    # ------------------------------------------------------
    # Utilities
    # ------------------------------------------------------

    @staticmethod
    def _to_numpy(
        x: torch.Tensor | np.ndarray,
    ) -> np.ndarray:

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()

        return np.asarray(x)

    @staticmethod
    def _create_time_axis(
        length: int,
        duration_seconds: float,
    ) -> np.ndarray:

        return np.linspace(
            0.0,
            duration_seconds,
            length,
        )

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------

    def plot(
        self,
        sample: Dict[str, Any],
        output_file: str | Path,
    ) -> None:
        """
        Generate emotional timeline visualization.

        Parameters
        ----------
        sample:
            Dataset sample.

        output_file:
            Target PNG path.
        """

        labels = sample[
            "continuous_labels"
        ]

        metadata = sample[
            "metadata"
        ]

        duration = metadata.get(
            "duration_seconds",
            300,
        )

        active_labels = [
            label
            for label in self.DEFAULT_LABELS
            if label in labels
        ]

        n_labels = len(
            active_labels
        )

        fig, axes = plt.subplots(
            n_labels,
            1,
            figsize=self.figsize,
            sharex=True,
        )

        if n_labels == 1:
            axes = [axes]

        fig.suptitle(
            (
                f"Emotional Timeline\n"
                f"Subject "
                f"{metadata.get('subject_id', 'N/A')} | "
                f"Session "
                f"{metadata.get('session_id', 'N/A')}"
            ),
            fontsize=15,
            fontweight="bold",
        )

        # ----------------------------------------------
        # Individual trajectories
        # ----------------------------------------------

        for ax, label_name in zip(
            axes,
            active_labels,
        ):

            values = self._to_numpy(
                labels[label_name]
            )

            t = self._create_time_axis(
                len(values),
                duration,
            )

            ax.plot(
                t,
                values,
                linewidth=1.2,
            )

            ax.set_ylabel(
                label_name.capitalize()
            )

            ax.grid(
                True,
                alpha=0.3,
            )

            mean_val = float(
                np.mean(values)
            )

            ax.axhline(
                mean_val,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label=(
                    f"mean="
                    f"{mean_val:.3f}"
                ),
            )

            ax.legend(
                loc="upper right"
            )

        # ----------------------------------------------
        # TSST phase markers
        # ----------------------------------------------

        phase_lines = [
            duration * 0.20,
            duration * 0.30,
            duration * 0.55,
            duration * 0.80,
        ]

        phase_names = [
            "Preparation",
            "Speech",
            "Arithmetic",
            "Recovery",
        ]

        for ax in axes:

            for phase_x in phase_lines:

                ax.axvline(
                    phase_x,
                    color="red",
                    linestyle=":",
                    alpha=0.5,
                )

        for x, name in zip(
            phase_lines,
            phase_names,
        ):
            axes[0].text(
                x,
                axes[0].get_ylim()[1],
                name,
                fontsize=8,
                rotation=90,
                verticalalignment="top",
            )

        axes[-1].set_xlabel(
            "Time (seconds)"
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

    # ------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------

    def generate(
        self,
        sample: Dict[str, Any],
        output_directory: str | Path,
    ) -> Path:
        """
        Generate standard emotion timeline figure.

        Returns
        -------
        Path
            Saved image path.
        """

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            output_directory
            / "emotion_timeline.png"
        )

        self.plot(
            sample=sample,
            output_file=output_file,
        )

        return output_file