"""
physiological_validation.py
===========================

Physiological Validation Framework

Purpose
-------
Provides quantitative validation utilities for the
Synthetic ULM-TSST Dataset.

This module serves as the foundation of the
physiological validation pipeline.

Responsibilities
----------------
1. Sample validation
2. Metadata extraction
3. Continuous-label analysis
4. Discrete-label analysis
5. Modality analysis
6. Emotional trajectory analysis
7. JSON export

This module intentionally DOES NOT perform:

    - plotting
    - visualization
    - report generation

Those responsibilities belong to:

    emotion_timeline.py
    ecg_validation.py
    eda_validation.py
    emg_validation.py
    signal_report.py

Design Goals
------------
✓ Dataset agnostic
✓ Modality agnostic
✓ Label agnostic
✓ Forecasting compatible
✓ Reusable
✓ Type-safe
✓ JSON exportable
✓ Auto-documented

Example
-------
validator = PhysiologicalValidator()

summary = validator.build_summary(sample)

validator.save_summary(
    summary,
    "emotion_summary.json"
)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import torch


# ==========================================================
# Validator
# ==========================================================


class PhysiologicalValidator:
    """
    Synthetic Dataset Validation Helper.

    Expected Sample Format
    ----------------------

    sample = {

        "modalities": {...},

        "continuous_labels": {...},

        "discrete_labels": {...},

        "masks": {...},

        "metadata": {...}
    }
    """

    # ======================================================
    # Constructor
    # ======================================================

    def __init__(self) -> None:
        pass

    # ======================================================
    # Utility Functions
    # ======================================================

    @staticmethod
    def _to_numpy(
        value: torch.Tensor | np.ndarray | list,
    ) -> np.ndarray:
        """
        Convert arbitrary tensor-like object
        into NumPy array.
        """

        if isinstance(value, np.ndarray):
            return value

        if isinstance(value, torch.Tensor):
            return (
                value
                .detach()
                .cpu()
                .numpy()
            )

        return np.asarray(value)

    @staticmethod
    def _basic_stats(
        values: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute standard statistics.
        """

        values = np.asarray(values)

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }

    @staticmethod
    def _range(
        values: np.ndarray,
    ) -> float:
        """
        Compute signal range.
        """

        return float(
            np.max(values)
            - np.min(values)
        )

    # ======================================================
    # Validation
    # ======================================================

    @staticmethod
    def validate_sample(
        sample: Dict[str, Any]
    ) -> None:
        """
        Validate required sample fields.
        """

        required_keys = [
            "modalities",
            "continuous_labels",
            "discrete_labels",
            "metadata",
        ]

        for key in required_keys:

            if key not in sample:

                raise KeyError(
                    f"Missing sample key: {key}"
                )

    # ======================================================
    # Metadata
    # ======================================================

    def metadata_summary(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:

        metadata = sample["metadata"]

        return {
            "clip_id":
                metadata.get(
                    "clip_id",
                    None,
                ),

            "subject_id":
                metadata.get(
                    "subject_id",
                    None,
                ),

            "session_id":
                metadata.get(
                    "session_id",
                    None,
                ),

            "duration_seconds":
                metadata.get(
                    "duration_seconds",
                    None,
                ),
        }

    # ======================================================
    # Continuous Labels
    # ======================================================

    def continuous_summary(
        self,
        labels: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Analyze continuous trajectories.
        """

        summary = {}

        for name, values in labels.items():

            arr = self._to_numpy(
                values
            )

            summary[name] = {

                **self._basic_stats(
                    arr
                ),

                "range":
                    self._range(arr),

                "trajectory_length":
                    int(len(arr)),
            }

        return summary

    # ======================================================
    # Emotional Dynamics
    # ======================================================

    def emotional_dynamics(
        self,
        labels: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        High-level emotional trajectory summary.
        """

        output = {}

        tracked = [
            "valence",
            "arousal",
            "dominance",
            "stress",
            "workload",
            "fatigue",
            "engagement",
        ]

        for name in tracked:

            if name not in labels:
                continue

            values = self._to_numpy(
                labels[name]
            )

            output[name] = {

                "mean":
                    float(
                        np.mean(values)
                    ),

                "std":
                    float(
                        np.std(values)
                    ),

                "minimum":
                    float(
                        np.min(values)
                    ),

                "maximum":
                    float(
                        np.max(values)
                    ),

                "range":
                    self._range(values),
            }

        return output

    # ======================================================
    # Discrete Labels
    # ======================================================

    def discrete_summary(
        self,
        labels: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Generic discrete label distribution.
        """

        output = {}

        for name, values in labels.items():

            arr = self._to_numpy(
                values
            ).astype(int)

            unique, counts = np.unique(
                arr,
                return_counts=True,
            )

            output[name] = {

                "num_classes":
                    int(len(unique)),

                "distribution": {

                    str(int(label)):
                    int(freq)

                    for label, freq
                    in zip(
                        unique,
                        counts,
                    )
                }
            }

        return output

    # ======================================================
    # Modalities
    # ======================================================

    def modality_summary(
        self,
        modalities: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:

        summary = {}

        for modality, values in (
            modalities.items()
        ):

            arr = self._to_numpy(
                values
            )

            summary[modality] = {

                "shape":
                    list(arr.shape),

                **self._basic_stats(
                    arr
                ),
            }

        return summary

    # ======================================================
    # Build Summary
    # ======================================================

    def build_summary(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build complete validation summary.
        """

        self.validate_sample(
            sample
        )

        continuous = sample[
            "continuous_labels"
        ]

        summary = {

            "metadata":
                self.metadata_summary(
                    sample
                ),

            "emotional_dynamics":
                self.emotional_dynamics(
                    continuous
                ),

            "continuous_labels":
                self.continuous_summary(
                    continuous
                ),

            "discrete_labels":
                self.discrete_summary(
                    sample[
                        "discrete_labels"
                    ]
                ),

            "modalities":
                self.modality_summary(
                    sample[
                        "modalities"
                    ]
                ),
        }

        return summary

    # ======================================================
    # Export
    # ======================================================

    @staticmethod
    def save_summary(
        summary: Dict[str, Any],
        output_file: str | Path,
    ) -> None:
        """
        Save validation summary as JSON.
        """

        output_file = Path(
            output_file
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as fp:

            json.dump(
                summary,
                fp,
                indent=4,
                ensure_ascii=False,
            )

    # ======================================================
    # Convenience API
    # ======================================================

    def validate(
        self,
        sample: Dict[str, Any],
        output_json: str | Path,
    ) -> Dict[str, Any]:
        """
        Full validation pipeline.

        Parameters
        ----------
        sample:
            Dataset sample.

        output_json:
            Output summary path.

        Returns
        -------
        dict
            Validation summary.
        """

        summary = self.build_summary(
            sample
        )

        self.save_summary(
            summary,
            output_json,
        )

        return summary
    # ======================================