"""
datasets.evaluation.correlations
================================

Scientific Correlation Validation Module

This module evaluates whether the synthetic
dataset exhibits realistic relationships between:

    • latent labels
    • physiology
    • sensor signals

The goal is not only verifying that tensors exist,
but also validating that generated trajectories
behave similarly to expected human behavior.

Examples

    Stress ↑
        →
    Heart Rate ↑

    Stress ↑
        →
    EDA ↑

    Stress ↑
        →
    Valence ↓

    Workload ↑
        →
    EEG Activation ↑

    Fatigue ↑
        →
    KSS ↑

Design Goals
------------

✓ Dataset agnostic

✓ Modality agnostic

✓ Label agnostic

✓ Scientific validation

✓ Thesis-ready metrics
"""

from __future__ import annotations

from typing import Any
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


# ==========================================================
# Utilities
# ==========================================================

def _flatten_tensor(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Convert tensor to 1D float tensor.
    """

    return (
        x.float()
         .reshape(-1)
    )


# ==========================================================
# Resampling
# ==========================================================

def _resample_to_common_length(
    x: torch.Tensor,
    y: torch.Tensor,
    max_length: int = 5000,
):
    """
    Resample two signals onto a common grid.

    This is critical because modalities operate
    at different frequencies.

    Example

        Stress : 15000

        BPM    : 2400

    Both are resampled before correlation.
    """

    x = _flatten_tensor(x)
    y = _flatten_tensor(y)

    target_length = min(
        max(
            len(x),
            len(y),
        ),
        max_length,
    )

    x = F.interpolate(
        x.view(1, 1, -1),
        size=target_length,
        mode="linear",
        align_corners=True,
    ).view(-1)

    y = F.interpolate(
        y.view(1, 1, -1),
        size=target_length,
        mode="linear",
        align_corners=True,
    ).view(-1)

    return x, y


# ==========================================================
# Correlation
# ==========================================================

def pearson_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """
    Robust Pearson correlation.

    Signals are first temporally aligned.
    """

    x, y = _resample_to_common_length(
        x,
        y,
    )

    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    if np.std(x_np) < 1e-8:
        return 0.0

    if np.std(y_np) < 1e-8:
        return 0.0

    return float(
        np.corrcoef(
            x_np,
            y_np,
        )[0, 1]
    )


# ==========================================================
# Correlation Analyzer
# ==========================================================

class CorrelationAnalyzer:
    """
    Scientific validation framework.
    """

    def __init__(
        self,
        sample: Dict[str, Any],
    ):

        self.sample = sample

    # --------------------------------------------------
    # Label Correlations
    # --------------------------------------------------

    def label_correlations(
        self,
    ) -> Dict[str, float]:

        labels = self.sample[
            "continuous_labels"
        ]

        results = {}

        names = list(
            labels.keys()
        )

        for i in range(len(names)):

            for j in range(
                i + 1,
                len(names),
            ):

                a = names[i]
                b = names[j]

                results[
                    f"{a}<->{b}"
                ] = pearson_correlation(
                    labels[a],
                    labels[b],
                )

        return results

    # --------------------------------------------------
    # Expected Scientific Relationships
    # --------------------------------------------------

    def expected_relationships(
        self,
    ) -> Dict[str, float]:

        results = {}

        labels = self.sample[
            "continuous_labels"
        ]

        modalities = self.sample[
            "modalities"
        ]

        # ----------------------------------------------
        # Stress ↔ BPM
        # ----------------------------------------------

        if (
            "stress" in labels
            and
            "bpm" in modalities
        ):

            results[
                "stress_vs_bpm"
            ] = pearson_correlation(
                labels["stress"],
                modalities["bpm"],
            )

        # ----------------------------------------------
        # Stress ↔ EDA
        # ----------------------------------------------

        if (
            "stress" in labels
            and
            "eda" in modalities
        ):

            results[
                "stress_vs_eda"
            ] = pearson_correlation(
                labels["stress"],
                modalities["eda"],
            )

        # ----------------------------------------------
        # Stress ↔ ECG
        # ----------------------------------------------

        if (
            "stress" in labels
            and
            "ecg" in modalities
        ):

            results[
                "stress_vs_ecg"
            ] = pearson_correlation(
                labels["stress"],
                modalities["ecg"],
            )

        # ----------------------------------------------
        # Valence ↔ Stress
        # ----------------------------------------------

        if (
            "valence" in labels
            and
            "stress" in labels
        ):

            results[
                "valence_vs_stress"
            ] = pearson_correlation(
                labels["valence"],
                labels["stress"],
            )

        # ----------------------------------------------
        # Workload ↔ EEG
        # ----------------------------------------------

        if (
            "workload" in labels
            and
            "eeg" in modalities
        ):

            results[
                "workload_vs_eeg"
            ] = pearson_correlation(
                labels["workload"],
                modalities["eeg"],
            )

        # ----------------------------------------------
        # Fatigue ↔ Text Stress Channel
        # ----------------------------------------------

        if (
            "fatigue" in labels
            and
            "text" in modalities
        ):

            results[
                "fatigue_vs_text"
            ] = pearson_correlation(
                labels["fatigue"],
                modalities["text"][:, 1],
            )

        return results

    # --------------------------------------------------
    # Matrix
    # --------------------------------------------------

    def correlation_matrix(
        self,
    ):

        labels = self.sample[
            "continuous_labels"
        ]

        names = list(
            labels.keys()
        )

        N = len(names)

        matrix = np.zeros(
            (N, N)
        )

        for i in range(N):

            for j in range(N):

                matrix[i, j] = (
                    pearson_correlation(
                        labels[names[i]],
                        labels[names[j]],
                    )
                )

        return names, matrix

    # --------------------------------------------------
    # Validation Checks
    # --------------------------------------------------

    def validation_checks(
        self,
    ) -> Dict[str, bool]:

        rel = self.expected_relationships()

        checks = {}

        if "stress_vs_bpm" in rel:
            checks[
                "stress_bpm_positive"
            ] = rel[
                "stress_vs_bpm"
            ] > 0.30

        if "stress_vs_eda" in rel:
            checks[
                "stress_eda_positive"
            ] = rel[
                "stress_vs_eda"
            ] > 0.20

        if "valence_vs_stress" in rel:
            checks[
                "valence_stress_negative"
            ] = rel[
                "valence_vs_stress"
            ] < -0.30

        if "workload_vs_eeg" in rel:
            checks[
                "workload_eeg_positive"
            ] = rel[
                "workload_vs_eeg"
            ] > 0.15

        return checks


# ==========================================================
# Reporting
# ==========================================================

def print_correlation_report(
    analyzer: CorrelationAnalyzer,
) -> None:

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CORRELATION VALIDATION"
    )

    print(
        "=" * 60
    )

    print(
        "\nEXPECTED RELATIONSHIPS\n"
    )

    expected = (
        analyzer.expected_relationships()
    )

    for name, value in expected.items():

        print(
            f"{name:35s}"
            f"{value:+.3f}"
        )

    print(
        "\nVALIDATION CHECKS\n"
    )

    checks = (
        analyzer.validation_checks()
    )

    for k, v in checks.items():

        status = (
            "PASS"
            if v
            else "FAIL"
        )

        print(
            f"{k:35s}"
            f"{status}"
        )

    print(
        "\n"
        + "=" * 60
    )