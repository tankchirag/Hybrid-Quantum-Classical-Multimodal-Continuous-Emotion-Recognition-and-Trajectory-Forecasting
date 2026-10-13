"""
datasets.evaluation.statistics
==============================

Dataset Statistics Module

Computes descriptive statistics and validation
metrics for the synthetic multimodal dataset.

Purpose
-------

This module serves three goals:

1. Dataset Validation

2. Dataset Understanding

3. Thesis Reporting

The generated outputs can be used directly in:

    Thesis tables

    Dataset reports

    Validation reports

    Debugging

Design Goals
------------

✓ Modality agnostic

✓ Label agnostic

✓ Dataset agnostic

✓ Extensible

✓ Research-grade reporting
"""

from __future__ import annotations

from typing import Dict
from typing import Any

import numpy as np
import torch


# ==========================================================
# Utility Functions
# ==========================================================

def tensor_statistics(
    x: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute basic statistics of a tensor.
    """

    x = x.float()

    return {

        "min":
            float(torch.min(x)),

        "max":
            float(torch.max(x)),

        "mean":
            float(torch.mean(x)),

        "std":
            float(torch.std(x)),

        "num_samples":
            int(x.numel()),
    }


# ==========================================================
# Dataset Statistics
# ==========================================================

class DatasetStatistics:
    """
    Statistics calculator for synthetic datasets.
    """

    def __init__(
        self,
        dataset,
    ) -> None:

        self.dataset = dataset

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    def dataset_summary(
        self,
    ) -> Dict[str, Any]:
        """
        General dataset properties.
        """

        return {

            "num_samples":
                len(self.dataset),

            "duration_seconds":
                self.dataset.duration_seconds,

            "num_subjects":
                self.dataset.num_subjects,

            "num_sessions":
                self.dataset.num_sessions,

            "modalities":
                list(
                    self.dataset.modalities.keys()
                ),

            "continuous_labels":
                list(
                    self.dataset.continuous_cfg.keys()
                ),

            "discrete_labels":
                list(
                    self.dataset.discrete_cfg.keys()
                ),
        }

    # ------------------------------------------------------
    # Modality Statistics
    # ------------------------------------------------------

    def modality_statistics(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Statistics for all modalities.
        """

        results = {}

        for (
            name,
            data,
        ) in sample[
            "modalities"
        ].items():

            results[name] = {

                "shape":
                    tuple(data.shape),

                **tensor_statistics(
                    data
                )
            }

        return results

    # ------------------------------------------------------
    # Continuous Labels
    # ------------------------------------------------------

    def continuous_label_statistics(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Statistics for continuous labels.
        """

        results = {}

        for (
            name,
            data,
        ) in sample[
            "continuous_labels"
        ].items():

            results[name] = {

                "shape":
                    tuple(data.shape),

                **tensor_statistics(
                    data
                )
            }

        return results

    # ------------------------------------------------------
    # Discrete Labels
    # ------------------------------------------------------

    def discrete_label_statistics(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Distribution information.
        """

        results = {}

        for (
            name,
            data,
        ) in sample[
            "discrete_labels"
        ].items():

            classes = torch.unique(
                data
            )

            results[name] = {

                "num_classes":
                    int(len(classes)),

                "classes":
                    classes.cpu()
                    .numpy()
                    .tolist(),
            }

        return results

    # ------------------------------------------------------
    # Integrity Checks
    # ------------------------------------------------------

    def validate_sample(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, bool]:
        """
        Basic dataset integrity checks.
        """

        checks = {

            "has_modalities":
                "modalities"
                in sample,

            "has_continuous_labels":
                "continuous_labels"
                in sample,

            "has_discrete_labels":
                "discrete_labels"
                in sample,

            "has_metadata":
                "metadata"
                in sample,
        }

        for (
            name,
            signal,
        ) in sample[
            "modalities"
        ].items():

            checks[
                f"{name}_finite"
            ] = bool(
                torch.isfinite(
                    signal
                ).all()
            )

        return checks

    # ------------------------------------------------------
    # Full Report
    # ------------------------------------------------------

    def generate_report(
        self,
        sample_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Generate statistics report.
        """

        sample = self.dataset[
            sample_idx
        ]

        return {

            "summary":
                self.dataset_summary(),

            "validation":
                self.validate_sample(
                    sample
                ),

            "modalities":
                self.modality_statistics(
                    sample
                ),

            "continuous_labels":
                self.continuous_label_statistics(
                    sample
                ),

            "discrete_labels":
                self.discrete_label_statistics(
                    sample
                ),
        }


# ==========================================================
# Pretty Printing
# ==========================================================

def print_statistics_report(
    report: Dict[str, Any],
) -> None:
    """
    Console report.
    """

    print(
        "\n"
        + "=" * 60
    )

    print(
        "ULM-TSST DATASET STATISTICS"
    )

    print(
        "=" * 60
    )

    print("\nSUMMARY")

    for k, v in report[
        "summary"
    ].items():

        print(
            f"{k}: {v}"
        )

    print(
        "\nVALIDATION"
    )

    for k, v in report[
        "validation"
    ].items():

        status = (
            "PASS"
            if v
            else "FAIL"
        )

        print(
            f"{k}: {status}"
        )

    print(
        "\nMODALITIES"
    )

    for (
        name,
        stats,
    ) in report[
        "modalities"
    ].items():

        print(
            f"{name}: "
            f"shape={stats['shape']} "
            f"mean={stats['mean']:.3f}"
        )

    print(
        "\nCONTINUOUS LABELS"
    )

    for (
        name,
        stats,
    ) in report[
        "continuous_labels"
    ].items():

        print(
            f"{name}: "
            f"mean={stats['mean']:.3f}"
        )

    print(
        "\nDISCRETE LABELS"
    )

    for (
        name,
        stats,
    ) in report[
        "discrete_labels"
    ].items():

        print(
            f"{name}: "
            f"{stats['num_classes']} classes"
        )

    print(
        "\n"
        + "=" * 60
    )