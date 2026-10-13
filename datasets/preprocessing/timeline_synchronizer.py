"""
datasets.preprocessing.timeline_synchronizer
============================================

Timeline Synchronizer

Purpose
-------
Align multimodal signals with differing native
sampling rates onto a shared timeline.

Examples
--------

ECG     300000
EDA      38400
EEG      76800
VIDEO     7500
TEXT      1500

↓

Synchronizer

↓

All Modalities

7500


Design Goals
------------

✓ Data Agnostic

✓ Dataset Agnostic

✓ Modality Agnostic

✓ Continuous Label Agnostic

✓ Discrete Label Agnostic

✓ Recognition Compatible

✓ Forecasting Compatible

✓ Configurable

✓ Future Proof

Important
---------

This module does NOT know:

    valence

    arousal

    stress

    workload

or any dataset-specific label.

It only synchronizes timelines.
"""

from __future__ import annotations

from typing import Any
from typing import Dict

import torch
import torch.nn.functional as F


class TimelineSynchronizer:
    """
    Universal timeline synchronizer.
    """

    def __init__(
        self,
        target_length_source: str = "continuous_labels",
        reference_label: str | None = None,
        interpolation_modes: Dict[str, str] | None = None,
    ):
        """
        Parameters
        ----------
        target_length_source

            continuous_labels
            discrete_labels

        reference_label

            Optional explicit reference label.

            Example:

            valence

            arousal

            stress

            etc.

            None = automatically use first
            available label.

        interpolation_modes

            Optional modality-specific
            interpolation modes.
        """

        self.target_length_source = (
            target_length_source
        )

        self.reference_label = (
            reference_label
        )

        self.interpolation_modes = (
            interpolation_modes
            or {}
        )

    # ======================================================
    # Utilities
    # ======================================================

    def _get_target_length(
        self,
        sample: Dict[str, Any],
    ) -> int:

        if (
            self.target_length_source
            == "continuous_labels"
        ):

            labels = sample[
                "continuous_labels"
            ]

        elif (
            self.target_length_source
            == "discrete_labels"
        ):

            labels = sample[
                "discrete_labels"
            ]

        else:

            raise ValueError(
                "Unsupported "
                "target_length_source"
            )

        if len(labels) == 0:

            raise RuntimeError(
                "No labels found."
            )

        if (
            self.reference_label
            is not None
        ):

            if (
                self.reference_label
                not in labels
            ):

                raise KeyError(
                    f"Reference label "
                    f"'{self.reference_label}' "
                    f"not found."
                )

            return len(
                labels[
                    self.reference_label
                ]
            )

        first_label = next(
            iter(
                labels.values()
            )
        )

        return len(
            first_label
        )

    # ======================================================

    def _get_mode(
        self,
        modality: str,
    ) -> str:

        return (
            self.interpolation_modes.get(
                modality,
                "linear",
            )
        )

    # ======================================================

    def _resample(
        self,
        signal: torch.Tensor,
        target_length: int,
        modality: str,
    ) -> torch.Tensor:

        mode = self._get_mode(
            modality
        )

        # ----------------------------------------------
        # (L,)
        # ----------------------------------------------

        if signal.ndim == 1:

            x = signal.view(
                1,
                1,
                -1,
            )

            y = F.interpolate(
                x,
                size=target_length,
                mode=mode,
                align_corners=(
                    False
                    if mode != "nearest"
                    else None
                ),
            )

            return y.view(
                target_length,
            )

        # ----------------------------------------------
        # (L,D)
        # ----------------------------------------------

        if signal.ndim == 2:

            x = (
                signal.T
                .unsqueeze(0)
            )

            y = F.interpolate(
                x,
                size=target_length,
                mode=mode,
                align_corners=(
                    False
                    if mode != "nearest"
                    else None
                ),
            )

            return (
                y.squeeze(0)
                .T
            )

        raise ValueError(
            f"Unsupported shape: "
            f"{signal.shape}"
        )

    # ======================================================
    # Main
    # ======================================================

    def synchronize(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:

        target_length = (
            self._get_target_length(
                sample
            )
        )

        synchronized = dict(
            sample
        )

        synchronized_modalities = {}

        for (
            modality,
            signal,
        ) in sample[
            "modalities"
        ].items():

            synchronized_modalities[
                modality
            ] = self._resample(

                signal=
                signal,

                target_length=
                target_length,

                modality=
                modality,
            )

        synchronized[
            "modalities"
        ] = (
            synchronized_modalities
        )

        synchronized[
            "metadata"
        ] = dict(
            synchronized.get(
                "metadata",
                {},
            )
        )

        synchronized[
            "metadata"
        ][
            "synchronized"
        ] = True

        synchronized[
            "metadata"
        ][
            "target_length"
        ] = target_length

        return synchronized

    # ======================================================

    def __call__(
        self,
        sample,
    ):

        return self.synchronize(
            sample
        )