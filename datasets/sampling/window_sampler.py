"""
datasets.sampling.window_sampler
================================

Universal Temporal Window Sampler

Purpose
-------
Extract supervised learning windows from
synchronized multimodal recordings.

Supports
--------

✓ Continuous Recognition

✓ Continuous Forecasting

✓ Discrete Classification

✓ Discrete Forecasting

✓ Multi-Task Learning

✓ Dataset Agnostic

✓ Modality Agnostic

✓ Label Agnostic

✓ Future Proof

Pipeline
--------

Disk Dataset
      ↓

TimelineSynchronizer
      ↓

WindowSampler
      ↓

Multimodal Collate
      ↓

Training

Important
---------

The sampler contains NO hardcoded:

    modality names

    label names

    sequence lengths

    datasets

Everything is driven by the incoming sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Iterable

import torch


# ==========================================================
# Output Container
# ==========================================================

@dataclass
class WindowSample:
    """
    Training-ready sample.
    """

    modalities: Dict[str, torch.Tensor]

    continuous_labels: Dict[str, torch.Tensor]

    discrete_labels: Dict[str, torch.Tensor]

    recognition_targets: Dict[
        str,
        Dict[str, torch.Tensor],
    ]

    forecast_targets: Dict[
        str,
        Dict[str, torch.Tensor],
    ]

    metadata: Dict[str, Any]


# ==========================================================
# Window Sampler
# ==========================================================

class WindowSampler:
    """
    Universal temporal sampler.
    """

    def __init__(
        self,
        history_length: int,
        forecast_horizon: int,
        forecast_continuous_labels: Iterable[str] | None = None,
        forecast_discrete_labels: Iterable[str] | None = None,
    ):

        self.history_length = int(
            history_length
        )

        self.forecast_horizon = int(
            forecast_horizon
        )

        self.forecast_continuous_labels = (

            set(
                forecast_continuous_labels
            )

            if forecast_continuous_labels
            is not None

            else None
        )

        self.forecast_discrete_labels = (

            set(
                forecast_discrete_labels
            )

            if forecast_discrete_labels
            is not None

            else None
        )

    # ======================================================
    # Utilities
    # ======================================================

    def _extract_history(
        self,
        signal: torch.Tensor,
        center_idx: int,
    ) -> torch.Tensor:

        start = (
            center_idx
            - self.history_length
        )

        return signal[
            start:center_idx
        ]

    # ======================================================

    def _extract_future(
        self,
        signal: torch.Tensor,
        center_idx: int,
    ) -> torch.Tensor:

        return signal[
            center_idx:
            center_idx
            +
            self.forecast_horizon
        ]

    # ======================================================

    def _validate_center(
        self,
        clip_length: int,
        center_idx: int,
    ):

        min_idx = (
            self.history_length
        )

        max_idx = (
            clip_length
            -
            self.forecast_horizon
        )

        if center_idx < min_idx:

            raise ValueError(
                f"center_idx={center_idx} "
                f"is smaller than "
                f"history_length="
                f"{self.history_length}"
            )

        if center_idx >= max_idx:

            raise ValueError(
                f"center_idx={center_idx} "
                f"is too close to clip end."
            )

    # ======================================================
    # Main
    # ======================================================

    def sample(
        self,
        synchronized_clip:
        Dict[str, Any],

        center_idx: int,
    ) -> WindowSample:
        """
        Extract supervised training sample.

        Assumptions
        -----------

        Synchronization already happened.

        All modalities and labels share
        the same timeline length.
        """

        # --------------------------------------------------
        # Determine Shared Length
        # --------------------------------------------------

        first_label = next(
            iter(
                synchronized_clip[
                    "continuous_labels"
                ].values()
            )
        )

        clip_length = len(
            first_label
        )

        self._validate_center(

            clip_length=
            clip_length,

            center_idx=
            center_idx,
        )

        # ==================================================
        # Modalities
        # ==================================================

        modality_histories = {}

        for (
            modality,
            signal,
        ) in synchronized_clip[
            "modalities"
        ].items():

            modality_histories[
                modality
            ] = self._extract_history(

                signal=
                signal,

                center_idx=
                center_idx,
            )

        # ==================================================
        # Continuous Labels
        # ==================================================

        continuous_histories = {}

        # ==================================================
        # Discrete Labels
        # ==================================================

        discrete_histories = {}

        # ==================================================
        # Targets
        # ==================================================

        recognition_targets = {

            "continuous": {},

            "discrete": {},
        }

        forecast_targets = {

            "continuous": {},

            "discrete": {},
        }

        # ==================================================
        # Continuous Labels
        # ==================================================

        for (
            label_name,
            signal,
        ) in synchronized_clip[
            "continuous_labels"
        ].items():

            continuous_histories[
                label_name
            ] = self._extract_history(

                signal,

                center_idx,
            )

            # ------------------------------------
            # Recognition
            # ------------------------------------

            recognition_targets[
                "continuous"
            ][label_name] = signal[
                center_idx
            ]

            # ------------------------------------
            # Forecasting
            # ------------------------------------

            if (

                self
                .forecast_continuous_labels
                is None

                or

                label_name
                in
                self
                .forecast_continuous_labels

            ):

                forecast_targets[
                    "continuous"
                ][label_name] = (
                    self._extract_future(
                        signal,
                        center_idx,
                    )
                )

        # ==================================================
        # Discrete Labels
        # ==================================================

        for (
            label_name,
            signal,
        ) in synchronized_clip[
            "discrete_labels"
        ].items():

            discrete_histories[
                label_name
            ] = self._extract_history(

                signal,

                center_idx,
            )

            # ------------------------------------
            # Recognition
            # ------------------------------------

            recognition_targets[
                "discrete"
            ][label_name] = signal[
                center_idx
            ]

            # ------------------------------------
            # Forecasting
            # ------------------------------------

            if (

                self
                .forecast_discrete_labels
                is None

                or

                label_name
                in
                self
                .forecast_discrete_labels

            ):

                forecast_targets[
                    "discrete"
                ][label_name] = (
                    self._extract_future(
                        signal,
                        center_idx,
                    )
                )

        # ==================================================
        # Metadata
        # ==================================================

        metadata = dict(

            synchronized_clip.get(
                "metadata",
                {},
            )
        )

        metadata[
            "center_idx"
        ] = center_idx

        metadata[
            "history_length"
        ] = (
            self.history_length
        )

        metadata[
            "forecast_horizon"
        ] = (
            self.forecast_horizon
        )

        # ==================================================
        # Output
        # ==================================================

        return WindowSample(

            modalities=
            modality_histories,

            continuous_labels=
            continuous_histories,

            discrete_labels=
            discrete_histories,

            recognition_targets=
            recognition_targets,

            forecast_targets=
            forecast_targets,

            metadata=
            metadata,
        )

    # ======================================================

    def valid_range(
        self,
        clip_length: int,
    ):

        start = (
            self.history_length
        )

        end = (
            clip_length
            -
            self.forecast_horizon
        )

        return range(
            start,
            end,
        )