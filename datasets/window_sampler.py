"""
window_sampler.py
=================

Temporal Window Sampler

Creates training windows from long-duration
multimodal recordings.

Design Goals
------------

✓ Recognition Support

✓ Forecasting Support

✓ Continuous Labels

✓ Discrete Labels

✓ Modality Agnostic

✓ Label Agnostic

✓ Dataset Agnostic

✓ Flexible Horizons

✓ History Labels

✓ Future Proof
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Dict

import torch


# ==========================================================
# Output Container
# ==========================================================

@dataclass
class WindowSample:
    """
    Sample returned by sampler.
    """

    modalities: Dict[str, torch.Tensor]

    continuous_labels: Dict[str, torch.Tensor]

    discrete_labels: Dict[str, torch.Tensor]

    recognition_targets: Dict[
        str,
        Dict[str, torch.Tensor]
    ]

    forecast_targets: Dict[
        str,
        Dict[str, torch.Tensor]
    ]

    metadata: Dict[str, Any]


# ==========================================================
# Sampler
# ==========================================================

class WindowSampler:
    """
    Temporal window extraction.

    Produces

        history windows

        recognition targets

        forecasting targets
    """

    def __init__(
        self,
        history_length: int,
        forecast_horizon: int,
        forecast_labels=None,
    ) -> None:

        self.history_length = int(
            history_length
        )

        self.forecast_horizon = int(
            forecast_horizon
        )

        self.forecast_labels = (
            forecast_labels
        )

    # ------------------------------------------------------
    # Modality Window
    # ------------------------------------------------------

    def _extract_modality_window(
        self,
        signal: torch.Tensor,
        center_idx: int,
    ) -> torch.Tensor:

        start = (
            center_idx
            - self.history_length
        )

        end = center_idx

        return signal[start:end]

    # ------------------------------------------------------
    # Main
    # ------------------------------------------------------

    def sample(
        self,
        clip: Dict[str, Any],
        center_idx: int,
    ) -> WindowSample:
        """
        Extract one training sample.
        """

        history_modalities = {}

        # ==================================================
        # Modalities
        # ==================================================

        for (
            name,
            signal,
        ) in clip[
            "modalities"
        ].items():

            mapped_idx = min(
                center_idx,
                signal.shape[0] - 1,
            )

            history_modalities[
                name
            ] = self._extract_modality_window(
                signal,
                mapped_idx,
            )

        # ==================================================
        # Label Histories
        # ==================================================

        continuous_windows = {}

        discrete_windows = {}

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
            name,
            signal,
        ) in clip[
            "continuous_labels"
        ].items():

            start = (
                center_idx
                - self.history_length
            )

            end = center_idx

            future_end = min(

                center_idx
                +
                self.forecast_horizon,

                signal.shape[0],
            )

            continuous_windows[
                name
            ] = signal[
                start:end
            ]

            # Recognition

            recognition_targets[
                "continuous"
            ][name] = signal[
                center_idx
            ]

            # Forecast

            if (
                self.forecast_labels
                is None
                or
                name
                in self.forecast_labels
            ):

                forecast_targets[
                    "continuous"
                ][name] = signal[
                    center_idx:future_end
                ]

        # ==================================================
        # Discrete Labels
        # ==================================================

        for (
            name,
            signal,
        ) in clip[
            "discrete_labels"
        ].items():

            start = (
                center_idx
                - self.history_length
            )

            end = center_idx

            future_end = min(

                center_idx
                +
                self.forecast_horizon,

                signal.shape[0],
            )

            discrete_windows[
                name
            ] = signal[
                start:end
            ]

            # Recognition

            recognition_targets[
                "discrete"
            ][name] = signal[
                center_idx
            ]

            # Forecast

            forecast_targets[
                "discrete"
            ][name] = signal[
                center_idx:future_end
            ]

        # ==================================================
        # Return
        # ==================================================

        return WindowSample(

            modalities=
            history_modalities,

            continuous_labels=
            continuous_windows,

            discrete_labels=
            discrete_windows,

            recognition_targets=
            recognition_targets,

            forecast_targets=
            forecast_targets,

            metadata=
            clip["metadata"],
        )

    # ------------------------------------------------------
    # Valid Range
    # ------------------------------------------------------

    def valid_range(
        self,
        clip_length: int,
    ):

        start = self.history_length

        end = (
            clip_length
            - self.forecast_horizon
        )

        return range(
            start,
            end,
        )