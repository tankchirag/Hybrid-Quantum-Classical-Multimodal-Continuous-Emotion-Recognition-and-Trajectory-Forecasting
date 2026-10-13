"""
datasets.synchronizer
=====================

Universal Multimodal Synchronizer

This module provides temporal alignment for arbitrary multimodal
datasets used in:

    • RECOLA
    • SEWA
    • ULM-TSST Synthetic
    • Future custom datasets

The synchronizer is intentionally:

    ✓ modality agnostic
    ✓ label agnostic
    ✓ dataset agnostic
    ✓ forecasting compatible
    ✓ configurable
    ✓ future proof

Motivation
----------
Different modalities often have different sequence lengths,
sampling frequencies or temporal resolutions:

    ECG     : 1000 Hz
    EDA     : 128 Hz
    RESP    : 256 Hz
    EEG     : 512 Hz
    Eye     : 120 Hz
    Audio   : 16000 Hz
    Video   : 25 fps
    Text    : token sequence

The Synchronizer maps all enabled temporal modalities onto a
common temporal grid of length T.

The output becomes suitable for:

    raw signal encoders
    multimodal tokenization
    fusion modules
    transformer backbones
    forecasting heads

without requiring modality-specific logic downstream.
"""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

import torch
import torch.nn.functional as F


class Synchronizer:
    """
    Universal temporal synchronizer.

    Parameters
    ----------
    target_seq_len : int

        Target synchronized sequence length.

    temporal_modalities : sequence[str], optional

        Modalities that should be synchronized.

    interpolation_modes : dict, optional

        Modality specific interpolation modes.

    device : str

        Processing device.

    Examples
    --------

    >>> sync = Synchronizer(
    ...     target_seq_len=50,
    ...     temporal_modalities=[
    ...         "audio",
    ...         "video",
    ...         "ecg",
    ...         "eda",
    ...         "eeg",
    ...     ]
    ... )

    >>> sample = sync(sample)
    """

    def __init__(
        self,
        target_seq_len: int,
        temporal_modalities: Optional[Sequence[str]] = None,
        interpolation_modes: Optional[Dict[str, str]] = None,
        device: str = "cpu",
    ) -> None:

        self.T = int(target_seq_len)

        self.device = torch.device(device)

        self.temporal_modalities = set(
            temporal_modalities or []
        )

        self.interpolation_modes = (
            interpolation_modes or {}
        )

    # ==================================================
    # Utility functions
    # ==================================================

    def _get_mode(self, modality: str) -> str:
        """
        Return interpolation mode.

        Defaults to linear.
        """

        return self.interpolation_modes.get(
            modality,
            "linear",
        )

    # ==================================================
    # Sequence resampling
    # ==================================================

    def _resample_scalar(
        self,
        signal: torch.Tensor,
        mode: str = "linear",
    ) -> torch.Tensor:
        """
        Resample scalar sequence.

        Input
        -----

            (L,)
            (L,1)

        Output
        ------

            (T,1)
        """

        signal = signal.float()

        if signal.ndim == 2:
            signal = signal.squeeze(-1)

        signal = signal.view(
            1,
            1,
            -1,
        )

        signal = F.interpolate(
            signal,
            size=self.T,
            mode=mode,
            align_corners=True
            if mode != "nearest"
            else None,
        )

        signal = signal.squeeze(0).transpose(
            0,
            1,
        )

        return signal

    # ==================================================

    def _resample_features(
        self,
        signal: torch.Tensor,
        mode: str = "linear",
    ) -> torch.Tensor:
        """
        Resample feature sequence.

        Input
        -----

            (L,D)

        Output
        ------

            (T,D)
        """

        signal = signal.float()

        signal = (
            signal.transpose(0, 1)
            .unsqueeze(0)
        )

        signal = F.interpolate(
            signal,
            size=self.T,
            mode=mode,
            align_corners=True
            if mode != "nearest"
            else None,
        )

        signal = (
            signal.squeeze(0)
            .transpose(0, 1)
        )

        return signal

    # ==================================================

    def _resample_video(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        """
        Temporal video resampling.

        Input
        -----

            (F,C,H,W)

        Output
        ------

            (T,C,H,W)
        """

        F_frames = video.shape[0]

        indices = torch.linspace(
            0,
            F_frames - 1,
            self.T,
        ).long()

        return video[indices]

    # ==================================================

    def _is_temporal_tensor(
        self,
        value: Any,
    ) -> bool:
        """
        Detect temporal tensor.
        """

        if not isinstance(
            value,
            torch.Tensor,
        ):
            return False

        return value.ndim in (
            1,
            2,
            4,
        )

    # ==================================================

    def _synchronize_tensor(
        self,
        modality: str,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Synchronize a modality tensor.
        """

        mode = self._get_mode(modality)

        if tensor.ndim == 1:
            return self._resample_scalar(
                tensor,
                mode,
            )

        if tensor.ndim == 2:

            if tensor.shape[-1] == 1:
                return self._resample_scalar(
                    tensor,
                    mode,
                )

            return self._resample_features(
                tensor,
                mode,
            )

        if tensor.ndim == 4:
            return self._resample_video(
                tensor
            )

        return tensor

    # ==================================================
    # Public API
    # ==================================================

    def synchronize(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Synchronize multimodal sample.

        Parameters
        ----------
        sample : dict

            Arbitrary multimodal sample.

        Returns
        -------
        dict

            Temporally aligned sample.
        """

        synchronized = {}

        for key, value in sample.items():

            if value is None:
                synchronized[key] = None
                continue

            if (
                key in self.temporal_modalities
                and self._is_temporal_tensor(
                    value
                )
            ):
                synchronized[key] = (
                    self._synchronize_tensor(
                        key,
                        value,
                    )
                )

            else:
                synchronized[key] = value

        return synchronized

    # ==================================================

    def __call__(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Callable interface.
        """

        return self.synchronize(sample)