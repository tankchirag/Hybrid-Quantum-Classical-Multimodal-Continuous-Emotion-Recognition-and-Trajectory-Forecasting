"""
datasets.synchronizer
=====================

Universal Temporal Synchronizer

Resamples multimodal recordings with different
native frequencies onto a common temporal axis.

Example
-------

Native Signals

    ECG   : 1000 Hz

    EDA   : 128 Hz

    EEG   : 256 Hz

    Video : 25 Hz

    Audio : 100 Hz


↓

Synchronizer


↓

Shared Timeline

    T = 3000

↓

Transformer

Design Goals
------------

✓ modality agnostic

✓ dataset agnostic

✓ label agnostic

✓ forecasting compatible

✓ configurable

✓ future-proof
"""

from __future__ import annotations

from typing import Dict
from typing import Any

import torch
import torch.nn.functional as F


class Synchronizer:
    """
    Universal multimodal synchronizer.
    """

    def __init__(
        self,
        target_seq_len: int,
        interpolation_modes: Dict[str, str] | None = None,
        device: str = "cpu",
    ):

        self.target_seq_len = int(
            target_seq_len
        )

        self.device = torch.device(
            device
        )

        self.interpolation_modes = (
            interpolation_modes
            or {}
        )

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def _mode(
        self,
        modality: str,
    ):

        return self.interpolation_modes.get(
            modality,
            "linear",
        )

    # --------------------------------------------------

    def _resample(
        self,
        x: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        """
        Resample time-series.

        Supported

            (L,)

            (L,1)

            (L,D)
        """

        mode = self._mode(
            modality
        )

        if x.ndim == 1:

            x = x.view(
                1,
                1,
                -1,
            )

            y = F.interpolate(
                x,
                size=self.target_seq_len,
                mode=mode,
                align_corners=
                    None if mode == "nearest"
                    else True,
            )

            return y.view(
                self.target_seq_len,
                1,
            )

        if x.ndim == 2:

            x = (
                x.T
                 .unsqueeze(0)
            )

            y = F.interpolate(
                x,
                size=self.target_seq_len,
                mode=mode,
                align_corners=
                    None if mode == "nearest"
                    else True,
            )

            return (
                y.squeeze(0)
                 .T
            )

        raise ValueError(
            f"Unsupported tensor shape "
            f"{x.shape}"
        )

    # --------------------------------------------------

    def synchronize(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Synchronize modalities.

        Labels remain untouched.
        """

        output = dict(sample)

        aligned = {}

        for (
            modality,
            data,
        ) in sample[
            "modalities"
        ].items():

            aligned[
                modality
            ] = self._resample(
                data,
                modality,
            )

        output[
            "modalities"
        ] = aligned

        return output

    # --------------------------------------------------

    def __call__(
        self,
        sample,
    ):
        return self.synchronize(
            sample
        )