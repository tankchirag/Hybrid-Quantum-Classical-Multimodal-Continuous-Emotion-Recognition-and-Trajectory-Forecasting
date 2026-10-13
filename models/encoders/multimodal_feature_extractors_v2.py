"""
models.encoders.multimodal_feature_extractors_v2
================================================

Unified Multimodal Feature Extraction Framework

Purpose
-------

Provides a single entry point for all modality
encoders used throughout the thesis framework.

Design Goals
------------

✓ Dataset Agnostic

✓ Modality Agnostic

✓ Label Agnostic

✓ Recognition Compatible

✓ Forecasting Compatible

✓ Missing Modality Compatible

✓ Config Driven

✓ Future Proof
"""

from __future__ import annotations

from typing import Any
from typing import Dict

import torch
import torch.nn as nn

from .audio_encoder import AudioEncoder
from .video_encoder import VideoEncoder
from .text_encoder import TextEncoder

from .ecg_encoder import ECGEncoder
from .eda_encoder import EDAEncoder
from .eeg_encoder import EEGEncoder
from .emg_encoder import EMGEncoder
from .resp_encoder import RESPEncoder
from .bpm_encoder import BPMEncoder

from .eye_encoder import EyeEncoder


class MultimodalFeatureExtractorsV2(
    nn.Module,
):
    """
    Unified multimodal encoder manager.

    Supports both:

        dict

    and

        SimpleNamespace
        ConfigNamespace
        custom config objects
    """

    DEFAULT_MODALITIES = {

        "ecg": True,
        "eda": True,
        "eeg": True,
        "emg": True,
        "resp": True,
        "bpm": True,

        "eye": True,

        "audio": True,
        "video": True,
        "text": True,
    }

    def __init__(
        self,
        embedding_dim: int = 128,
        modalities: Any | None = None,
    ):
        super().__init__()

        self.embedding_dim = (
            embedding_dim
        )

        self.modalities = (
            self._normalize_modalities(
                modalities
            )
        )

        self.encoders = (
            nn.ModuleDict()
        )

        self._build_encoders()

    # ======================================================
    # Utilities
    # ======================================================

    def _normalize_modalities(
        self,
        modalities,
    ) -> Dict[str, bool]:
        """
        Convert various config types
        into a plain dictionary.
        """

        if modalities is None:

            return dict(
                self.DEFAULT_MODALITIES
            )

        # -----------------------------
        # Dict
        # -----------------------------

        if isinstance(
            modalities,
            dict,
        ):
            return dict(
                modalities
            )

        # -----------------------------
        # Namespace-like
        # -----------------------------

        if hasattr(
            modalities,
            "__dict__",
        ):
            return dict(
                vars(
                    modalities
                )
            )

        raise TypeError(

            "Unsupported modalities type: "

            f"{type(modalities)}"
        )

    # ======================================================
    # Build Encoders
    # ======================================================

    def _build_encoders(
        self,
    ):
        """
        Dynamically create enabled
        modality encoders.
        """

        # ----------------------------------
        # Physiological
        # ----------------------------------

        if self.modalities.get(
            "ecg",
            False,
        ):
            self.encoders["ecg"] = (
                ECGEncoder(
                    self.embedding_dim
                )
            )

        if self.modalities.get(
            "eda",
            False,
        ):
            self.encoders["eda"] = (
                EDAEncoder(
                    self.embedding_dim
                )
            )

        if self.modalities.get(
            "eeg",
            False,
        ):
            self.encoders["eeg"] = (
                EEGEncoder(
                    self.embedding_dim
                )
            )

        if self.modalities.get(
            "emg",
            False,
        ):
            self.encoders["emg"] = (
                EMGEncoder(
                    self.embedding_dim
                )
            )

        if self.modalities.get(
            "resp",
            False,
        ):
            self.encoders["resp"] = (
                RESPEncoder(
                    self.embedding_dim
                )
            )

        if self.modalities.get(
            "bpm",
            False,
        ):
            self.encoders["bpm"] = (
                BPMEncoder(
                    self.embedding_dim
                )
            )

        # ----------------------------------
        # Behavioral
        # ----------------------------------

        if self.modalities.get(
            "eye",
            False,
        ):
            self.encoders["eye"] = (
                EyeEncoder(
                    self.embedding_dim
                )
            )

        if self.modalities.get(
            "audio",
            False,
        ):
            self.encoders["audio"] = (
                AudioEncoder(
                    self.embedding_dim
                )
            )

        if self.modalities.get(
            "video",
            False,
        ):
            self.encoders["video"] = (
                VideoEncoder(
                    self.embedding_dim
                )
            )

        if self.modalities.get(
            "text",
            False,
        ):
            self.encoders["text"] = (
                TextEncoder(
                    self.embedding_dim
                )
            )

    # ======================================================
    # Forward
    # ======================================================

    def forward(
        self,
        modalities: Dict[
            str,
            torch.Tensor,
        ],
    ) -> Dict[
        str,
        torch.Tensor,
    ]:

        outputs = {}

        for (
            modality_name,
            encoder,
        ) in self.encoders.items():

            signal = modalities.get(
                modality_name,
                None,
            )

            if signal is None:
                continue

            outputs[
                modality_name
            ] = encoder(
                signal
            )

        return outputs

    # ======================================================
    # Metadata
    # ======================================================

    @property
    def active_modalities(
        self,
    ):

        return list(
            self.encoders.keys()
        )

    def encoder_summary(
        self,
    ) -> dict:

        return {

            "embedding_dim":
                self.embedding_dim,

            "active_modalities":
                self.active_modalities,

            "num_modalities":
                len(
                    self.active_modalities
                ),
        }


# ==========================================================
# Backward Compatibility
# ==========================================================

MultimodalFeatureExtractors = (
    MultimodalFeatureExtractorsV2
)