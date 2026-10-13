"""
models.wrappers.baseline_b3
===========================

Baseline B3

Pure Quantum Fusion Baseline

Pipeline
--------

Source Inputs
    ↓

Feature Extractors
    ↓

PQC Fusion
    ↓

Transformer Encoder-Decoder
    ↓

Recognition Head
    ↓

Forecasting Head

Scientific Purpose
------------------

Baseline B3 evaluates whether a Parameterized
Quantum Circuit (PQC) can model cross-source
relationships without classical attention.

Fusion Pipeline

    Source Embeddings
        ↓

    128 → 16
        ↓

    Source Aggregation
        ↓

    16 → 64
        ↓

    Quantum Encoding
        ↓

    PQC
        ↓

    Q → 256
        ↓

    Transformer

Design Goals
------------

✓ Dataset Agnostic

✓ Label Agnostic

✓ Continuous Labels

✓ Discrete Labels

✓ Recognition Compatible

✓ Forecasting Compatible

✓ Variable Source Count

✓ Future Proof
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.encoders.multimodal_feature_extractors_v2 import (
    MultimodalFeatureExtractors,
)

from models.fusion.pqc_fusion import (
    BaselineB3PQCFusion,
)

from models.temporal.transformer_ed import (
    TransformerEncoderDecoder,
)

from models.heads.recognition_head import (
    RecognitionHead,
)

from models.heads.forecasting_head import (
    ForecastingHead,
)


class BaselineB3(
    nn.Module,
):
    """
    Baseline B3.

    Pure Quantum Fusion.
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.config = config

        # --------------------------------------------------
        # Targets
        # --------------------------------------------------

        self.continuous_targets = getattr(
            config,
            "continuous_targets",
            [],
        )

        self.discrete_targets = getattr(
            config,
            "discrete_targets",
            {},
        )

        self.history_dim = getattr(
            config,
            "history_dim",
            0,
        )

        # --------------------------------------------------
        # Feature Extraction
        # --------------------------------------------------

        self.extractors = (
            MultimodalFeatureExtractors(
                config=config,
            )
        )

        # --------------------------------------------------
        # PQC Fusion
        # --------------------------------------------------

        self.fusion = (
            BaselineB3PQCFusion(
                config=config,
            )
        )

        # --------------------------------------------------
        # Temporal Model
        # --------------------------------------------------

        self.temporal_model = (
            TransformerEncoderDecoder(
                d_model=256,
                nhead=getattr(
                    config,
                    "num_heads",
                    4,
                ),
                num_encoder_layers=getattr(
                    config,
                    "num_layers",
                    4,
                ),
                num_decoder_layers=getattr(
                    config,
                    "num_layers",
                    4,
                ),
                dim_feedforward=1024,
                dropout=0.1,
                forecast_horizon=getattr(
                    config,
                    "forecast_horizon",
                    10,
                ),
                history_dim=self.history_dim,
            )
        )

        # --------------------------------------------------
        # Recognition Head
        # --------------------------------------------------

        self.recognition_head = (
            RecognitionHead(
                continuous_targets=
                self.continuous_targets,

                discrete_targets=
                self.discrete_targets,
            )
        )

        # --------------------------------------------------
        # Forecasting Head
        # --------------------------------------------------

        self.forecasting_head = (
            ForecastingHead(
                continuous_targets=
                self.continuous_targets,

                discrete_targets=
                self.discrete_targets,
            )
        )

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------

    def forward(
        self,
        batch: dict,
    ) -> dict:
        """
        Forward pass.

        Parameters
        ----------
        batch

            Input batch containing:

            source signals

            optional label history

        Returns
        -------
        dict
        """

        # ----------------------------------
        # Source Embeddings
        # ----------------------------------

        embeddings = (
            self.extractors(
                batch
            )
        )

        # ----------------------------------
        # PQC Fusion
        # ----------------------------------

        fused = self.fusion(
            embeddings
        )  # (B,T,256)

        # ----------------------------------
        # Optional Label History
        # ----------------------------------

        label_history = batch.get(
            "label_history",
            None,
        )

        # ----------------------------------
        # Temporal Modeling
        # ----------------------------------

        (
            encoder_outputs,
            decoder_outputs,
        ) = self.temporal_model(
            fused=fused,
            label_history=label_history,
        )

        # ----------------------------------
        # Recognition
        # ----------------------------------

        recognition_outputs = (
            self.recognition_head(
                encoder_outputs
            )
        )

        # ----------------------------------
        # Forecasting
        # ----------------------------------

        forecasting_outputs = (
            self.forecasting_head(
                decoder_outputs
            )
        )

        return {

            "recognition":
                recognition_outputs,

            "forecasting":
                forecasting_outputs,

            "encoder_outputs":
                encoder_outputs,

            "decoder_outputs":
                decoder_outputs,
        }

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    def summary(
        self,
    ) -> dict:

        num_qubits = getattr(
            self.config,
            "num_qubits",
            4,
        )

        return {

            "model":
                "BaselineB3",

            "fusion":
                "BaselineB3PQCFusion",

            "num_qubits":
                num_qubits,

            "continuous_targets":
                self.continuous_targets,

            "discrete_targets":
                self.discrete_targets,

            "history_dim":
                self.history_dim,
        }

    def extra_repr(
        self,
    ) -> str:

        return (
            f"qubits="
            f"{getattr(self.config, 'num_qubits', 4)}, "
            f"continuous_targets="
            f"{len(self.continuous_targets)}, "
            f"discrete_targets="
            f"{len(self.discrete_targets)}"
        )