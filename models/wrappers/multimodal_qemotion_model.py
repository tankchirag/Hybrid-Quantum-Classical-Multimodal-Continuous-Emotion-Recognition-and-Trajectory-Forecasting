"""
models.wrappers.multimodal_qemotion_model
=========================================

Unified Multimodal QEmotion Model

Purpose
-------

High-level model wrapper responsible for
constructing the complete architecture from
configuration.

Supported Fusion Methods
------------------------

B1

    ClassicalFusion

B2

    UnifiedConcatFusion

B3

    BaselineB3PQCFusion

QSAL

    QSALFusion

Pipeline
--------

Source Inputs
    ↓

Feature Extractors
    ↓

Tokenizer (where required)
    ↓

Fusion
    ↓

Transformer Encoder-Decoder
    ↓

Recognition Head
    ↓

Forecasting Head

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

Input
-----

Batch Dictionary

Output
------

{
    "recognition": ...,

    "forecasting": ...,

    "encoder_outputs": ...,

    "decoder_outputs": ...
}
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.encoders.multimodal_feature_extractors_v2 import (
    MultimodalFeatureExtractors,
)

from models.tokenizers.tokenizer import (
    Tokenizer,
)

from models.fusion.classical_fusion import (
    ClassicalFusion,
)

from models.fusion.unified_concat_fusion import (
    UnifiedConcatFusion,
)

from models.fusion.pqc_fusion import (
    BaselineB3PQCFusion,
)

from models.fusion.qsal_fusion import (
    QSALFusion,
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


class MultimodalQEmotionModel(
    nn.Module,
):
    """
    Unified model wrapper.

    Fusion type is selected dynamically.

    Supported

        B1
        B2
        B3
        QSAL
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
        # Fusion Selection
        # --------------------------------------------------

        self.fusion_type = getattr(
            config,
            "fusion_type",
            "qsal",
        ).lower()

        # --------------------------------------------------
        # Feature Extractors
        # --------------------------------------------------

        self.extractors = (
            MultimodalFeatureExtractors(
                config=config,
            )
        )

        # --------------------------------------------------
        # Optional Tokenizer
        # --------------------------------------------------

        self.tokenizer = (
            Tokenizer(
                embedding_dim=128,
            )
        )

        # --------------------------------------------------
        # Fusion Module
        # --------------------------------------------------

        if self.fusion_type == "b1":

            self.fusion = ClassicalFusion(
                emb_dim=128,
                fusion_dim=64,
                transformer_dim=256,
                num_heads=2,
            )

            self.requires_tokens = True

        elif self.fusion_type == "b2":

            self.fusion = UnifiedConcatFusion(
                emb_dim=128,
                proj_dim=64,
                transformer_dim=256,
            )

            self.requires_tokens = False

        elif self.fusion_type == "b3":

            self.fusion = (
                BaselineB3PQCFusion(
                    config=config,
                )
            )

            self.requires_tokens = False

        elif self.fusion_type == "qsal":

            self.fusion = QSALFusion(
                emb_dim=128,
                model_dim=256,
            )

            self.requires_tokens = False

        else:

            raise ValueError(
                f"Unsupported fusion type: "
                f"{self.fusion_type}"
            )

        # --------------------------------------------------
        # Temporal Backbone
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
    # Fusion Dispatch
    # ------------------------------------------------------

    def _run_fusion(
        self,
        embeddings,
    ):
        """
        Route embeddings to the selected
        fusion architecture.
        """

        if self.fusion_type == "b1":

            tokens = self.tokenizer(
                embeddings
            )

            return self.fusion(
                tokens
            )

        elif self.fusion_type == "b2":

            return self.fusion(
                embeddings
            )

        elif self.fusion_type == "b3":

            return self.fusion(
                embeddings
            )

        elif self.fusion_type == "qsal":

            return self.fusion(
                embeddings
            )

        raise RuntimeError(
            "Invalid fusion type"
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
        # Fusion
        # ----------------------------------

        fused = self._run_fusion(
            embeddings
        )

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
            label_history=
            label_history,
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

            "fusion_type":
                self.fusion_type,
        }

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    def summary(
        self,
    ) -> dict:

        return {

            "model":
                "MultimodalQEmotionModel",

            "fusion_type":
                self.fusion_type,

            "continuous_targets":
                self.continuous_targets,

            "discrete_targets":
                self.discrete_targets,

            "history_dim":
                self.history_dim,
        }