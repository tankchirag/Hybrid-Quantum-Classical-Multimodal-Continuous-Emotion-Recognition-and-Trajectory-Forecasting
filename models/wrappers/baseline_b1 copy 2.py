"""
models.wrappers.baseline_b1
===========================

Baseline B1

Classical Self-Attention Fusion Baseline

Pipeline
--------

Source Inputs
    ↓
Feature Extractors
    ↓
Tokenizer
    ↓
Classical Fusion
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

from models.temporal.transformer_ed import (
    TransformerEncoderDecoder,
)

from models.heads.recognition_head import (
    RecognitionHead,
)

from models.heads.forecasting_head import (
    ForecastingHead,
)


class BaselineB1(nn.Module):
    """
    Baseline B1.

    Classical Attention Fusion.
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
        # Tokenization
        # --------------------------------------------------

        self.tokenizer = (
            Tokenizer(
                embedding_dim=128,
            )
        )

        # --------------------------------------------------
        # Fusion
        # --------------------------------------------------

        self.fusion = (
            ClassicalFusion(
                emb_dim=128,
                fusion_dim=64,
                transformer_dim=256,
                num_heads=2,
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
        # Recognition
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
        # Forecasting
        # --------------------------------------------------

        self.forecasting_head = (
            ForecastingHead(
                continuous_targets=
                self.continuous_targets,

                discrete_targets=
                self.discrete_targets,
            )
        )

    def forward(
        self,
        batch: dict,
    ) -> dict:
        """
        Forward pass.

        Parameters
        ----------
        batch

            Input batch.

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
        # Tokenization
        # ----------------------------------

        tokens = self.tokenizer(
            embeddings
        )

        # ----------------------------------
        # Classical Fusion
        # ----------------------------------

        fused = self.fusion(
            tokens
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

        # ----------------------------------
        # Return
        # ----------------------------------

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

    def summary(
        self,
    ) -> dict:

        return {

            "model":
                "BaselineB1",

            "fusion":
                "ClassicalFusion",

            "continuous_targets":
                self.continuous_targets,

            "discrete_targets":
                self.discrete_targets,

            "history_dim":
                self.history_dim,
        }