"""
models.wrappers.baseline_b2
===========================

Baseline B2

Concatenation Fusion Baseline

Pipeline
--------

Modalities
    ↓

Feature Extractors
    ↓

Tokenizer
    ↓

Unified Concatenation Fusion
    ↓

Transformer Encoder-Decoder
    ↓

Recognition Head

Forecasting Head

Scientific Purpose
------------------

Evaluates whether simple projection +
concatenation fusion can achieve
competitive performance relative to
attention-based fusion.

Design Goals
------------

✓ Dataset Agnostic

✓ Modality Agnostic

✓ Label Agnostic

✓ Continuous Labels

✓ Discrete Labels

✓ Recognition Compatible

✓ Forecasting Compatible

✓ Config Driven

✓ Future Proof
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from models.encoders.multimodal_feature_extractors_v2 import (
    MultimodalFeatureExtractorsV2,
)

from models.tokenizers.tokenizer import (
    Tokenizer,
)

from models.fusion.unified_concat_fusion import (
    UnifiedConcatFusion,
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


class BaselineB2(nn.Module):
    """
    Projection + Concatenation Fusion Baseline.
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()

        # ==================================================
        # Config Handling
        # ==================================================

        if isinstance(
            config,
            dict,
        ):
            config = SimpleNamespace(
                **config
            )

        self.config = config

        # ==================================================
        # Modalities
        # ==================================================

        self.modalities = getattr(
            config,
            "modalities",
            {},
        )

        # ==================================================
        # Targets
        # ==================================================

        if hasattr(
            config,
            "continuous_targets",
        ):
            self.continuous_targets = (
                config.continuous_targets
            )

        elif hasattr(
            config,
            "continuous_labels",
        ):
            self.continuous_targets = list(
                config.continuous_labels
            )

        else:
            self.continuous_targets = []

        if hasattr(
            config,
            "discrete_targets",
        ):
            self.discrete_targets = (
                config.discrete_targets
            )

        elif hasattr(
            config,
            "discrete_labels",
        ):
            self.discrete_targets = {}

        else:
            self.discrete_targets = {}

        # ==================================================
        # Model Parameters
        # ==================================================

        model_cfg = getattr(
            config,
            "model",
            config,
        )

        self.embedding_dim = getattr(
            model_cfg,
            "emb_dim",
            128,
        )

        self.proj_dim = getattr(
            model_cfg,
            "proj_dim",
            64,
        )

        self.transformer_dim = getattr(
            model_cfg,
            "transformer_dim",
            256,
        )

        self.num_heads = getattr(
            model_cfg,
            "num_heads",
            4,
        )

        self.num_layers = getattr(
            model_cfg,
            "num_layers",
            4,
        )

        self.dropout = getattr(
            model_cfg,
            "dropout",
            0.1,
        )

        # ==================================================
        # Forecast
        # ==================================================

        self.forecast_horizon = getattr(
            config.forecasting,
            "horizon",
            25,
        )

        # ==================================================
        # History
        # ==================================================

        self.history_dim = getattr(
            config,
            "history_dim",
            0,
        )

        # ==================================================
        # Feature Extractors
        # ==================================================

        self.extractors = (
            MultimodalFeatureExtractorsV2(

                embedding_dim=
                self.embedding_dim,

                modalities=
                self.modalities,
            )
        )

        # ==================================================
        # Tokenizer
        # ==================================================

        self.tokenizer = (
            Tokenizer(
                embedding_dim=
                self.embedding_dim,
            )
        )

        # ==================================================
        # Concatenation Fusion
        # ==================================================

        self.fusion = (
            UnifiedConcatFusion(

                emb_dim=
                self.embedding_dim,

                proj_dim=
                self.proj_dim,

                transformer_dim=
                self.transformer_dim,
            )
        )

        # ==================================================
        # Transformer
        # ==================================================

        self.temporal_model = (
            TransformerEncoderDecoder(

                d_model=
                self.transformer_dim,

                nhead=
                self.num_heads,

                num_encoder_layers=
                self.num_layers,

                num_decoder_layers=
                self.num_layers,

                dim_feedforward=
                4 * self.transformer_dim,

                dropout=
                self.dropout,

                forecast_horizon=
                self.forecast_horizon,

                history_dim=
                self.history_dim,
            )
        )

        # ==================================================
        # Recognition Head
        # ==================================================

        self.recognition_head = (
            RecognitionHead(

                continuous_targets=
                self.continuous_targets,

                discrete_targets=
                self.discrete_targets,

                input_dim=
                self.transformer_dim,
            )
        )

        # ==================================================
        # Forecast Head
        # ==================================================

        self.forecasting_head = (
            ForecastingHead(

                continuous_targets=
                self.continuous_targets,

                discrete_targets=
                self.discrete_targets,

                input_dim=
                self.transformer_dim,
            )
        )

    # ======================================================
    # Forward
    # ======================================================

    def forward(
        self,
        batch,
    ):

        modalities = batch[
            "modalities"
        ]

        embeddings = (
            self.extractors(
                modalities
            )
        )

        tokens = (
            self.tokenizer(
                embeddings
            )
        )

        fused = (
            self.fusion(
                tokens
            )
        )

        label_history = batch.get(
            "label_history",
            None,
        )

        (
            encoder_outputs,
            decoder_outputs,
        ) = self.temporal_model(

            fused=fused,

            label_history=
            label_history,
        )

        recognition_outputs = (
            self.recognition_head(
                encoder_outputs
            )
        )

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

            "tokens":
                tokens,

            "fused":
                fused,
        }

    # ======================================================
    # Metadata
    # ======================================================

    def summary(
        self,
    ):

        return {

            "model":
                "BaselineB2",

            "fusion":
                "UnifiedConcatFusion",

            "embedding_dim":
                self.embedding_dim,

            "proj_dim":
                self.proj_dim,

            "transformer_dim":
                self.transformer_dim,

            "forecast_horizon":
                self.forecast_horizon,

            "continuous_targets":
                self.continuous_targets,

            "discrete_targets":
                self.discrete_targets,

            "history_dim":
                self.history_dim,
        }