"""
models.wrappers.baseline_b1
===========================

Baseline B1

Classical Self-Attention Fusion Baseline

Pipeline
--------

Modalities
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

Forecasting Head

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
    Classical Attention Fusion Baseline.
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()

        # ----------------------------------
        # Config Handling
        # ----------------------------------

        if isinstance(
            config,
            dict,
        ):
            config = SimpleNamespace(
                **config
            )

        self.config = config

        # ----------------------------------
        # Modalities
        # ----------------------------------

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

            # Temporary thesis-safe default

            self.discrete_targets = {}

        else:
            self.discrete_targets = {}

        # ----------------------------------
        # Model Parameters
        # ----------------------------------

        self.embedding_dim = getattr(
            config,
            "embedding_dim",
            128,
        )

        self.fusion_dim = getattr(
            config,
            "fusion_dim",
            64,
        )

        self.transformer_dim = getattr(
            config,
            "transformer_dim",
            256,
        )

        self.num_heads = getattr(
            config,
            "num_heads",
            4,
        )

        self.num_layers = getattr(
            config,
            "num_layers",
            4,
        )

        self.dropout = getattr(
            config,
            "dropout",
            0.1,
        )

        self.forecast_horizon = getattr(
            config,
            "forecast_horizon",
            25,
        )

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

        self.tokenizer = Tokenizer(

            embedding_dim=
            self.embedding_dim,
        )

        # ==================================================
        # Fusion
        # ==================================================

        self.fusion = ClassicalFusion(

            emb_dim=
            self.embedding_dim,

            fusion_dim=
            self.fusion_dim,

            transformer_dim=
            self.transformer_dim,

            num_heads=
            self.num_heads,

            dropout=
            self.dropout,
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
        # Forecasting Head
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
        batch: dict,
    ) -> dict:
        """
        Forward pass.

        Input
        -----

        batch

            generated by:

                Dataset
                    ↓
                Synchronizer
                    ↓
                WindowSampler
                    ↓
                Collate

        Returns
        -------

        dict
        """

        # ----------------------------------
        # Modalities
        # ----------------------------------

        modalities = batch[
            "modalities"
        ]

        # ----------------------------------
        # Feature Extraction
        # ----------------------------------

        embeddings = (
            self.extractors(
                modalities
            )
        )

        # ----------------------------------
        # Tokenization
        # ----------------------------------

        tokens = self.tokenizer(
            embeddings
        )

        # ----------------------------------
        # Fusion
        # ----------------------------------

        fused = self.fusion(
            tokens
        )

        # ----------------------------------
        # Optional History
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

        # ----------------------------------
        # Output
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
    ) -> dict:

        return {

            "model":
                "BaselineB1",

            "fusion":
                "ClassicalFusion",

            "embedding_dim":
                self.embedding_dim,

            "fusion_dim":
                self.fusion_dim,

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