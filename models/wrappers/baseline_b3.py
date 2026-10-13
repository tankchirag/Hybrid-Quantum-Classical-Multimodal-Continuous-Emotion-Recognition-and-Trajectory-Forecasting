"""
models.wrappers.baseline_b3
===========================

Baseline B3

Parameterized Quantum Circuit Fusion Baseline

Pipeline
--------

Modalities
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

Baseline B3 evaluates whether
Parameterized Quantum Circuits (PQCs)
can model cross-source relationships
without relying on classical attention
mechanisms.

Fusion Pipeline
---------------

Source Embeddings

    (B,T,128)

        ↓

Source Compression

    128 → 16

        ↓

Source Aggregation

    16 → 64

        ↓

Quantum Encoding

    64 → Q

        ↓

Parameterized Quantum Circuit

        ↓

Quantum Representation

    Q → 256

        ↓

Transformer Encoder-Decoder

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

✓ Variable Qubit Counts

✓ Future Proof
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from models.encoders.multimodal_feature_extractors_v2 import (
    MultimodalFeatureExtractorsV2,
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


class BaselineB3(nn.Module):
    """
    Parameterized Quantum Circuit Fusion Baseline.

    B3 replaces classical fusion mechanisms
    with a trainable quantum fusion module while
    preserving the common temporal modelling and
    prediction infrastructure used across the thesis.
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

            #
            # Class counts are currently
            # dataset dependent.
            #

            self.discrete_targets = {}

        else:

            self.discrete_targets = {}

        # ==================================================
        # Model Configuration
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
            16,
        )

        self.fused_dim = getattr(
            model_cfg,
            "fused_dim",
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
        # Forecast Horizon
        # ==================================================

        self.forecast_horizon = getattr(
            config.forecasting,
            "horizon",
            25,
        )

        # ==================================================
        # History Configuration
        # ==================================================

        self.history_dim = getattr(
            config,
            "history_dim",
            0,
        )

        # ==================================================
        # PQC Configuration
        # ==================================================

        if isinstance(config, dict):

            pqc_cfg = config.get(
                "pqc",
                {},
            )

        else:

            pqc_cfg = getattr(
                config,
                "pqc",
                {},
            )

        if not isinstance(
            pqc_cfg,
            dict,
        ):
            pqc_cfg = vars(
                pqc_cfg
            )

        self.num_qubits = pqc_cfg.get(
            "num_qubits",
            4,
        )

        self.pqc_depth = pqc_cfg.get(
            "depth",
            3,
        )

        self.pqc_backend = pqc_cfg.get(
            "backend",
            "lightning.qubit",
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
        # PQC Fusion
        # ==================================================

        self.fusion = (
            BaselineB3PQCFusion(
                config=config,
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

        Parameters
        ----------
        batch

            Output produced by:

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
        # PQC Fusion
        # ----------------------------------

        fused = (
            self.fusion(
                embeddings
            )
        )

        # ----------------------------------
        # Optional Label History
        # ----------------------------------

        label_history = batch.get(
            "label_history",
            None,
        )

        # ----------------------------------
        # Temporal Modelling
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
        # Outputs
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
                "BaselineB3",

            "fusion":
                "BaselineB3PQCFusion",

            "num_qubits":
                self.num_qubits,

            "depth":
                self.pqc_depth,

            "backend":
                self.pqc_backend,

            "embedding_dim":
                self.embedding_dim,

            "proj_dim":
                self.proj_dim,

            "fused_dim":
                self.fused_dim,

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

    def extra_repr(
        self,
    ) -> str:

        return (

            f"qubits={self.num_qubits}, "

            f"depth={self.pqc_depth}, "

            f"backend={self.pqc_backend}, "

            f"continuous_targets="
            f"{len(self.continuous_targets)}, "

            f"discrete_targets="
            f"{len(self.discrete_targets)}"
        )