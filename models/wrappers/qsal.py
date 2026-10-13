# models/wrappers/qsal.py

"""
models.wrappers.qsal
====================

QSAL

Quantum Self-Attention Layer

This wrapper implements the proposed
hybrid quantum-classical architecture.

Pipeline
--------

Source Inputs
    ↓

Feature Extractors
    ↓

QSAL Fusion
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

from types import SimpleNamespace

import torch
import torch.nn as nn

from models.encoders.multimodal_feature_extractors_v2 import (
    MultimodalFeatureExtractorsV2,
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


class QSAL(nn.Module):
    """
    Quantum Self-Attention Learning.

    Main thesis architecture.

    Pipeline

        Modalities
            ↓

        Feature Extractors
            ↓

        QSAL Fusion
            ↓

        Transformer Encoder Decoder
            ↓

        Recognition Head
            ↓

        Forecasting Head
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()

        # ==================================================
        # Config
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
        # QSAL Configuration
        # ==================================================

        if isinstance(config, dict):

            qsal_cfg = config.get(
                "qsal",
                {},
            )

        else:

            qsal_cfg = getattr(
                config,
                "qsal",
                {},
            )

        if not isinstance(
            qsal_cfg,
            dict,
        ):

            qsal_cfg = vars(
                qsal_cfg
            )

        self.num_qubits = qsal_cfg.get(
            "num_qubits",
            4,
        )

        self.circuit_depth = qsal_cfg.get(
            "circuit_depth",
            2,
        )

        self.fusion_dim = qsal_cfg.get(
            "fusion_dim",
            64,
        )

        self.backend = qsal_cfg.get(
            "backend",
            "lightning.qubit",
        )

        self.encoding = qsal_cfg.get(
            "encoding",
            "angle",
        )

        self.entanglement = qsal_cfg.get(
            "entanglement",
            "strong",
        )

        self.shots = qsal_cfg.get(
            "shots",
            None,
        )

        self.forecast_horizon = getattr(
            config.forecasting,
            "horizon",
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
        # QSAL Fusion
        # ==================================================

        self.fusion = (
            QSALFusion(
                emb_dim=self.embedding_dim,

                model_dim=self.transformer_dim,

                fused_dim=self.fusion_dim,

                num_qubits=self.num_qubits,

                circuit_depth=self.circuit_depth,

                backend=self.backend,

                sources=None,
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

        fused = (
            self.fusion(
                embeddings
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
                "QSAL",

            "fusion":
                "QSALFusion",

            "num_qubits":
                self.num_qubits,

            "circuit_depth":
                self.circuit_depth,

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

            f"depth={self.circuit_depth}, "

            f"continuous_targets="
            f"{len(self.continuous_targets)}, "

            f"discrete_targets="
            f"{len(self.discrete_targets)}"
        )