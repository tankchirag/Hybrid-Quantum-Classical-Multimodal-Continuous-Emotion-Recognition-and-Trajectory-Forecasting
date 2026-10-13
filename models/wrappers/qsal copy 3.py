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

import torch
import torch.nn as nn

from models.encoders.multimodal_feature_extractors_v2 import (
    MultimodalFeatureExtractors,
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
    QSAL Wrapper.

    Proposed Hybrid Quantum-Classical Model.
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
        # Feature Extractors
        # --------------------------------------------------

        self.extractors = (
            MultimodalFeatureExtractors(
                config=config,
            )
        )

        # --------------------------------------------------
        # QSAL Fusion
        # --------------------------------------------------

        self.fusion = (
            QSALFusion(
                emb_dim=128,
                model_dim=256,
                fused_dim=getattr(
                    config,
                    "fusion_dim",
                    64,
                ),
                sources=getattr(
                    config,
                    "sources",
                    None,
                ),
                init_sigma=getattr(
                    config,
                    "init_sigma",
                    1.0,
                ),
                learnable_sigma=getattr(
                    config,
                    "learnable_sigma",
                    True,
                ),
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
        batch : dict

        Returns
        -------
        dict
        """

        # ----------------------------------
        # Feature Extraction
        # ----------------------------------

        embeddings = (
            self.extractors(
                batch
            )
        )

        # ----------------------------------
        # QSAL Fusion
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

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    def summary(
        self,
    ) -> dict:

        return {

            "model":
                "QSAL",

            "fusion":
                "QSALFusion",

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
            f"continuous_targets="
            f"{len(self.continuous_targets)}, "
            f"discrete_targets="
            f"{len(self.discrete_targets)}"
        )