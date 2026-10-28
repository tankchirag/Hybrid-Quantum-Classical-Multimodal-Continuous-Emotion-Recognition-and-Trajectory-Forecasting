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

from models.history.adapter import (
    HistoryAdapter,
)

from models.history.representation import (
    HistoryRepresentation,
)

from models.history.encoder import (
    HistoryEncoder,
)

from models.history.injection import (
    HistoryInjection,
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

        else:

            dataset_cfg = getattr(
                config,
                "dataset",
                None,
            )

            self.discrete_targets = {}

            if (
                dataset_cfg is not None
                and
                hasattr(
                    dataset_cfg,
                    "discrete_cardinalities",
                )
            ):

                discrete_cardinalities = (
                    dataset_cfg
                    .discrete_cardinalities
                )

                if isinstance(
                    discrete_cardinalities,
                    dict,
                ):

                    self.discrete_targets = (
                        discrete_cardinalities
                    )

                elif hasattr(
                    discrete_cardinalities,
                    "__dict__",
                ):

                    self.discrete_targets = dict(
                        vars(
                            discrete_cardinalities
                        )
                    )


        # ----------------------------------
        # Model Parameters
        # ----------------------------------

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

        self.fusion_dim = getattr(
            model_cfg,
            "fusion_dim",
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

        forecast_cfg = getattr(
            config,
            "forecasting",
            None,
        )

        if (
            forecast_cfg is not None
            and
            hasattr(
                forecast_cfg,
                "horizon",
            )
        ):
            self.forecast_horizon = (
                forecast_cfg.horizon
            )
        else:
            self.forecast_horizon = getattr(
                config,
                "forecast_horizon",
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

        self.history_ratio = None

        history_cfg = None

        # ----------------------------------
        # Safe defaults
        # ----------------------------------

        history_hidden_dim = None

        history_ratio = 4

        history_auto_scale = True

        history_num_layers = 2

        history_dropout = 0.20

        history_max_continuous_labels = 16

        if isinstance(
            config,
            dict,
        ):
            history_cfg = config.get(
                "history_encoder",
                {},
            )
        else:
            history_cfg = getattr(
                config,
                "history_encoder",
                None,
            )

        self.history_enabled = False

        if history_cfg is not None:
            if isinstance(
                history_cfg,
                dict,
            ):
                self.history_enabled = bool(
                    history_cfg.get(
                        "enabled",
                        False,
                    )
                )

                history_hidden_dim = history_cfg.get(
                    "hidden_dim",
                    None,
                )

                history_ratio = history_cfg.get(
                    "ratio",
                    4,
                )

                self.history_ratio = history_ratio

                history_auto_scale = history_cfg.get(
                    "auto_scale",
                    True,
                )

                history_num_layers = history_cfg.get(
                    "num_layers",
                    2,
                )

                history_dropout = history_cfg.get(
                    "history_dropout",
                    0.20,
                )

                history_max_continuous_labels = (
                    history_cfg.get(
                        "max_continuous_labels",
                        16,
                    )
                )

            else:
                self.history_enabled = bool(
                    getattr(
                        history_cfg,
                        "enabled",
                        False,
                    )
                )
                history_hidden_dim = getattr(
                    history_cfg,
                    "hidden_dim",
                    None,
                )
                history_ratio = getattr(
                    history_cfg,
                    "ratio",
                    4,
                )

                self.history_ratio = history_ratio

                history_auto_scale = getattr(
                    history_cfg,
                    "auto_scale",
                    True,
                )
                history_num_layers = getattr(
                    history_cfg,
                    "num_layers",
                    2,
                )
                history_dropout = getattr(
                    history_cfg,
                    "history_dropout",
                    0.20,
                )
                history_max_continuous_labels = getattr(
                    history_cfg,
                    "max_continuous_labels",
                    16,
                )
        else:
            history_hidden_dim = None
            history_num_layers = 2
            history_dropout = 0.20
            history_max_continuous_labels = 16

        if self.history_enabled:
            if history_hidden_dim is not None:
                self.history_dim = int(
                    history_hidden_dim
                )
            elif self.history_dim <= 0:

                if history_auto_scale:

                    self.history_dim = max(
                        8,
                        self.transformer_dim //
                        int(history_ratio)
                    )

                else:

                    self.history_dim = (
                        self.transformer_dim // 4
                    )

            self.history_max_continuous_labels = int(
                history_max_continuous_labels
            )

            discrete_cardinalities = getattr(
                getattr(
                    config,
                    "dataset",
                    {},
                ),
                "discrete_cardinalities",
                {},
            )

            if not isinstance(
                discrete_cardinalities,
                dict,
            ):
                if hasattr(
                    discrete_cardinalities,
                    "__dict__",
                ):
                    discrete_cardinalities = dict(
                        vars(
                            discrete_cardinalities
                        )
                    )
                else:
                    discrete_cardinalities = {}

            self.history_adapter = (
                HistoryAdapter()
            )

            self.history_representation = (
                HistoryRepresentation(
                    history_dim=
                    self.history_dim,
                    discrete_cardinalities=
                    discrete_cardinalities,
                    max_continuous_labels=
                    self.history_max_continuous_labels,
                )
            )

            self.history_encoder = (
                HistoryEncoder(
                    history_dim=
                    self.history_dim,
                    num_layers=
                    int(history_num_layers),
                    history_dropout=
                    float(history_dropout),
                )
            )

            self.history_injection = (
                HistoryInjection(
                    model_dim=
                    self.transformer_dim,
                    history_dim=
                    self.history_dim,
                )
            )
        else:
            self.history_adapter = None
            self.history_representation = None
            self.history_encoder = None
            self.history_injection = None


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

            token_order=
            list(
                self.modalities.keys()
            )
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
                0,
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


        print(
            self.extractors.active_modalities
        )

        # ----------------------------------
        # Feature Extraction
        # ----------------------------------

        embeddings = (
            self.extractors(
                modalities
            )
        )



        for k, v in embeddings.items():

            print(
                k,
                tuple(v.shape),
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

         #---------------------------------
        # History
        #---------------------------------

        history_state = None

        if self.history_enabled:

            history = self.history_adapter(

                continuous_labels=
                batch.get(
                    "continuous_labels",
                    None,
                ),

                discrete_labels=
                batch.get(
                    "discrete_labels",
                    None,
                ),
            )

            history_repr = (
                self.history_representation(

                    continuous=
                    history["continuous"],

                    discrete=
                    history["discrete"],

                    discrete_label_names=(
                        list(
                            batch[
                                "discrete_labels"
                            ].keys()
                        )
                        if batch.get(
                            "discrete_labels"
                        )
                        else None
                    ),
                )
            )

            history_state = (
                self.history_encoder(
                    history_repr
                )
            )

            fused = (
                self.history_injection(
                    fused,
                    history_state,
                )
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
            None,
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

            "history_state":
                history_state,

            "history_enabled":
                self.history_enabled,                
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

            "history_enabled":
                self.history_enabled,

            "history_ratio":
                self.history_ratio,
        }