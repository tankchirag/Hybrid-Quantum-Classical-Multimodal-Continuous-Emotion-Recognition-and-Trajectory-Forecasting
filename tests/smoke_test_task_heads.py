"""
tests.smoke_test_task_heads
===========================

Recognition + Forecasting Head Smoke Test

Pipeline
--------

Dataset
    ↓

TimelineSynchronizer
    ↓

WindowSampler
    ↓

Collate
    ↓

Feature Extractors
    ↓

Tokenizer
    ↓

Fusion
    ↓

Transformer

    ├── Encoder Outputs
    │       ↓
    │   Recognition Head
    │
    └── Decoder Outputs
            ↓
        Forecasting Head

Checks
------

✓ Recognition Head

✓ Forecasting Head

✓ Continuous Outputs

✓ Discrete Outputs

✓ Shape Validation

✓ Full Task Layer Integration
"""

from __future__ import annotations

import sys
from pathlib import Path


# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ==========================================================
# Imports
# ==========================================================

from datasets.loaders.ulm_tsst_disk_dataset import (
    ULMTSSTDiskDataset,
)

from datasets.preprocessing.timeline_synchronizer import (
    TimelineSynchronizer,
)

from datasets.sampling.window_sampler import (
    WindowSampler,
)

from datasets.loaders.collate import (
    collate,
)

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


# ==========================================================
# Config
# ==========================================================

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "ulm_tsst_v1"
)

BATCH_SIZE = 4

HISTORY_LENGTH = 50

FORECAST_HORIZON = 25

EMBEDDING_DIM = 128

TRANSFORMER_DIM = 256


CONTINUOUS_TARGETS = [

    "valence",
    "arousal",
    "dominance",
    "stress",
    "workload",
    "fatigue",
]

DISCRETE_TARGETS = {

    "emotion_class": 8,

    "stress_class": 3,

    "workload_class": 3,

    "alertness_class": 3,

    "drowsiness_class": 3,
}


# ==========================================================
# Helper
# ==========================================================

def header(title):

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


# ==========================================================
# Main
# ==========================================================

def main():

    header(
        "TASK HEADS SMOKE TEST"
    )

    # ------------------------------------------------------
    # Dataset Pipeline
    # ------------------------------------------------------

    dataset = ULMTSSTDiskDataset(
        dataset_root=DATASET_ROOT,
        split="train",
    )

    synchronizer = TimelineSynchronizer()

    sampler = WindowSampler(
        history_length=HISTORY_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
    )

    center_idx = (
        HISTORY_LENGTH
        + 100
    )

    windows = []

    for i in range(BATCH_SIZE):

        sample = dataset[i]

        sample = synchronizer(
            sample
        )

        sample = sampler.sample(
            synchronized_clip=sample,
            center_idx=center_idx,
        )

        windows.append(
            sample
        )

    batch = collate(
        windows
    )

    # ------------------------------------------------------
    # Feature Extraction
    # ------------------------------------------------------

    extractor = (
        MultimodalFeatureExtractorsV2(
            embedding_dim=EMBEDDING_DIM
        )
    )

    embeddings = extractor(
        batch["modalities"]
    )

    # ------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------

    tokenizer = Tokenizer(
        embedding_dim=EMBEDDING_DIM
    )

    tokens = tokenizer(
        embeddings
    )

    # ------------------------------------------------------
    # Fusion
    # ------------------------------------------------------

    fusion = ClassicalFusion()

    fused = fusion(
        tokens
    )

    # ------------------------------------------------------
    # Transformer
    # ------------------------------------------------------

    transformer = (
        TransformerEncoderDecoder(
            d_model=TRANSFORMER_DIM,
            forecast_horizon=FORECAST_HORIZON,
            history_dim=0,
        )
    )

    encoder_outputs, decoder_outputs = (
        transformer(
            fused=fused,
            label_history=None,
        )
    )

    print(
        "\n[PASS] Transformer Forward"
    )

    # ------------------------------------------------------
    # Recognition Head
    # ------------------------------------------------------

    recognition_head = (
        RecognitionHead(
            continuous_targets=
            CONTINUOUS_TARGETS,

            discrete_targets=
            DISCRETE_TARGETS,
        )
    )

    recognition_outputs = (
        recognition_head(
            encoder_outputs
        )
    )

    print(
        "[PASS] Recognition Head"
    )

    # ------------------------------------------------------
    # Forecasting Head
    # ------------------------------------------------------

    forecasting_head = (
        ForecastingHead(
            continuous_targets=
            CONTINUOUS_TARGETS,

            discrete_targets=
            DISCRETE_TARGETS,
        )
    )

    forecasting_outputs = (
        forecasting_head(
            decoder_outputs
        )
    )

    print(
        "[PASS] Forecasting Head"
    )

    # ======================================================
    # Recognition Validation
    # ======================================================

    header(
        "RECOGNITION OUTPUTS"
    )

    for (
        name,
        tensor,
    ) in recognition_outputs[
        "continuous"
    ].items():

        print(
            f"{name:<15}"
            f"{tuple(tensor.shape)}"
        )

        assert tensor.shape == (
            BATCH_SIZE,
            HISTORY_LENGTH,
        )

    # ======================================================
    # Forecast Validation
    # ======================================================

    header(
        "FORECAST OUTPUTS"
    )

    for (
        name,
        tensor,
    ) in forecasting_outputs[
        "continuous"
    ].items():

        print(
            f"{name:<15}"
            f"{tuple(tensor.shape)}"
        )

        assert tensor.shape == (
            BATCH_SIZE,
            FORECAST_HORIZON,
        )

    # ======================================================
    # Discrete Checks
    # ======================================================

    header(
        "DISCRETE TARGETS"
    )

    for (
        name,
        tensor,
    ) in recognition_outputs[
        "discrete"
    ].items():

        print(
            f"{name:<20}"
            f"{tuple(tensor.shape)}"
        )

    for (
        name,
        tensor,
    ) in forecasting_outputs[
        "discrete"
    ].items():

        print(
            f"{name:<20}"
            f"{tuple(tensor.shape)}"
        )

    # ======================================================
    # Final
    # ======================================================

    header(
        "PIPELINE STATUS"
    )

    print("\n✅ Dataset")
    print("✅ Synchronizer")
    print("✅ WindowSampler")
    print("✅ Collate")
    print("✅ Feature Extractors")
    print("✅ Tokenizer")
    print("✅ Fusion")
    print("✅ Transformer")
    print("✅ Recognition Head")
    print("✅ Forecasting Head")

    print(
        "\n✅ Task Layer Ready"
    )

    print(
        "\nNEXT:"
    )

    print(
        "    tests/smoke_test_baseline_b1.py"
    )


if __name__ == "__main__":
    main()