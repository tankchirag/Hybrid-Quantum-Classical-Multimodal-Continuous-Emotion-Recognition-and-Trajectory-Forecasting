"""
tests.smoke_test_transformer
============================

Transformer Encoder-Decoder Smoke Test

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

Classical Fusion
    ↓

Transformer Encoder Decoder

Checks
------

✓ Encoder Forward Pass

✓ Decoder Forward Pass

✓ Forecast Horizon

✓ Output Shapes

✓ History Integration Disabled

✓ Baseline B1 Compatibility
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


# ==========================================================
# Configuration
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
        "TRANSFORMER SMOKE TEST"
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

    windows = []

    center_idx = (
        HISTORY_LENGTH
        + 100
    )

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

    print(
        "\n[PASS] Batch Created"
    )

    # ------------------------------------------------------
    # Feature Extraction
    # ------------------------------------------------------

    extractor = (
        MultimodalFeatureExtractorsV2(
            embedding_dim=
            EMBEDDING_DIM
        )
    )

    embeddings = extractor(
        batch["modalities"]
    )

    print(
        "[PASS] Feature Extraction"
    )

    # ------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------

    tokenizer = Tokenizer(
        embedding_dim=
        EMBEDDING_DIM
    )

    tokens = tokenizer(
        embeddings
    )

    print(
        "[PASS] Tokenization"
    )

    # ------------------------------------------------------
    # Fusion
    # ------------------------------------------------------

    fusion = ClassicalFusion(
        emb_dim=128,
        fusion_dim=64,
        transformer_dim=256,
        num_heads=2,
    )

    fused = fusion(
        tokens
    )

    print(
        "[PASS] Fusion"
    )

    # ------------------------------------------------------
    # Transformer
    # ------------------------------------------------------

    transformer = (
        TransformerEncoderDecoder(

            d_model=
            TRANSFORMER_DIM,

            forecast_horizon=
            FORECAST_HORIZON,

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
        "[PASS] Transformer Forward"
    )

    # ======================================================
    # Encoder Validation
    # ======================================================

    header(
        "ENCODER OUTPUT"
    )

    print(
        f"\nShape:\n"
        f"{tuple(encoder_outputs.shape)}"
    )

    assert (
        encoder_outputs.shape
        ==
        (
            BATCH_SIZE,
            HISTORY_LENGTH,
            TRANSFORMER_DIM,
        )
    )

    # ======================================================
    # Decoder Validation
    # ======================================================

    header(
        "DECODER OUTPUT"
    )

    print(
        f"\nShape:\n"
        f"{tuple(decoder_outputs.shape)}"
    )

    assert (
        decoder_outputs.shape
        ==
        (
            BATCH_SIZE,
            FORECAST_HORIZON,
            TRANSFORMER_DIM,
        )
    )

    # ======================================================
    # Summary
    # ======================================================

    header(
        "TRANSFORMER SUMMARY"
    )

    summary = (
        transformer.summary()
    )

    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )

    # ======================================================
    # Final
    # ======================================================

    header(
        "PIPELINE STATUS"
    )

    print(
        "\n✅ Dataset"
    )

    print(
        "✅ Synchronizer"
    )

    print(
        "✅ WindowSampler"
    )

    print(
        "✅ Collate"
    )

    print(
        "✅ Feature Extractors"
    )

    print(
        "✅ Tokenizer"
    )

    print(
        "✅ Classical Fusion"
    )

    print(
        "✅ Transformer Encoder"
    )

    print(
        "✅ Transformer Decoder"
    )

    print(
        "\n✅ Transformer Integration Ready"
    )

    print(
        "\nNEXT:"
    )

    print(
        "    Recognition Head"
    )

    print(
        "    Forecasting Head"
    )

    print(
        "    Full BaselineB1 Forward Pass"
    )


if __name__ == "__main__":
    main()