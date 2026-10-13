"""
tests.smoke_test_fusion
=======================

Classical Fusion Smoke Test

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

Checks
------

✓ Token Generation

✓ Fusion Forward Pass

✓ Fused Representation

✓ Shape Validation

✓ Transformer Compatibility
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

FUSION_DIM = 64

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
        "FUSION SMOKE TEST"
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

    for i in range(
        BATCH_SIZE
    ):

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
    # Tokenization
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

        emb_dim=
        EMBEDDING_DIM,

        fusion_dim=
        FUSION_DIM,

        transformer_dim=
        TRANSFORMER_DIM,

        num_heads=2,
    )

    fused = fusion(
        tokens
    )

    print(
        "[PASS] Fusion Forward"
    )

    # ------------------------------------------------------
    # Shape Validation
    # ------------------------------------------------------

    header(
        "FUSION OUTPUT SHAPE"
    )

    print(
        f"\nShape:\n"
        f"{tuple(fused.shape)}"
    )

    assert (
        fused.ndim
        == 3
    )

    B, T, D = fused.shape

    print(
        f"\nBatch Size      : {B}"
    )

    print(
        f"Time Steps      : {T}"
    )

    print(
        f"Feature Dim     : {D}"
    )

    assert (
        B == BATCH_SIZE
    )

    assert (
        T == HISTORY_LENGTH
    )

    assert (
        D == TRANSFORMER_DIM
    )

    print(
        "\n[PASS] Fusion Shape Validation"
    )

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    header(
        "FUSION SUMMARY"
    )

    summary = fusion.summary()

    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )

    # ------------------------------------------------------
    # Final
    # ------------------------------------------------------

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
        "\n✅ Fusion Integration Ready"
    )

    print(
        "\nNEXT:"
    )

    print(
        "    Transformer Encoder-Decoder"
    )

    print(
        "    (tests/smoke_test_transformer.py)"
    )


if __name__ == "__main__":
    main()