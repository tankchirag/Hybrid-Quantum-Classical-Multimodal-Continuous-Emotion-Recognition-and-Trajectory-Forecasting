"""
tests.smoke_test_tokenizer
==========================

Tokenizer Smoke Test

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

Checks
------

✓ Embedding Generation

✓ Token Generation

✓ Token Shapes

✓ Active Token Count

✓ Integration Compatibility
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


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
        "TOKENIZER SMOKE TEST"
    )

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------

    dataset = ULMTSSTDiskDataset(

        dataset_root=
            DATASET_ROOT,

        split="train",
    )

    synchronizer = (
        TimelineSynchronizer()
    )

    sampler = WindowSampler(

        history_length=
            HISTORY_LENGTH,

        forecast_horizon=
            FORECAST_HORIZON,
    )

    # ------------------------------------------------------
    # Batch
    # ------------------------------------------------------

    windows = []

    center_idx = (
        HISTORY_LENGTH
        + 100
    )

    for i in range(
        BATCH_SIZE
    ):

        sample = dataset[i]

        sample = (
            synchronizer(
                sample
            )
        )

        window = sampler.sample(

            synchronized_clip=
                sample,

            center_idx=
                center_idx,
        )

        windows.append(
            window
        )

    batch = collate(
        windows
    )

    print(
        "\n[PASS] Batch Created"
    )

    # ------------------------------------------------------
    # Feature Extractor
    # ------------------------------------------------------

    extractor = (

        MultimodalFeatureExtractorsV2(

            embedding_dim=
                EMBEDDING_DIM
        )
    )

    embeddings = extractor(

        batch[
            "modalities"
        ]
    )

    print(
        "[PASS] Embeddings Created"
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
        "[PASS] Tokenizer Forward"
    )

    # ------------------------------------------------------
    # Shape Checks
    # ------------------------------------------------------

    header(
        "TOKEN SHAPE"
    )

    print(
        f"\nTokens Shape:\n"
        f"{tuple(tokens.shape)}"
    )

    assert (
        tokens.ndim
        == 4
    )

    B, T, N, D = (
        tokens.shape
    )

    print(
        f"\nBatch Size      : {B}"
    )

    print(
        f"Time Steps      : {T}"
    )

    print(
        f"Token Count     : {N}"
    )

    print(
        f"Embedding Dim   : {D}"
    )

    # ------------------------------------------------------
    # Validation
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

    # ------------------------------------------------------
    # Additional Validation
    # ------------------------------------------------------

    header(
        "TOKEN VALIDATION"
    )

    print(
        f"\nExpected Batch Size : {BATCH_SIZE}"
    )

    print(
        f"Observed Batch Size : {B}"
    )

    print(
        f"Observed Time Steps : {T}"
    )

    print(
        f"Observed Tokens     : {N}"
    )

    print(
        f"Observed Embedding  : {D}"
    )

    assert B == BATCH_SIZE

    assert T == HISTORY_LENGTH

    assert D == EMBEDDING_DIM

    assert N > 0

    print(
        "\n[PASS] Token Shape Validation"
    )

    # ------------------------------------------------------
    # Token Metadata
    # ------------------------------------------------------

    header(
        "TOKENIZER METADATA"
    )

    print(
        f"\nSupported Tokens:"
    )

    for token_name in tokenizer.supported_tokens:

        print(
            f"  {token_name}"
        )

    print(
        f"\nTotal Supported Tokens: "
        f"{tokenizer.num_supported_tokens}"
    )

    # ------------------------------------------------------
    # Final Status
    # ------------------------------------------------------

    header(
        "INTEGRATION STATUS"
    )

    print(
        "\n✅ Dataset Integration"
    )

    print(
        "✅ Synchronization"
    )

    print(
        "✅ History Window Extraction"
    )

    print(
        "✅ Batch Construction"
    )

    print(
        "✅ Feature Extraction"
    )

    print(
        "✅ Tokenization"
    )

    print(
        "\n✅ Tokenizer Integration Ready"
    )

    print(
        "\nNEXT:"
    )

    print(
        "    Classical Fusion"
    )

    print(
        "    (tests/smoke_test_fusion.py)"
    )


if __name__ == "__main__":
    main()