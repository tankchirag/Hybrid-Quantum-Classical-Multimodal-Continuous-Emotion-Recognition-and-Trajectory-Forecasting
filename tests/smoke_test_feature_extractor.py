"""
tests.smoke_test_feature_extractor
==================================

Feature Extractor Smoke Test

Purpose
-------
Validate:

    Disk Dataset
        ↓
    Timeline Synchronizer
        ↓
    Window Sampler
        ↓
    Collate
        ↓
    MultimodalFeatureExtractorsV2

Checks
------

✓ Dataset loading

✓ Synchronization

✓ Window extraction

✓ Batch creation

✓ Feature extraction

✓ Output embeddings

✓ Shape consistency

✓ Active modalities

Design Goals
------------

✓ Dataset Agnostic

✓ Modality Agnostic

✓ Future Proof
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


# ==========================================================
# Config
# ==========================================================

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "ulm_tsst_v1"
)

HISTORY_LENGTH = 50

FORECAST_HORIZON = 25

BATCH_SIZE = 4

EMBEDDING_DIM = 128


# ==========================================================
# Helpers
# ==========================================================

def header(
    title: str,
):

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


# ==========================================================
# Main
# ==========================================================

def main():

    header(
        "FEATURE EXTRACTOR SMOKE TEST"
    )

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------

    dataset = ULMTSSTDiskDataset(

        dataset_root=
            DATASET_ROOT,

        split="train",
    )

    print(
        f"\nDataset Size: {len(dataset)}"
    )

    # ------------------------------------------------------
    # Synchronizer
    # ------------------------------------------------------

    synchronizer = (
        TimelineSynchronizer()
    )

    # ------------------------------------------------------
    # Window Sampler
    # ------------------------------------------------------

    sampler = WindowSampler(

        history_length=
            HISTORY_LENGTH,

        forecast_horizon=
            FORECAST_HORIZON,
    )

    # ------------------------------------------------------
    # Build Batch
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
        "\n[PASS] Batch created"
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

    header(
        "FEATURE EXTRACTION"
    )

    embeddings = extractor(

        batch[
            "modalities"
        ]
    )

    print(
        "\n[PASS] Forward successful"
    )

    # ------------------------------------------------------
    # Embedding Shapes
    # ------------------------------------------------------

    header(
        "EMBEDDING SHAPES"
    )

    for (
        modality,
        embedding,
    ) in embeddings.items():

        print(

            f"{modality:<15}"

            f"{tuple(embedding.shape)}"
        )

        # ----------------------------------
        # Expected:
        #
        # (B,T,D)
        # ----------------------------------

        assert (
            embedding.ndim
            == 3
        )

        assert (
            embedding.shape[0]
            ==
            BATCH_SIZE
        )

        assert (
            embedding.shape[1]
            ==
            HISTORY_LENGTH
        )

        assert (
            embedding.shape[2]
            ==
            EMBEDDING_DIM
        )

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    header(
        "ACTIVE MODALITIES"
    )

    print(
        extractor.active_modalities
    )

    print(
        f"\nTotal: "
        f"{len(extractor.active_modalities)}"
    )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    header(
        "PIPELINE STATUS"
    )

    print(
        "\n✅ Disk Dataset"
    )

    print(
        "✅ Timeline Synchronizer"
    )

    print(
        "✅ Window Sampler"
    )

    print(
        "✅ Collate"
    )

    print(
        "✅ Feature Extractors"
    )

    print(
        "\n✅ Encoder Integration Ready"
    )


if __name__ == "__main__":
    main()