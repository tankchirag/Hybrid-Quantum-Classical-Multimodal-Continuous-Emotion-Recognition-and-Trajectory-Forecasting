"""
tests.smoke_test_training_pipeline
==================================

Training Pipeline Smoke Test

Purpose
-------
Verify the complete preprocessing pipeline:

    Disk Dataset
        ↓
    Timeline Synchronizer
        ↓
    Window Sampler
        ↓
    Collate
        ↓
    Training Batch

Design Goals
------------

✓ Dataset Agnostic

✓ Modality Agnostic

✓ Label Agnostic

✓ Recognition Compatible

✓ Forecasting Compatible

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


# ==========================================================
# Pretty Print
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
        "TRAINING PIPELINE SMOKE TEST"
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
        f"\nDataset Size: "
        f"{len(dataset)}"
    )

    # ------------------------------------------------------
    # Synchronizer
    # ------------------------------------------------------

    synchronizer = (
        TimelineSynchronizer()
    )

    # ------------------------------------------------------
    # Sampler
    # ------------------------------------------------------

    sampler = WindowSampler(

        history_length=
            HISTORY_LENGTH,

        forecast_horizon=
            FORECAST_HORIZON,
    )

    header(
        "SINGLE SAMPLE TEST"
    )

    disk_sample = dataset[0]

    synchronized = (
        synchronizer(
            disk_sample
        )
    )

    # ------------------------------------------
    # Determine valid center index
    # ------------------------------------------

    reference_label = next(

        iter(
            synchronized[
                "continuous_labels"
            ].values()
        )
    )

    center_idx = (

        HISTORY_LENGTH

        + 100
    )

    window = sampler.sample(

        synchronized_clip=
            synchronized,

        center_idx=
            center_idx,
    )

    print(
        "\n[PASS] Window created"
    )

    print(
        f"\nModalities: "
        f"{len(window.modalities)}"
    )

    print(
        f"Continuous Labels: "
        f"{len(window.continuous_labels)}"
    )

    print(
        f"Discrete Labels: "
        f"{len(window.discrete_labels)}"
    )

    # ======================================================
    # Batch Build
    # ======================================================

    header(
        "BATCH BUILD TEST"
    )

    windows = []

    for sample_idx in range(
        BATCH_SIZE
    ):

        disk_sample = (
            dataset[
                sample_idx
            ]
        )

        synchronized = (
            synchronizer(
                disk_sample
            )
        )

        window = sampler.sample(

            synchronized,

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
        "\n[PASS] Batch built"
    )

    print(
        f"\nBatch Size: "
        f"{batch['batch_size']}"
    )

    # ======================================================
    # Modalities
    # ======================================================

    header(
        "MODALITY SHAPES"
    )

    for (
        modality,
        tensor,
    ) in batch[
        "modalities"
    ].items():

        print(

            f"{modality:<15}"

            f"{tuple(tensor.shape)}"
        )

        assert (
            tensor.shape[0]
            ==
            BATCH_SIZE
        )

    # ======================================================
    # Continuous Labels
    # ======================================================

    header(
        "CONTINUOUS LABEL SHAPES"
    )

    for (
        label,
        tensor,
    ) in batch[
        "continuous_labels"
    ].items():

        print(

            f"{label:<15}"

            f"{tuple(tensor.shape)}"
        )

    # ======================================================
    # Discrete Labels
    # ======================================================

    header(
        "DISCRETE LABEL SHAPES"
    )

    for (
        label,
        tensor,
    ) in batch[
        "discrete_labels"
    ].items():

        print(

            f"{label:<20}"

            f"{tuple(tensor.shape)}"
        )

    # ======================================================
    # Recognition Targets
    # ======================================================

    header(
        "RECOGNITION TARGETS"
    )

    for (
        label,
        tensor,
    ) in batch[
        "recognition_targets"
    ][
        "continuous"
    ].items():

        print(

            f"{label:<15}"

            f"{tuple(tensor.shape)}"
        )

    # ======================================================
    # Forecast Targets
    # ======================================================

    header(
        "FORECAST TARGETS"
    )

    for (
        label,
        tensor,
    ) in batch[
        "forecast_targets"
    ][
        "continuous"
    ].items():

        print(

            f"{label:<15}"

            f"{tuple(tensor.shape)}"
        )

    # ======================================================
    # Final Check
    # ======================================================

    header(
        "PIPELINE STATUS"
    )

    print(
        "\n✅ Disk Dataset"
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
        "✅ Batch Creation"
    )

    print(
        "\n✅ Training Pipeline Ready"
    )


if __name__ == "__main__":
    main()