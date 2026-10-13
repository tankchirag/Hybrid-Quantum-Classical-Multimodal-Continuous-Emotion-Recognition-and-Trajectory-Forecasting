"""
tests.smoke_test_disk_dataset
=============================

ULM-TSST Disk Dataset Smoke Test

Purpose
-------
Verify that the disk-based dataset loader
works correctly before integrating with:

    Synchronizer

    WindowSampler

    DataLoader

    Training Pipeline

Checks
------

✓ Train split loading

✓ Validation split loading

✓ Test split loading

✓ Sample structure

✓ Modalities

✓ Continuous labels

✓ Discrete labels

✓ Metadata

✓ Tensor shapes

Expected Dataset Layout
-----------------------

data/generated/ulm_tsst_v1/

    train/

    val/

    test/
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
        str(PROJECT_ROOT)
    )


# ==========================================================
# Imports
# ==========================================================

from datasets.loaders.ulm_tsst_disk_dataset import (
    ULMTSSTDiskDataset,
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


# ==========================================================
# Pretty Print
# ==========================================================

def print_header(
    title: str,
):

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


# ==========================================================
# Inspect Sample
# ==========================================================

def inspect_sample(
    sample,
):

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    print_header(
        "METADATA"
    )

    for key, value in sample[
        "metadata"
    ].items():

        print(
            f"{key}: {value}"
        )

    # ------------------------------------------------------
    # Modalities
    # ------------------------------------------------------

    print_header(
        "MODALITIES"
    )

    total_mods = 0

    for (
        name,
        tensor,
    ) in sample[
        "modalities"
    ].items():

        assert isinstance(
            tensor,
            torch.Tensor,
        )

        total_mods += 1

        print(
            f"{name:<15}"
            f"{tuple(tensor.shape)}"
        )

    print(
        f"\nModalities: "
        f"{total_mods}"
    )

    # ------------------------------------------------------
    # Continuous Labels
    # ------------------------------------------------------

    print_header(
        "CONTINUOUS LABELS"
    )

    for (
        name,
        tensor,
    ) in sample[
        "continuous_labels"
    ].items():

        print(
            f"{name:<15}"
            f"{tuple(tensor.shape)}"
        )

    # ------------------------------------------------------
    # Discrete Labels
    # ------------------------------------------------------

    print_header(
        "DISCRETE LABELS"
    )

    for (
        name,
        tensor,
    ) in sample[
        "discrete_labels"
    ].items():

        print(
            f"{name:<20}"
            f"{tuple(tensor.shape)}"
        )


# ==========================================================
# Verify Split
# ==========================================================

def verify_split(
    split_name: str,
):

    print_header(
        f"VERIFYING SPLIT: {split_name.upper()}"
    )

    dataset = ULMTSSTDiskDataset(

        dataset_root=
        DATASET_ROOT,

        split=
        split_name,

        device="cpu",
    )

    print(
        f"\nDataset Length: "
        f"{len(dataset)}"
    )

    assert len(dataset) > 0

    sample = dataset[0]

    required_keys = [

        "modalities",

        "continuous_labels",

        "discrete_labels",

        "metadata",
    ]

    for key in required_keys:

        if key not in sample:

            raise KeyError(
                f"Missing key: {key}"
            )

    print(
        "\n[PASS] Required keys present"
    )

    inspect_sample(
        sample
    )

    print(
        f"\n[PASS] "
        f"{split_name} split verified"
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print_header(
        "ULM-TSST DISK DATASET SMOKE TEST"
    )

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(

            f"Dataset not found:\n"
            f"{DATASET_ROOT}"
        )

    verify_split(
        "train"
    )

    verify_split(
        "val"
    )

    verify_split(
        "test"
    )

    print_header(
        "SMOKE TEST PASSED"
    )

    print(
        "\nULMTSSTDiskDataset is "
        "ready for training integration."
    )


if __name__ == "__main__":
    main()