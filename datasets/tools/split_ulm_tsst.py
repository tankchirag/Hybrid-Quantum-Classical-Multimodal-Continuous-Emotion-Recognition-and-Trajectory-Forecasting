"""
datasets.tools.split_ulm_tsst
=============================

ULM-TSST Dataset Splitter

Purpose
-------
Create reproducible train/validation/test splits from a
validated ULM-TSST dataset release.

Features
--------
✓ Stratified split using emotion class

✓ Reproducible

✓ Safe re-run support

✓ Train / Validation / Test

✓ Distribution reporting

✓ Split manifest generation

✓ Thesis-ready

Dataset Structure
-----------------

Before

    raw/
        clip_000000.npz
        ...

After

    train/
    val/
    test/

    split_manifest.json

    split_statistics.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


# ==========================================================
# Configuration
# ==========================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


# ==========================================================
# Utilities
# ==========================================================

def save_json(
    path: Path,
    data: dict,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as fp:

        json.dump(
            data,
            fp,
            indent=4,
        )


# ==========================================================
# Extract Stratification Label
# ==========================================================

def get_primary_emotion_class(
    npz_file: Path,
) -> int:
    """
    Use dominant emotion class in sequence.

    Returns
    -------
    int
    """

    sample = np.load(
        npz_file,
        allow_pickle=True,
    )

    labels = sample[
        "disc_emotion_class"
    ]

    values, counts = np.unique(
        labels,
        return_counts=True,
    )

    dominant_class = values[
        np.argmax(counts)
    ]

    return int(dominant_class)


# ==========================================================
# Class Statistics
# ==========================================================

def compute_class_distribution(
    files,
):

    distribution = {}

    for file in files:

        label = (
            get_primary_emotion_class(
                file
            )
        )

        distribution[label] = (

            distribution.get(
                label,
                0,
            )
            + 1
        )

    return distribution


# ==========================================================
# Create Directories
# ==========================================================

def create_split_dirs(
    dataset_root,
):

    train_dir = (
        dataset_root /
        "train"
    )

    val_dir = (
        dataset_root /
        "val"
    )

    test_dir = (
        dataset_root /
        "test"
    )

    train_dir.mkdir(
        exist_ok=True
    )

    val_dir.mkdir(
        exist_ok=True
    )

    test_dir.mkdir(
        exist_ok=True
    )

    return (
        train_dir,
        val_dir,
        test_dir,
    )


# ==========================================================
# Copy Files
# ==========================================================

def copy_files(
    files,
    destination,
):

    for file in files:

        shutil.copy2(
            file,
            destination /
            file.name,
        )


# ==========================================================
# Main Split Logic
# ==========================================================

def split_dataset(
    dataset_root: Path,
):

    raw_dir = (
        dataset_root /
        "raw"
    )

    if not raw_dir.exists():

        raise FileNotFoundError(

            f"Missing:\n{raw_dir}"
        )

    files = sorted(
        raw_dir.glob(
            "*.npz"
        )
    )

    if len(files) == 0:

        raise RuntimeError(
            "No NPZ files found."
        )

    print(
        "\nCollecting split labels..."
    )

    labels = [

        get_primary_emotion_class(
            file
        )

        for file

        in files
    ]

    labels = np.asarray(
        labels
    )

    # ------------------------------------------------------
    # Train
    # ------------------------------------------------------

    train_files, temp_files, train_labels, temp_labels = (

        train_test_split(

            files,

            labels,

            test_size=
            (1 - TRAIN_RATIO),

            stratify=
            labels,

            random_state=
            RANDOM_SEED,
        )
    )

    # ------------------------------------------------------
    # Validation / Test
    # ------------------------------------------------------

    val_files, test_files, _, _ = (

        train_test_split(

            temp_files,

            temp_labels,

            test_size=0.5,

            stratify=
            temp_labels,

            random_state=
            RANDOM_SEED,
        )
    )

    train_dir, val_dir, test_dir = (

        create_split_dirs(
            dataset_root
        )
    )

    print(
        "\nCopying train files..."
    )

    copy_files(
        train_files,
        train_dir,
    )

    print(
        "Copying val files..."
    )

    copy_files(
        val_files,
        val_dir,
    )

    print(
        "Copying test files..."
    )

    copy_files(
        test_files,
        test_dir,
    )

    # ======================================================
    # Statistics
    # ======================================================

    split_statistics = {

        "train": {
            "count":
                len(train_files),

            "emotion_distribution":
                compute_class_distribution(
                    train_files
                ),
        },

        "val": {
            "count":
                len(val_files),

            "emotion_distribution":
                compute_class_distribution(
                    val_files
                ),
        },

        "test": {
            "count":
                len(test_files),

            "emotion_distribution":
                compute_class_distribution(
                    test_files
                ),
        },
    }

    save_json(

        dataset_root /
        "split_statistics.json",

        split_statistics,
    )

    # ======================================================
    # Manifest
    # ======================================================

    manifest = {

        "dataset_split": True,

        "split_strategy":
            "stratified_primary_emotion",

        "random_seed":
            RANDOM_SEED,

        "train_ratio":
            TRAIN_RATIO,

        "val_ratio":
            VAL_RATIO,

        "test_ratio":
            TEST_RATIO,

        "total_files":
            len(files),

        "train_files":
            len(train_files),

        "val_files":
            len(val_files),

        "test_files":
            len(test_files),
    }

    save_json(

        dataset_root /
        "split_manifest.json",

        manifest,
    )

    # ======================================================
    # Summary
    # ======================================================

    print("\n")
    print("=" * 70)

    print(
        "ULM-TSST SPLIT COMPLETE"
    )

    print("=" * 70)

    print(
        f"\nTotal : {len(files)}"
    )

    print(
        f"Train : {len(train_files)}"
    )

    print(
        f"Val   : {len(val_files)}"
    )

    print(
        f"Test  : {len(test_files)}"
    )

    print(
        "\nStatistics:"
        "\n"
        f"{dataset_root/'split_statistics.json'}"
    )

    print(
        "\nManifest:"
        "\n"
        f"{dataset_root/'split_manifest.json'}"
    )


# ==========================================================
# CLI
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset root",
    )

    args = parser.parse_args()

    split_dataset(
        Path(
            args.dataset
        )
    )


if __name__ == "__main__":
    main()