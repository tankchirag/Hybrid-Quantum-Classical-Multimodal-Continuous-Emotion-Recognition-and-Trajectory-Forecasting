"""
datasets.split_ulm_tsst
=======================

ULM-TSST Dataset Splitter

Purpose
-------
Create reproducible train/validation/test splits.

Features
--------
✓ Stratified split

✓ Reproducible

✓ Balanced class distributions

✓ Split manifest generation

✓ Thesis-ready

Output
------

train/
val/
test/

split_manifest.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from sklearn.model_selection import (
    train_test_split,
)


# ==========================================================
# Helpers
# ==========================================================

def save_json(
    path,
    data,
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

def extract_primary_class(
    npz_path: Path,
):

    sample = np.load(
        npz_path,
        allow_pickle=True,
    )

    emotion = sample[
        "disc_emotion_class"
    ]

    values, counts = np.unique(
        emotion,
        return_counts=True,
    )

    return int(
        values[
            np.argmax(counts)
        ]
    )


# ==========================================================
# Create Split Directories
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
# Main
# ==========================================================

def split_dataset(
    dataset_root: Path,
):

    raw_dir = (
        dataset_root /
        "raw"
    )

    files = sorted(
        raw_dir.glob(
            "*.npz"
        )
    )

    if len(files) == 0:

        raise RuntimeError(
            "No raw files found."
        )

    print(
        "\nCollecting labels..."
    )

    labels = []

    for file in files:

        labels.append(
            extract_primary_class(
                file
            )
        )

    labels = np.asarray(
        labels
    )

    # ------------------------------------------------------
    # Train
    # ------------------------------------------------------

    train_files, temp_files, train_y, temp_y = (
        train_test_split(

            files,

            labels,

            test_size=0.20,

            stratify=labels,

            random_state=42,
        )
    )

    # ------------------------------------------------------
    # Val / Test
    # ------------------------------------------------------

    val_files, test_files, _, _ = (

        train_test_split(

            temp_files,

            temp_y,

            test_size=0.50,

            stratify=temp_y,

            random_state=42,
        )
    )

    train_dir, val_dir, test_dir = (
        create_split_dirs(
            dataset_root
        )
    )

    print(
        "\nCopying files..."
    )

    for file in train_files:

        shutil.copy2(
            file,
            train_dir /
            file.name,
        )

    for file in val_files:

        shutil.copy2(
            file,
            val_dir /
            file.name,
        )

    for file in test_files:

        shutil.copy2(
            file,
            test_dir /
            file.name,
        )

    # ------------------------------------------------------
    # Manifest
    # ------------------------------------------------------

    manifest = {

        "split_strategy":
            "stratified_emotion_class",

        "random_seed":
            42,

        "total_files":
            len(files),

        "train_files":
            len(train_files),

        "val_files":
            len(val_files),

        "test_files":
            len(test_files),

        "train_ratio":
            0.80,

        "val_ratio":
            0.10,

        "test_ratio":
            0.10,
    }

    save_json(

        dataset_root /
        "split_manifest.json",

        manifest,
    )

    print("\n")
    print("=" * 60)

    print("DATASET SPLIT COMPLETE")

    print("=" * 60)

    print(
        f"\nTrain : {len(train_files)}"
    )

    print(
        f"Val   : {len(val_files)}"
    )

    print(
        f"Test  : {len(test_files)}"
    )


# ==========================================================
# CLI
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    args = parser.parse_args()

    split_dataset(
        Path(
            args.dataset
        )
    )


if __name__ == "__main__":
    main()
