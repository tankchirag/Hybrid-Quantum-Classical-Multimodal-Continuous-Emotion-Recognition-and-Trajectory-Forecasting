"""
tests.inspect_generated_dataset
===============================

Generated Dataset Inspection Tool

Purpose
-------

Inspect a generated ULM-TSST dataset release.

Checks
------

✓ Dataset directory

✓ Manifest

✓ README

✓ Config snapshot

✓ NPZ contents

✓ Modalities

✓ Continuous labels

✓ Discrete labels

✓ Shapes

✓ Metadata

This is a human-inspection utility used
before automated validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ==========================================================
# CONFIG
# ==========================================================

DATASET_NAME = "ulm_tsst_v1"


# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / DATASET_NAME
)


# ==========================================================
# Utilities
# ==========================================================

def print_header(title):

    print("\n")
    print("=" * 60)
    print(title)
    print("=" * 60)


# ==========================================================
# Main
# ==========================================================

def main():

    print_header(
        "ULM-TSST DATASET INSPECTION"
    )

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{DATASET_ROOT}\n\n"
            f"Generate dataset first."
        )

    # ------------------------------------------------------
    # Manifest
    # ------------------------------------------------------

    manifest_path = (
        DATASET_ROOT
        / "dataset_manifest.json"
    )

    if manifest_path.exists():

        print_header(
            "DATASET MANIFEST"
        )

        with open(
            manifest_path,
            "r",
            encoding="utf-8",
        ) as fp:

            manifest = json.load(fp)

        for (
            key,
            value,
        ) in manifest.items():

            print(
                f"{key}: {value}"
            )

    # ------------------------------------------------------
    # Raw Files
    # ------------------------------------------------------

    raw_dir = (
        DATASET_ROOT
        / "raw"
    )

    files = sorted(
        raw_dir.glob("*.npz")
    )

    print_header(
        "DATASET FILES"
    )

    print(
        f"Number of files: "
        f"{len(files)}"
    )

    if len(files) == 0:

        raise RuntimeError(
            "No dataset files found."
        )

    sample_file = files[0]

    print(
        f"\nExample file:\n"
        f"{sample_file.name}"
    )

    # ------------------------------------------------------
    # Load One Sample
    # ------------------------------------------------------

    print_header(
        "SAMPLE CONTENTS"
    )

    sample = np.load(
        sample_file,
        allow_pickle=True,
    )

    keys = list(
        sample.keys()
    )

    for key in keys:

        print(key)

    print_header(
        "TENSOR SHAPES"
    )

    for key in keys:

        array = sample[key]

        print(

            f"{key:<40}"

            f"{array.shape}"
        )

    # ------------------------------------------------------
    # Modalities
    # ------------------------------------------------------

    print_header(
        "MODALITIES"
    )

    for key in keys:

        if key.startswith(
            "mod_"
        ):

            print(
                f"{key}"
            )

    # ------------------------------------------------------
    # Continuous Labels
    # ------------------------------------------------------

    print_header(
        "CONTINUOUS LABELS"
    )

    for key in keys:

        if key.startswith(
            "cont_"
        ):

            print(
                f"{key}"
            )

    # ------------------------------------------------------
    # Discrete Labels
    # ------------------------------------------------------

    print_header(
        "DISCRETE LABELS"
    )

    for key in keys:

        if key.startswith(
            "disc_"
        ):

            print(
                f"{key}"
            )

    print_header(
        "INSPECTION COMPLETE"
    )


if __name__ == "__main__":
    main()