"""
tests.smoke_test_dataset_generation
===================================

Dataset Generation Smoke Test

Purpose
-------

Verify that the ULM-TSST generation pipeline works
before generating a full dataset release.

Checks
------

✓ Config Loading

✓ Dataset Construction

✓ Sample Generation

✓ Continuous Labels

✓ Discrete Labels

✓ Modalities

✓ Metadata

✓ Dataset Saving

✓ One-Sample Generation

This is NOT a full dataset generation script.

It only validates that the generation pipeline is
functional.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
import torch


# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.append(
    str(PROJECT_ROOT)
)

# ==========================================================
# Imports
# ==========================================================

from datasets.ulm_tsst_synthetic import (
    SyntheticULMTSSTDataset
)


# ==========================================================
# Config Loader
# ==========================================================

def load_yaml(
    path: str,
):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as fp:

        return yaml.safe_load(fp)


# ==========================================================
# Main
# ==========================================================

def main():

    print("\n")
    print("=" * 60)
    print("ULM-TSST DATASET GENERATION SMOKE TEST")
    print("=" * 60)

    config_path = (
        PROJECT_ROOT
        / "configs"
        / "datasets"
        / "ulm_tsst_v1.yaml"
    )

    print(
        f"\n[INFO] Loading config:"
        f"\n{config_path}"
    )

    cfg = load_yaml(
        config_path
    )

    print("[PASS] Config loaded")

    # ------------------------------------------------------
    # Temporary test settings
    # ------------------------------------------------------

    test_samples = 2

    save_dir = (
        PROJECT_ROOT
        / "tmp_dataset_smoke_test"
    )

    save_dir.mkdir(
        exist_ok=True
    )

    # ------------------------------------------------------
    # Construct Dataset
    # ------------------------------------------------------

    print(
        "\n[INFO] Building dataset..."
    )

    dataset = (
        SyntheticULMTSSTDataset(

            num_samples=
            test_samples,

            duration_seconds=
            cfg["recording"][
                "duration_seconds"
            ],

            modalities=
            cfg["modalities"],

            continuous_labels=
            cfg[
                "continuous_labels"
            ],

            discrete_labels=
            cfg[
                "discrete_labels"
            ],

            sampling_rates=
            cfg[
                "sampling_rates"
            ],

            label_sampling_rate=
            cfg["labels"][
                "label_sampling_rate"
            ],

            num_subjects=
            cfg["samples"][
                "num_subjects"
            ],

            num_sessions=
            cfg["samples"][
                "num_sessions_per_subject"
            ],

            save_dir=
            str(save_dir),

            seed=
            cfg[
                "reproducibility"
            ]["seed"],

            device="cpu",
        )
    )

    print("[PASS] Dataset built")

    # ------------------------------------------------------
    # Generate 1 Sample
    # ------------------------------------------------------

    print(
        "\n[INFO] Generating sample[0]..."
    )

    sample = dataset[0]

    print(
        "[PASS] Sample generated"
    )

    # ------------------------------------------------------
    # Check Required Keys
    # ------------------------------------------------------

    required_keys = [

        "modalities",

        "continuous_labels",

        "discrete_labels",

        "masks",

        "metadata",
    ]

    for key in required_keys:

        if key not in sample:

            raise KeyError(
                f"Missing key: {key}"
            )

    print(
        "[PASS] Required keys exist"
    )

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    metadata = (
        sample["metadata"]
    )

    print(
        "\n[INFO] Metadata:"
    )

    for key, value in (
        metadata.items()
    ):

        print(
            f"  {key}: {value}"
        )

    # ------------------------------------------------------
    # Modalities
    # ------------------------------------------------------

    print(
        "\n[INFO] Modalities"
    )

    for (
        name,
        tensor,
    ) in sample[
        "modalities"
    ].items():

        if not isinstance(
            tensor,
            torch.Tensor,
        ):

            raise TypeError(
                f"{name} "
                f"is not tensor"
            )

        print(
            f"  {name:<10}"
            f"{tuple(tensor.shape)}"
        )

    # ------------------------------------------------------
    # Continuous Labels
    # ------------------------------------------------------

    print(
        "\n[INFO] Continuous Labels"
    )

    for (
        name,
        tensor,
    ) in sample[
        "continuous_labels"
    ].items():

        print(
            f"  {name:<15}"
            f"{tuple(tensor.shape)}"
        )

    # ------------------------------------------------------
    # Discrete Labels
    # ------------------------------------------------------

    print(
        "\n[INFO] Discrete Labels"
    )

    for (
        name,
        tensor,
    ) in sample[
        "discrete_labels"
    ].items():

        print(
            f"  {name:<20}"
            f"{tuple(tensor.shape)}"
        )

    # ------------------------------------------------------
    # Verify Saved File
    # ------------------------------------------------------

    generated_files = list(
        save_dir.glob(
            "clip_*.npz"
        )
    )

    if len(
        generated_files
    ) == 0:

        raise RuntimeError(
            "No dataset files saved."
        )

    print(
        f"\n[PASS] Saved Files: "
        f"{len(generated_files)}"
    )

    print(
        f"Example:\n"
        f"{generated_files[0]}"
    )

    # ------------------------------------------------------
    # Success
    # ------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("DATASET GENERATION SMOKE TEST PASSED")
    print("=" * 60)
    print("\n")


if __name__ == "__main__":
    main()