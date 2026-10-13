"""
tests.smoke_test_baseline_b1
============================

End-to-End Baseline B1 Smoke Test

Purpose
-------

Validate the complete Baseline B1 pipeline:

    Dataset
        ↓
    TimelineSynchronizer
        ↓
    WindowSampler
        ↓
    Collate
        ↓
    BaselineB1
        ↓
    Recognition Outputs
        ↓
    Forecast Outputs

Checks
------

✓ Dataset Loading

✓ Synchronization

✓ Window Sampling

✓ Batch Creation

✓ BaselineB1 Initialization

✓ Forward Pass

✓ Recognition Outputs

✓ Forecasting Outputs

✓ Shape Consistency

✓ End-to-End Integration
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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

from models.wrappers.baseline_b1 import (
    BaselineB1,
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


# ==========================================================
# Helper
# ==========================================================

def header(title):

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


# ==========================================================
# Model Config
# ==========================================================

def build_test_config():

    return SimpleNamespace(

        modalities={

            "ecg": True,
            "eda": True,
            "eeg": True,
            "emg": True,
            "resp": True,
            "bpm": True,

            "eye": True,

            "audio": True,
            "video": True,
            "text": True,
        },

        continuous_targets=[

            "valence",
            "arousal",
            "dominance",
            "stress",
            "workload",
            "fatigue",
        ],

        discrete_targets={

            "emotion_class": 8,

            "stress_class": 3,

            "workload_class": 3,

            "alertness_class": 3,

            "drowsiness_class": 3,
        },

        embedding_dim=128,

        fusion_dim=64,

        transformer_dim=256,

        num_heads=4,

        num_layers=4,

        dropout=0.1,

        forecast_horizon=25,

        history_dim=0,
    )


# ==========================================================
# Main
# ==========================================================

def main():

    header(
        "BASELINE B1 SMOKE TEST"
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

            synchronized_clip=
            sample,

            center_idx=
            center_idx,
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
    # Model
    # ------------------------------------------------------

    model = BaselineB1(
        build_test_config()
    )

    model.eval()

    print(
        "[PASS] Model Initialized"
    )

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            batch
        )

    print(
        "[PASS] Forward Pass"
    )

    # ======================================================
    # Recognition
    # ======================================================

    recognition = outputs[
        "recognition"
    ]

    header(
        "RECOGNITION OUTPUTS"
    )

    for (
        name,
        tensor,
    ) in recognition[
        "continuous"
    ].items():

        print(
            f"{name:<15}"
            f"{tuple(tensor.shape)}"
        )

        assert tensor.shape == (
            BATCH_SIZE,
            HISTORY_LENGTH,
        )

    # ======================================================
    # Forecasting
    # ======================================================

    forecasting = outputs[
        "forecasting"
    ]

    header(
        "FORECAST OUTPUTS"
    )

    for (
        name,
        tensor,
    ) in forecasting[
        "continuous"
    ].items():

        print(
            f"{name:<15}"
            f"{tuple(tensor.shape)}"
        )

        assert tensor.shape == (
            BATCH_SIZE,
            FORECAST_HORIZON,
        )

    # ======================================================
    # Encoder / Decoder
    # ======================================================

    header(
        "LATENT REPRESENTATIONS"
    )

    print(
        "encoder_outputs:",
        tuple(
            outputs[
                "encoder_outputs"
            ].shape
        )
    )

    print(
        "decoder_outputs:",
        tuple(
            outputs[
                "decoder_outputs"
            ].shape
        )
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
        "✅ BaselineB1"
    )

    print(
        "✅ Recognition"
    )

    print(
        "✅ Forecasting"
    )

    print(
        "\n✅ End-to-End Integration Ready"
    )


if __name__ == "__main__":
    main()