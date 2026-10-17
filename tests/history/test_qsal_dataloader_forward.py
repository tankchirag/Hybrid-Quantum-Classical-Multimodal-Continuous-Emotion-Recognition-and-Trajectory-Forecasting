"""
tests.history.test_qsal_dataloader_forward

Milestone 5B

REAL PIPELINE VALIDATION

ULMTSSTDiskDataset
        ↓
TimelineSynchronizer
        ↓
WindowSampler
        ↓
WindowedDataset
        ↓
collate
        ↓
QSAL

Purpose
-------
Verify that the history-aware QSAL model
can consume a real training batch generated
by the production data pipeline.
"""

from __future__ import annotations

import pathlib
import sys

import torch

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = (
    pathlib.Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

# ==========================================================
# Imports
# ==========================================================

from utils.config import load_config

from models.wrappers.qsal import (
    QSAL,
)

from training.base_training_script import (
    to_namespace,
    build_dataloaders,
)

# ==========================================================
# Test
# ==========================================================


def test_qsal_dataloader_forward():

    print("\n")
    print("=" * 70)
    print("MILESTONE 5B")
    print("REAL DATA PIPELINE VALIDATION")
    print("=" * 70)

    # ------------------------------------------------------
    # Config
    # ------------------------------------------------------

    config_dict = load_config(
        "configs/qsal.yaml"
    )

    config = to_namespace(
        config_dict
    )

    # ------------------------------------------------------
    # IMPORTANT
    # Enable history branch for this test
    # ------------------------------------------------------

    if hasattr(
        config,
        "history_encoder",
    ):
        config.history_encoder.enabled = True

    # ------------------------------------------------------
    # Data
    # ------------------------------------------------------

    train_loader, _ = (
        build_dataloaders(
            config,
            torch.device("cpu"),
        )
    )

    batch = next(
        iter(train_loader)
    )

    print("\n[BATCH]")
    print("-" * 50)

    print(
        "batch_size:",
        batch["batch_size"]
    )

    print(
        "modalities:",
        batch[
            "active_modalities"
        ]
    )

    print(
        "continuous labels:",
        batch[
            "active_continuous_labels"
        ]
    )

    print(
        "discrete labels:",
        batch[
            "active_discrete_labels"
        ]
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = QSAL(
        config
    )

    model.eval()

    with torch.no_grad():

        outputs = model(
            batch
        )

    print("\n[MODEL]")
    print("-" * 50)

    print(
        "history_enabled:",
        outputs[
            "history_enabled"
        ]
    )

    print(
        "history_state shape:",
        outputs[
            "history_state"
        ].shape
        if outputs[
            "history_state"
        ]
        is not None
        else None
    )

    print(
        "encoder shape:",
        outputs[
            "encoder_outputs"
        ].shape
    )

    print(
        "decoder shape:",
        outputs[
            "decoder_outputs"
        ].shape
    )

    # ------------------------------------------------------
    # Assertions
    # ------------------------------------------------------

    assert (
        "recognition"
        in outputs
    )

    assert (
        "forecasting"
        in outputs
    )

    assert (
        "encoder_outputs"
        in outputs
    )

    assert (
        "decoder_outputs"
        in outputs
    )

    assert (
        outputs[
            "history_state"
        ]
        is not None
    )

    print("\n")
    print("=" * 70)
    print("MILESTONE 5B PASSED")
    print("=" * 70)
    print()


if __name__ == "__main__":

    test_qsal_dataloader_forward()