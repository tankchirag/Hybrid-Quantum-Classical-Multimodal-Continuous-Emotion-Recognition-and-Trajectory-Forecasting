"""
tests.history.test_qsal_mini_training

Milestone 5D

Mini Training Run

Validates:

    Dataset
        ↓
    Synchronizer
        ↓
    WindowSampler
        ↓
    Collate
        ↓
    QSAL
        ↓
    Trainer
        ↓
    Train Step
        ↓
    Validation Step

using the production training stack.

Intentionally tiny:

    epochs = 1
    max_train_batches = 1
    max_val_batches = 1
"""

from __future__ import annotations

import pathlib
import sys

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

from training.base_training_script import (
    to_namespace,
    build_dataloaders,
)

from training.optimizer import (
    build_optimizer,
)

from training.scheduler import (
    build_scheduler,
)

from training.trainer import (
    Trainer,
)

from models.wrappers.qsal import (
    QSAL,
)

import torch


# ==========================================================
# Test
# ==========================================================

def test_qsal_mini_training():

    print("\n")
    print("=" * 70)
    print("MILESTONE 5D")
    print("MINI TRAINING RUN")
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
    # History ON
    # ------------------------------------------------------

    config.history_encoder.enabled = True

    # ------------------------------------------------------
    # Tiny Run
    # ------------------------------------------------------

    config.training.epochs = 1

    config.training.batch_size = 2

    config.training.max_train_batches = 1

    config.training.max_val_batches = 1

    config.training.patience = 1

    # ------------------------------------------------------
    # Device
    # ------------------------------------------------------

    device = torch.device(
        "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # ------------------------------------------------------
    # Data
    # ------------------------------------------------------

    print(
        "\nBuilding dataloaders..."
    )

    train_loader, val_loader = (
        build_dataloaders(
            config,
            device,
        )
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    print(
        "Building model..."
    )

    model = QSAL(
        config
    )

    # ------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------

    optimizer = (
        build_optimizer(
            model,
            config,
        )
    )

    scheduler = (
        build_scheduler(
            optimizer,
            config,
        )
    )

    # ------------------------------------------------------
    # Trainer
    # ------------------------------------------------------

    trainer = Trainer(

        model=model,

        optimizer=optimizer,

        scheduler=scheduler,

        device=device,

        patience=1,

        save_path=None,

        use_tqdm=True,

        log_batches=True,

        max_train_batches=1,

        max_val_batches=1,
    )

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------

    print(
        "\nStarting mini training..."
    )

    history = trainer.fit(

        train_loader=train_loader,

        val_loader=val_loader,

        epochs=1,
    )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    assert len(history) == 1

    epoch_stats = history[0]

    print("\n")
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        "Train Loss:",
        epoch_stats[
            "train"
        ][
            "loss"
        ]
    )

    print(
        "Validation Loss:",
        epoch_stats[
            "val"
        ][
            "loss"
        ]
    )

    print("\n")
    print("=" * 70)
    print("MILESTONE 5D PASSED")
    print("=" * 70)

    return True


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":

    test_qsal_mini_training()