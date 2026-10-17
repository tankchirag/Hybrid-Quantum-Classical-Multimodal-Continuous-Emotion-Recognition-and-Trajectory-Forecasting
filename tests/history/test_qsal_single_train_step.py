"""
tests.history.test_qsal_single_train_step

Milestone 5C

Single Training Step Validation

Purpose
-------
Validate:

    Forward
        ↓

    Loss
        ↓

    Backward
        ↓

    Optimizer Step

using the real QSAL training pipeline.
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

from utils.config import (
    load_config,
)

from models.wrappers.qsal import (
    QSAL,
)

from training.losses import (
    UnifiedMultiTaskLoss,
)

from training.optimizer import (
    build_optimizer,
)

from training.base_training_script import (
    to_namespace,
    build_dataloaders,
)

# ==========================================================
# Test
# ==========================================================

def test_qsal_single_train_step():

    print("\n")
    print("=" * 70)
    print("MILESTONE 5C")
    print("SINGLE TRAIN STEP VALIDATION")
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

    print(
        f"\nBatch Size: {batch['batch_size']}"
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = QSAL(
        config
    )

    model.train()

    optimizer = (
        build_optimizer(
            model,
            config,
        )
    )

    loss_fn = (
        UnifiedMultiTaskLoss(
            alpha=0.2,
            recognition_weight=1.0,
            forecast_weight=1.0,
        )
    )

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------

    print(
        "\n[1] Forward..."
    )

    outputs = model(
        batch
    )

    # ------------------------------------------------------
    # Loss
    # ------------------------------------------------------

    print(
        "[2] Loss..."
    )

    losses = loss_fn(

        outputs=outputs,

        recognition_targets=
        batch[
            "recognition_targets"
        ],

        forecast_targets=
        batch[
            "forecast_targets"
        ],
    )

    total_loss = losses[
        "total_loss"
    ]

    print(
        f"Loss = "
        f"{float(total_loss.item()):.6f}"
    )

    # ------------------------------------------------------
    # Backward
    # ------------------------------------------------------

    print(
        "[3] Backward..."
    )

    optimizer.zero_grad()

    total_loss.backward()

    # ------------------------------------------------------
    # Gradient Check
    # ------------------------------------------------------

    num_gradients = 0

    for p in model.parameters():

        if p.grad is not None:

            num_gradients += 1

    print(
        f"Parameters With Gradients = "
        f"{num_gradients}"
    )

    assert num_gradients > 0

    # ------------------------------------------------------
    # Optimizer Step
    # ------------------------------------------------------

    print(
        "[4] Optimizer Step..."
    )

    optimizer.step()

    # ------------------------------------------------------
    # Success
    # ------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MILESTONE 5C PASSED")
    print("=" * 70)
    print()

    return True


if __name__ == "__main__":

    test_qsal_single_train_step()