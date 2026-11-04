"""
tests.history.test_qsal_history_disabled

Milestone 6A

History Disabled Validation

Purpose
-------
Verify that the original QSAL architecture
still functions when the history subsystem
is disabled.

Validation

Dataset
    ↓

WindowSampler
    ↓

Collate
    ↓

QSAL (History OFF)
    ↓

Forward
    ↓

Loss
    ↓

Backward
    ↓

Optimizer Step
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

from training.base_training_script import (
    to_namespace,
    build_dataloaders,
)

from training.optimizer import (
    build_optimizer,
)

from training.losses import (
    UnifiedMultiTaskLoss,
)

from models.wrappers.qsal import (
    QSAL,
)


# ==========================================================
# Test
# ==========================================================

def test_qsal_history_disabled():

    print("\n")
    print("=" * 70)
    print("MILESTONE 6A")
    print("HISTORY DISABLED VALIDATION")
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
    # Disable History
    # ------------------------------------------------------

    config.history_encoder.enabled = False

    print(
        "\nHistory Enabled:",
        config.history_encoder.enabled,
    )

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
        f"\nBatch Size: "
        f"{batch['batch_size']}"
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
    # History Validation
    # ------------------------------------------------------

    print(
        "[2] History Check..."
    )

    history_enabled = outputs.get(
        "history_enabled",
        None,
    )

    history_state = outputs.get(
        "history_state",
        None,
    )

    print(
        "history_enabled =",
        history_enabled,
    )

    print(
        "history_state =",
        history_state,
    )

    assert (
        history_enabled is False
        or history_enabled is None
    ), (
        "History unexpectedly enabled."
    )

    assert (
        history_state is None
    ), (
        "History should be disabled "
        "but history_state exists."
    )

    # ------------------------------------------------------
    # Loss
    # ------------------------------------------------------

    print(
        "[3] Loss..."
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

    total_loss = (
        losses[
            "total_loss"
        ]
    )

    print(
        "loss =",
        float(
            total_loss.item()
        ),
    )

    print(
        "recognition_loss =",
        float(
            losses[
                "recognition_loss"
            ].item()
        ),
    )

    print(
        "forecasting_loss =",
        float(
            losses[
                "forecasting_loss"
            ].item()
        ),
    )

    # ------------------------------------------------------
    # Backward
    # ------------------------------------------------------

    print(
        "[4] Backward..."
    )

    optimizer.zero_grad()

    total_loss.backward()

    num_grads = 0

    for p in model.parameters():

        if p.grad is not None:
            num_grads += 1

    print(
        "parameters_with_gradients =",
        num_grads,
    )

    assert (
        num_grads > 0
    )

    # ------------------------------------------------------
    # Optimizer Step
    # ------------------------------------------------------

    print(
        "[5] Optimizer Step..."
    )

    optimizer.step()

    # ------------------------------------------------------
    # Output Shapes
    # ------------------------------------------------------

    print(
        "\nencoder_outputs:",
        outputs[
            "encoder_outputs"
        ].shape,
    )

    print(
        "decoder_outputs:",
        outputs[
            "decoder_outputs"
        ].shape,
    )

    # ------------------------------------------------------
    # Recognition Outputs
    # ------------------------------------------------------

    if (
        "recognition"
        in outputs
    ):

        print(
            "\nrecognition continuous labels:"
        )

        print(
            list(
                outputs[
                    "recognition"
                ][
                    "continuous"
                ].keys()
            )
        )

        print(
            "\nrecognition discrete labels:"
        )

        print(
            list(
                outputs[
                    "recognition"
                ][
                    "discrete"
                ].keys()
            )
        )

    # ------------------------------------------------------
    # Forecast Outputs
    # ------------------------------------------------------

    if (
        "forecasting"
        in outputs
    ):

        print(
            "\nforecast continuous labels:"
        )

        print(
            list(
                outputs[
                    "forecasting"
                ][
                    "continuous"
                ].keys()
            )
        )

        print(
            "\nforecast discrete labels:"
        )

        print(
            list(
                outputs[
                    "forecasting"
                ][
                    "discrete"
                ].keys()
            )
        )

    # ------------------------------------------------------
    # Success
    # ------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MILESTONE 6A PASSED")
    print("=" * 70)

    return True


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":

    test_qsal_history_disabled()