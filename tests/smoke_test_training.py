"""
tests.smoke_test_training
=========================

Training Smoke Test

Purpose
-------

Verifies that all models can:

    ✓ Load configuration

    ✓ Build dataset

    ✓ Build dataloader

    ✓ Instantiate model

    ✓ Run forward pass

    ✓ Compute loss

    ✓ Run backward pass

    ✓ Perform optimizer step

This script is intentionally minimal.

It is NOT a training benchmark.

It only validates that the complete training
pipeline is functional.

Models
------

    • Baseline B1

    • Baseline B2

    • Baseline B3

    • QSAL
"""

from __future__ import annotations

import os
import sys

import torch


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if ROOT not in sys.path:
    sys.path.append(ROOT)


from utils.config import load_config

from models.wrappers.baseline_b1 import (
    BaselineB1,
)

from models.wrappers.baseline_b2 import (
    BaselineB2,
)

from models.wrappers.baseline_b3 import (
    BaselineB3,
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
    build_dataloaders,
    get_device,
)


# ==========================================================
# Helpers
# ==========================================================

def get_single_batch(
    dataloader,
):

    return next(
        iter(dataloader)
    )


def move_to_device(
    obj,
    device,
):

    if torch.is_tensor(obj):

        return obj.to(device)

    if isinstance(
        obj,
        dict,
    ):

        return {

            k: move_to_device(
                v,
                device,
            )

            for k, v in obj.items()
        }

    if isinstance(
        obj,
        list,
    ):

        return [

            move_to_device(
                item,
                device,
            )

            for item in obj
        ]

    return obj


# ==========================================================
# Single Model Smoke Test
# ==========================================================

def smoke_test_model(
    name,
    model_class,
    config_path,
):

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"SMOKE TEST :: {name}"
    )

    print(
        f"{'=' * 60}\n"
    )

    config = load_config(
        config_path
    )

    device = get_device(
        config
    )

    print(
        f"[INFO] Device: {device}"
    )

    (
        train_loader,
        _
    ) = build_dataloaders(
        config,
        device,
    )

    batch = get_single_batch(
        train_loader
    )

    batch = move_to_device(
        batch,
        device,
    )

    model = model_class(
        config=config
    ).to(device)

    optimizer = build_optimizer(
        model=model,
        config=config,
    )

    loss_fn = (
        UnifiedMultiTaskLoss()
    )

    # --------------------------------------
    # Forward
    # --------------------------------------

    outputs = model(
        batch
    )

    print(
        "[PASS] Forward"
    )

    # --------------------------------------
    # Loss
    # --------------------------------------

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
        "[PASS] Loss"
    )

    # --------------------------------------
    # Backward
    # --------------------------------------

    optimizer.zero_grad()

    total_loss.backward()

    print(
        "[PASS] Backward"
    )

    # --------------------------------------
    # Optimizer
    # --------------------------------------

    optimizer.step()

    print(
        "[PASS] Optimizer Step"
    )

    print(
        f"[PASS] Loss = "
        f"{total_loss.item():.6f}"
    )

    return True


# ==========================================================
# Main
# ==========================================================

def main():

    tests = [

        (
            "Baseline B1",
            BaselineB1,
            "configs/baseline_b1.yaml",
        ),

        (
            "Baseline B2",
            BaselineB2,
            "configs/baseline_b2.yaml",
        ),

        (
            "Baseline B3",
            BaselineB3,
            "configs/baseline_b3.yaml",
        ),

        (
            "QSAL",
            QSAL,
            "configs/qsal.yaml",
        ),
    ]

    passed = 0

    for (
        name,
        model_class,
        config_path,
    ) in tests:

        try:

            smoke_test_model(

                name=name,

                model_class=
                model_class,

                config_path=
                config_path,
            )

            passed += 1

        except Exception as e:

            print(
                f"\n[FAILED] "
                f"{name}"
            )

            print(
                f"{type(e).__name__}: "
                f"{e}"
            )

    print(
        f"\n\nRESULT: "
        f"{passed}/{len(tests)} "
        f"MODELS PASSED"
    )


if __name__ == "__main__":
    main()