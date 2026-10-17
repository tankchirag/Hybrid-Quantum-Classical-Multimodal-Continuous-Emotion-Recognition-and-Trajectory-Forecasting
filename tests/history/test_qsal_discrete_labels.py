"""
tests.history.test_qsal_discrete_labels

Diagnostic Test

Purpose
-------
Trace discrete labels through:

Dataset
    ↓

WindowSampler
    ↓

Collate
    ↓

QSAL
    ↓

Outputs

Goal
----
Determine whether discrete labels are lost:

    1. Dataset
    2. WindowSampler
    3. Collate
    4. QSAL Heads
    5. Output Assembly
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

from models.wrappers.qsal import (
    QSAL,
)

# ==========================================================
# Helpers
# ==========================================================

def print_keys(
    title,
    obj,
):

    print("\n" + title)

    if obj is None:

        print("None")
        return

    try:

        print(
            list(
                obj.keys()
            )
        )

    except Exception:

        print(
            type(obj)
        )


# ==========================================================
# Diagnostic
# ==========================================================

def test_qsal_discrete_labels():

    print("\n")
    print("=" * 80)
    print("DISCRETE LABEL DIAGNOSTIC")
    print("=" * 80)

    # ------------------------------------------------------
    # Config
    # ------------------------------------------------------

    cfg = load_config(
        "configs/qsal.yaml"
    )

    cfg = to_namespace(
        cfg
    )

    cfg.history_encoder.enabled = False

    # ------------------------------------------------------
    # Real DataLoader
    # ------------------------------------------------------

    train_loader, _ = (
        build_dataloaders(
            cfg,
            torch.device("cpu"),
        )
    )

    batch = next(
        iter(train_loader)
    )

    # ======================================================
    # Batch Inputs
    # ======================================================

    print("\n")
    print("=" * 80)
    print("BATCH CONTENTS")
    print("=" * 80)

    print_keys(
        "continuous_labels",
        batch[
            "continuous_labels"
        ],
    )

    print_keys(
        "discrete_labels",
        batch[
            "discrete_labels"
        ],
    )

    print_keys(
        "recognition_targets.continuous",
        batch[
            "recognition_targets"
        ][
            "continuous"
        ],
    )

    print_keys(
        "recognition_targets.discrete",
        batch[
            "recognition_targets"
        ][
            "discrete"
        ],
    )

    print_keys(
        "forecast_targets.continuous",
        batch[
            "forecast_targets"
        ][
            "continuous"
        ],
    )

    print_keys(
        "forecast_targets.discrete",
        batch[
            "forecast_targets"
        ][
            "discrete"
        ],
    )

    # ======================================================
    # Shapes
    # ======================================================

    print("\n")
    print("=" * 80)
    print("DISCRETE TARGET SHAPES")
    print("=" * 80)

    for name, tensor in (
        batch[
            "recognition_targets"
        ][
            "discrete"
        ]
        .items()
    ):

        print(
            f"{name:<25}"
            f"{tuple(tensor.shape)}"
        )

    for name, tensor in (
        batch[
            "forecast_targets"
        ][
            "discrete"
        ]
        .items()
    ):

        print(
            f"{name:<25}"
            f"{tuple(tensor.shape)}"
        )

    # ======================================================
    # Model
    # ======================================================

    model = QSAL(
        cfg
    )

    model.eval()

    with torch.no_grad():

        outputs = model(
            batch
        )

    # ======================================================
    # Model Outputs
    # ======================================================

    print("\n")
    print("=" * 80)
    print("MODEL OUTPUTS")
    print("=" * 80)

    print_keys(
        "recognition",
        outputs[
            "recognition"
        ],
    )

    print_keys(
        "forecasting",
        outputs[
            "forecasting"
        ],
    )

    print_keys(
        "recognition.continuous",
        outputs[
            "recognition"
        ][
            "continuous"
        ],
    )

    print_keys(
        "recognition.discrete",
        outputs[
            "recognition"
        ][
            "discrete"
        ],
    )

    print_keys(
        "forecasting.continuous",
        outputs[
            "forecasting"
        ][
            "continuous"
        ],
    )

    print_keys(
        "forecasting.discrete",
        outputs[
            "forecasting"
        ][
            "discrete"
        ],
    )

    # ======================================================
    # Compare
    # ======================================================

    batch_discrete = set(
        batch[
            "recognition_targets"
        ][
            "discrete"
        ].keys()
    )

    output_discrete = set(
        outputs[
            "recognition"
        ][
            "discrete"
        ].keys()
    )

    print("\n")
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print(
        "Batch Discrete Labels:"
    )

    print(
        sorted(
            batch_discrete
        )
    )

    print()

    print(
        "Model Discrete Labels:"
    )

    print(
        sorted(
            output_discrete
        )
    )

    print()

    print(
        "Missing Labels:"
    )

    print(
        sorted(
            batch_discrete
            - output_discrete
        )
    )

    print("\n")
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":

    test_qsal_discrete_labels()