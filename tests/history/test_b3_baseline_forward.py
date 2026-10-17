"""
tests.history.test_b3_history_forward

Verify that B3 correctly executes
the shared history framework.

Expected:

history_state:
    (B,64)

encoder_outputs:
    (B,history_length,256)

decoder_outputs:
    (B,forecast_horizon,256)
"""

from __future__ import annotations

import pathlib
import sys

import torch

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

from utils.config import load_config

from training.base_training_script import (
    to_namespace,
    build_dataloaders,
)

from models.wrappers.baseline_b3 import (
    BaselineB3,
)


def main():

    print("\n")
    print("=" * 70)
    print("B3 HISTORY FORWARD TEST")
    print("=" * 70)

    cfg = to_namespace(
        load_config(
            "configs/baseline_b3.yaml"
        )
    )

    train_loader, _ = (
        build_dataloaders(
            cfg,
            torch.device("cpu"),
        )
    )

    batch = next(
        iter(train_loader)
    )

    model = BaselineB3(
        cfg
    )

    model.eval()

    with torch.no_grad():

        outputs = model(
            batch
        )

    print("\n")
    print("-" * 70)

    print(
        "History Enabled:",
        outputs[
            "history_enabled"
        ]
    )

    if (
        outputs[
            "history_state"
        ]
        is not None
    ):
        print(
            "History State:",
            outputs[
                "history_state"
            ].shape
        )
    else:
        print(
            "History State: None"
        )

    print(
        "Fused:",
        outputs[
            "fused"
        ].shape
    )

    print(
        "Encoder Outputs:",
        outputs[
            "encoder_outputs"
        ].shape
    )

    print(
        "Decoder Outputs:",
        outputs[
            "decoder_outputs"
        ].shape
    )

    print("\n")
    print("=" * 70)
    print("B3 HISTORY FORWARD PASS OK")
    print("=" * 70)


if __name__ == "__main__":

    main()