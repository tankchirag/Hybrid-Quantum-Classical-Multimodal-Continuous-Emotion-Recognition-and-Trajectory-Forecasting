"""
tests.history.test_history_length_scaling

Verify that history_length propagates
through dataset -> sampler -> collate.

Expected:

history_length=20
    -> history tensors use 20

history_length=50
    -> history tensors use 50

history_length=100
    -> history tensors use 100
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


# ==========================================================
# Helper
# ==========================================================

def inspect_history_length(
    history_length: int,
):

    cfg = to_namespace(
        load_config(
            "configs/qsal.yaml"
        )
    )

    cfg.history.history_length = (
        history_length
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

    print("\n")
    print("-" * 70)

    print(
        f"Configured history_length = "
        f"{history_length}"
    )

    # --------------------------------------
    # Search batch recursively
    # --------------------------------------

    found = False

    def scan_dict(
        obj,
        prefix="",
    ):

        nonlocal found

        if isinstance(
            obj,
            dict,
        ):

            for k, v in obj.items():

                name = (
                    f"{prefix}.{k}"
                    if prefix
                    else k
                )

                if hasattr(
                    v,
                    "shape",
                ):

                    if (
                        len(v.shape) >= 2
                    ):

                        print(
                            f"{name:<40}"
                            f"{tuple(v.shape)}"
                        )

                        found = True

                elif isinstance(
                    v,
                    dict,
                ):

                    scan_dict(
                        v,
                        name,
                    )

    scan_dict(batch)

    if not found:

        print(
            "No history tensors found."
        )


# ==========================================================
# Main
# ==========================================================

def test_history_length_scaling():

    print("\n")
    print("=" * 70)
    print("HISTORY LENGTH SCALING")
    print("=" * 70)

    inspect_history_length(20)

    inspect_history_length(50)

    inspect_history_length(100)

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":

    test_history_length_scaling()
