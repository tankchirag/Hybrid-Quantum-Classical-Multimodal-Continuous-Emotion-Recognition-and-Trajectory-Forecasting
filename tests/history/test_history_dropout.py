"""
tests.history.test_history_dropout

Fast verification that History Dropout
actually activates during training.
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

from models.history.encoder import (
    HistoryEncoder,
)


def test_history_dropout():

    print("\n")
    print("=" * 70)
    print("HISTORY DROPOUT TEST")
    print("=" * 70)

    torch.manual_seed(42)

    encoder = HistoryEncoder(
        history_dim=64,
        num_layers=2,
        history_dropout=0.20,
    )

    encoder.train()

    # ----------------------------------
    # Fake history representation
    # ----------------------------------

    history = torch.randn(
        8,      # batch
        50,     # history length
        64,     # history dim
    )

    kept = 0
    dropped = 0

    # ----------------------------------
    # only 20 iterations needed
    # ----------------------------------

    for _ in range(20):

        history_state = encoder(
            history
        )

        # sample-wise norms

        norms = (
            history_state
            .abs()
            .sum(dim=1)
        )

        dropped += (
            norms == 0
        ).sum().item()

        kept += (
            norms > 0
        ).sum().item()

    total = (
        kept
        + dropped
    )

    print(
        f"Kept    : {kept}"
    )

    print(
        f"Dropped : {dropped}"
    )

    print(
        f"Dropout : "
        f"{100*dropped/total:.2f}%"
    )

    print("\n")
    print("=" * 70)

    assert dropped > 0

    print(
        "HISTORY DROPOUT ACTIVE ✅"
    )

    print("=" * 70)


if __name__ == "__main__":

    test_history_dropout()