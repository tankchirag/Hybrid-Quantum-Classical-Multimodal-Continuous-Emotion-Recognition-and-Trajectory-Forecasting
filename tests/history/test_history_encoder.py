"""
HistoryEncoder Smoke Test
"""

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


def main():
    history_dim = 64
    num_layers = 2
    dropout = 0.1
    history_dropout = 0.20

    model = HistoryEncoder(
        history_dim=history_dim,
        num_layers=num_layers,
        dropout=dropout,
        history_dropout=history_dropout,
    )

    history_representation = torch.randn(8, 50, 64)

    output = model(history_representation)

    print("\nOutput Shape:", output.shape)
    assert output.shape == (8, 64)
    print("PASS")


if __name__ == "__main__":
    main()