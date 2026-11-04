"""
HistoryInjection Smoke Test
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

from models.history.injection import (
    HistoryInjection,
)


def test_history_injection():

    model = (
        HistoryInjection(
            model_dim=256,
            history_dim=64,
        )
    )

    fused = torch.randn(
        8,
        50,
        256,
    )

    history = torch.randn(
        8,
        64,
    )

    output = model(
        fused,
        history,
    )

    print(
        "Output Shape:",
        output.shape,
    )

    assert (
        output.shape
        ==
        (
            8,
            50,
            256,
        )
    )

    print(
        "PASS"
    )


if __name__ == "__main__":
    test_history_injection()