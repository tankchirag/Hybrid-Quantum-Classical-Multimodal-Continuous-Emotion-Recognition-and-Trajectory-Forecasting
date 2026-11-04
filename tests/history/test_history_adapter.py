"""
HistoryAdapter Smoke Test
"""

import torch
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

from models.history.adapter import (
    HistoryAdapter,
)


def test_history_adapter():

    adapter = (
        HistoryAdapter()
    )

    continuous = {
        "valence": torch.randn(8, 50),
        "arousal": torch.randn(8, 50),
    }

    discrete = {
        "emotion_class": torch.randint(
            0,
            8,
            (8, 50),
        ),
    }

    output = adapter(
        continuous_labels=continuous,
        discrete_labels=discrete,
    )

    print(
        "continuous:",
        output["continuous"].shape
    )

    print(
        "discrete:",
        output["discrete"].shape
    )


if __name__ == "__main__":
    test_history_adapter()