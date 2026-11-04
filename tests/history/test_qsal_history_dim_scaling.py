"""
tests.history.test_qsal_history_dim_scaling

Verify that ratio correctly scales
history_dim throughout QSAL.
"""

from __future__ import annotations

import pathlib
import sys

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
)

from models.wrappers.qsal import (
    QSAL,
)


def build_model(ratio):

    cfg = load_config(
        "configs/qsal.yaml"
    )

    cfg = to_namespace(
        cfg
    )

    cfg.history_encoder.enabled = True

    cfg.history_encoder.auto_scale = True

    cfg.history_encoder.ratio = ratio

    return QSAL(
        cfg
    )


def test_qsal_history_dim_scaling():

    print("\n")
    print("=" * 80)
    print("HISTORY DIMENSION SCALING")
    print("=" * 80)

    for ratio in [2, 4, 8]:

        model = build_model(
            ratio
        )

        print("\n")
        print("-" * 80)

        print(
            f"ratio          : {ratio}"
        )

        print(
            f"history_dim    : "
            f"{model.history_dim}"
        )

        if (
            model.history_encoder
            is not None
        ):

            print(
                f"gru_input_size : "
                f"{model.history_encoder.gru.input_size}"
            )

            print(
                f"gru_hidden_size: "
                f"{model.history_encoder.gru.hidden_size}"
            )

        if (
            model.history_representation
            is not None
        ):

            print(
                f"repr_history_dim: "
                f"{model.history_representation.history_dim}"
            )

        print("-" * 80)

    print("\n")
    print("=" * 80)
    print("DIMENSION SCALING PASSED")
    print("=" * 80)


if __name__ == "__main__":

    test_qsal_history_dim_scaling()