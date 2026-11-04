"""
tests.history.test_qsal_history_ratio_comparison

Milestone 6C

History Ratio Comparison

Compare:

ratio=2
ratio=4
ratio=8

for parameter complexity.
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

from models.wrappers.qsal import QSAL


# ======================================================
# Utils
# ======================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


def build_model(
    history_enabled,
    ratio,
):

    cfg = load_config(
        "configs/qsal.yaml"
    )

    cfg = to_namespace(
        cfg
    )

    cfg.history_encoder.enabled = (
        history_enabled
    )

    cfg.history_encoder.ratio = (
        ratio
    )

    return QSAL(
        cfg
    )


# ======================================================
# Main
# ======================================================

def test_qsal_history_ratio_comparison():

    print("\n")
    print("=" * 80)
    print("MILESTONE 6C")
    print("HISTORY RATIO COMPARISON")
    print("=" * 80)

    # ---------------------------------------------
    # Baseline
    # ---------------------------------------------

    baseline = build_model(
        history_enabled=False,
        ratio=4,
    )

    baseline_params = (
        count_parameters(
            baseline
        )
    )

    print(
        f"\nBaseline Params: "
        f"{baseline_params:,}"
    )

    # ---------------------------------------------
    # Ratios
    # ---------------------------------------------

    ratios = [2, 4, 8]

    print("\n")
    print("-" * 80)

    print(
        f"{'Ratio':<10}"
        f"{'Params':>20}"
        f"{'Added':>20}"
        f"{'Overhead %':>20}"
    )

    print("-" * 80)

    for ratio in ratios:

        model = build_model(
            history_enabled=True,
            ratio=ratio,
        )

        params = (
            count_parameters(
                model
            )
        )

        added = (
            params
            - baseline_params
        )

        overhead = (
            100.0
            * added
            / baseline_params
        )

        print(
            f"{ratio:<10}"
            f"{params:>20,}"
            f"{added:>20,}"
            f"{overhead:>19.2f}%"
        )

    print("-" * 80)

    print("\n")
    print("=" * 80)
    print("MILESTONE 6C PASSED")
    print("=" * 80)


if __name__ == "__main__":

    test_qsal_history_ratio_comparison()