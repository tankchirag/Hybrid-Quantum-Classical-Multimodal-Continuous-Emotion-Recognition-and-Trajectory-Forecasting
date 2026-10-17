"""
tests.history.test_qsal_parameter_comparison

Milestone 6B

Parameter and Complexity Comparison

Compare:

    QSAL

vs

    QSAL + History
"""

from __future__ import annotations

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

# ==========================================================
# Imports
# ==========================================================

from utils.config import (
    load_config,
)

from training.base_training_script import (
    to_namespace,
)

from models.wrappers.qsal import (
    QSAL,
)

# ==========================================================
# Helpers
# ==========================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


def count_trainable_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def count_trainable_tensors(model):

    return sum(
        1
        for p in model.parameters()
        if p.requires_grad
    )


# ==========================================================
# Build Models
# ==========================================================

def build_model(
    history_enabled: bool,
):

    config_dict = load_config(
        "configs/qsal.yaml"
    )

    config = to_namespace(
        config_dict
    )

    config.history_encoder.enabled = (
        history_enabled
    )

    return QSAL(
        config
    )


# ==========================================================
# Main
# ==========================================================

def test_qsal_parameter_comparison():

    print("\n")
    print("=" * 70)
    print("MILESTONE 6B")
    print("PARAMETER COMPARISON")
    print("=" * 70)

    # ------------------------------------------------------
    # History OFF
    # ------------------------------------------------------

    qsal_base = build_model(
        history_enabled=False
    )

    base_params = (
        count_parameters(
            qsal_base
        )
    )

    base_trainable = (
        count_trainable_parameters(
            qsal_base
        )
    )

    base_tensors = (
        count_trainable_tensors(
            qsal_base
        )
    )

    # ------------------------------------------------------
    # History ON
    # ------------------------------------------------------

    qsal_history = build_model(
        history_enabled=True
    )

    history_params = (
        count_parameters(
            qsal_history
        )
    )

    history_trainable = (
        count_trainable_parameters(
            qsal_history
        )
    )

    history_tensors = (
        count_trainable_tensors(
            qsal_history
        )
    )

    # ------------------------------------------------------
    # Delta
    # ------------------------------------------------------

    delta_params = (
        history_params
        - base_params
    )

    delta_trainable = (
        history_trainable
        - base_trainable
    )

    overhead_pct = (
        100.0
        * delta_trainable
        / base_trainable
    )

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    print("\n")
    print("-" * 70)

    print(
        f"{'Metric':<30}"
        f"{'History OFF':>15}"
        f"{'History ON':>15}"
    )

    print("-" * 70)

    print(
        f"{'Total Parameters':<30}"
        f"{base_params:>15,}"
        f"{history_params:>15,}"
    )

    print(
        f"{'Trainable Parameters':<30}"
        f"{base_trainable:>15,}"
        f"{history_trainable:>15,}"
    )

    print(
        f"{'Trainable Tensors':<30}"
        f"{base_tensors:>15}"
        f"{history_tensors:>15}"
    )

    print("-" * 70)

    print(
        f"{'Additional Params':<30}"
        f"{delta_params:>15,}"
    )

    print(
        f"{'Additional Trainable':<30}"
        f"{delta_trainable:>15,}"
    )

    print(
        f"{'Overhead %':<30}"
        f"{overhead_pct:>14.2f}%"
    )

    print("-" * 70)

    assert (
        history_params
        >= base_params
    )

    print("\n")
    print("=" * 70)
    print("MILESTONE 6B PASSED")
    print("=" * 70)

    return True


if __name__ == "__main__":

    test_qsal_parameter_comparison()