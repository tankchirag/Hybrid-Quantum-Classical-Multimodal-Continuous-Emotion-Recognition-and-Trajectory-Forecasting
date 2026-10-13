"""
training.metrics
================

Unified Evaluation Metrics Framework

Supports

    • Recognition

    • Forecasting

Across

    • Continuous Targets

    • Discrete Targets

Design Goals
------------

✓ Dataset Agnostic

✓ Modality Agnostic

✓ Label Agnostic

✓ Continuous Labels

✓ Discrete Labels

✓ Recognition Compatible

✓ Forecasting Compatible

✓ Continuous Forecasting

✓ Discrete Forecasting

✓ Multi-Task Compatible

✓ Future Proof
"""

from __future__ import annotations

from typing import Dict

import torch


# ============================================================
# Utilities
# ============================================================

def _flatten(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Flatten arbitrary tensor shapes.

    Examples

        (B,)
        (B,T)
        (B,S)
        (B,T,C)
        (B,S,C)

    →

        (N,)
    """

    return tensor.reshape(-1)


# ============================================================
# Continuous Metrics
# ============================================================

def concordance_cc(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Concordance Correlation Coefficient.
    """

    pred = _flatten(pred).float()
    target = _flatten(target).float()

    mean_pred = pred.mean()
    mean_target = target.mean()

    var_pred = pred.var(
        unbiased=False
    )

    var_target = target.var(
        unbiased=False
    )

    covariance = (
        (pred - mean_pred)
        *
        (target - mean_target)
    ).mean()

    ccc = (

        2.0
        *
        covariance

    ) / (

        var_pred
        +
        var_target
        +
        (mean_pred - mean_target) ** 2
        +
        eps

    )

    return float(
        ccc.item()
    )


def pearson_cc(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Pearson Correlation Coefficient.
    """

    pred = _flatten(pred).float()
    target = _flatten(target).float()

    pred_centered = (
        pred
        -
        pred.mean()
    )

    target_centered = (
        target
        -
        target.mean()
    )

    numerator = (
        pred_centered
        *
        target_centered
    ).sum()

    denominator = (

        torch.sqrt(
            (pred_centered ** 2).sum()
        )

        *

        torch.sqrt(
            (target_centered ** 2).sum()
        )

        +

        eps
    )

    return float(
        (
            numerator
            /
            denominator
        ).item()
    )


def mse(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:

    return float(

        torch.mean(
            (pred - target) ** 2
        ).item()
    )


def rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:

    return float(

        torch.sqrt(
            torch.mean(
                (pred - target) ** 2
            )
        ).item()
    )


def mae(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:

    return float(

        torch.mean(
            torch.abs(
                pred - target
            )
        ).item()
    )


# ============================================================
# Discrete Metrics
# ============================================================

def accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Classification Accuracy.

    Supports

        (B,C)

        (B,T,C)

        (B,S,C)
    """

    pred_class = logits.argmax(
        dim=-1
    )

    correct = (
        pred_class
        ==
        target
    ).float()

    return float(
        correct.mean().item()
    )


# ============================================================
# Continuous Evaluation
# ============================================================

def evaluate_continuous(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:

    return {

        "ccc":
            concordance_cc(
                pred,
                target,
            ),

        "pcc":
            pearson_cc(
                pred,
                target,
            ),

        "mse":
            mse(
                pred,
                target,
            ),

        "rmse":
            rmse(
                pred,
                target,
            ),

        "mae":
            mae(
                pred,
                target,
            ),
    }


# ============================================================
# Discrete Evaluation
# ============================================================

def evaluate_discrete(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:

    return {

        "accuracy":
            accuracy(
                logits,
                target,
            )
    }


# ============================================================
# Unified Model Evaluation
# ============================================================

def evaluate_outputs(
    outputs: Dict,
    recognition_targets: Dict,
    forecast_targets: Dict,
) -> Dict:
    """
    Unified evaluation for all tasks.

    Parameters
    ----------
    outputs

        Model outputs.

    recognition_targets

        Recognition ground truth.

    forecast_targets

        Forecast ground truth.
    """

    results = {

        "recognition": {

            "continuous": {},

            "discrete": {},
        },

        "forecasting": {

            "continuous": {},

            "discrete": {},
        },
    }

    # ========================================================
    # Recognition Continuous
    # ========================================================

    for (
        target_name,
        pred,
    ) in outputs[
        "recognition"
    ][
        "continuous"
    ].items():

        if (
            target_name
            not in
            recognition_targets[
                "continuous"
            ]
        ):
            continue

        gt = (
            recognition_targets[
                "continuous"
            ][
                target_name
            ]
        )

        results[
            "recognition"
        ][
            "continuous"
        ][
            target_name
        ] = evaluate_continuous(
            pred,
            gt,
        )

    # ========================================================
    # Recognition Discrete
    # ========================================================

    for (
        target_name,
        logits,
    ) in outputs[
        "recognition"
    ][
        "discrete"
    ].items():

        if (
            target_name
            not in
            recognition_targets[
                "discrete"
            ]
        ):
            continue

        gt = (
            recognition_targets[
                "discrete"
            ][
                target_name
            ]
        )

        results[
            "recognition"
        ][
            "discrete"
        ][
            target_name
        ] = evaluate_discrete(
            logits,
            gt,
        )

    # ========================================================
    # Forecast Continuous
    # ========================================================

    for (
        target_name,
        pred,
    ) in outputs[
        "forecasting"
    ][
        "continuous"
    ].items():

        if (
            target_name
            not in
            forecast_targets[
                "continuous"
            ]
        ):
            continue

        gt = (
            forecast_targets[
                "continuous"
            ][
                target_name
            ]
        )

        results[
            "forecasting"
        ][
            "continuous"
        ][
            target_name
        ] = evaluate_continuous(
            pred,
            gt,
        )

    # ========================================================
    # Forecast Discrete
    # ========================================================

    for (
        target_name,
        logits,
    ) in outputs[
        "forecasting"
    ][
        "discrete"
    ].items():

        if (
            target_name
            not in
            forecast_targets[
                "discrete"
            ]
        ):
            continue

        gt = (
            forecast_targets[
                "discrete"
            ][
                target_name
            ]
        )

        results[
            "forecasting"
        ][
            "discrete"
        ][
            target_name
        ] = evaluate_discrete(
            logits,
            gt,
        )

    return results