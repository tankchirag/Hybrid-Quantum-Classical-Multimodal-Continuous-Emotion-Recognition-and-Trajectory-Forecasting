"""
training.losses
===============

Unified Multi-Task Loss Framework

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
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CCC LOSS
# ============================================================

def ccc_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Concordance Correlation Coefficient Loss.
    """

    if pred.shape != target.shape:
        raise ValueError(
            "CCC loss requires prediction and target shapes to match; "
            f"got pred={tuple(pred.shape)}, target={tuple(target.shape)}."
        )

    pred = pred.reshape(-1).float()
    target = target.reshape(-1).float()

    pred_mean = pred.mean()
    target_mean = target.mean()

    pred_var = pred.var(
        unbiased=False
    )

    target_var = target.var(
        unbiased=False
    )

    covariance = (
        (pred - pred_mean)
        *
        (target - target_mean)
    ).mean()

    ccc = (

        2.0
        *
        covariance

    ) / (

        pred_var
        +
        target_var
        +
        (pred_mean - target_mean) ** 2
        +
        eps

    )

    return 1.0 - ccc


# ============================================================
# CONTINUOUS LOSS
# ============================================================

class ContinuousLoss(
    nn.Module,
):
    """
    CCC + alpha * MSE
    """

    def __init__(
        self,
        alpha: float = 0.2,
    ):
        super().__init__()

        self.alpha = alpha

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        loss_ccc = ccc_loss(
            pred,
            target,
        )

        loss_mse = F.mse_loss(
            pred,
            target,
        )

        return (

            loss_ccc

            +

            self.alpha
            *
            loss_mse
        )


# ============================================================
# DISCRETE LOSS
# ============================================================

class DiscreteLoss(
    nn.Module,
):
    """
    Cross Entropy Loss.

    Handles

        Recognition

        Forecasting

    for

        (B,C)

        (B,T,C)

        (B,S,C)
    """

    def __init__(self):
        super().__init__()

        self.ce = (
            nn.CrossEntropyLoss()
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        num_classes = logits.size(-1)

        logits = logits.reshape(
            -1,
            num_classes,
        )

        target = target.reshape(
            -1
        )

        return self.ce(
            logits,
            target.long(),
        )


# ============================================================
# UNIFIED LOSS
# ============================================================

class UnifiedMultiTaskLoss(
    nn.Module,
):
    """
    Unified loss for

        Recognition Continuous

        Recognition Discrete

        Forecast Continuous

        Forecast Discrete
    """

    def __init__(
        self,
        alpha: float = 0.2,
        recognition_weight: float = 1.0,
        forecast_weight: float = 1.0,
    ):
        super().__init__()

        self.continuous_loss = (
            ContinuousLoss(
                alpha=alpha
            )
        )

        self.discrete_loss = (
            DiscreteLoss()
        )

        self.recognition_weight = (
            recognition_weight
        )

        self.forecast_weight = (
            forecast_weight
        )

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------

    def _infer_device(
        self,
        outputs: Dict,
    ):

        for task_name in [

            "recognition",

            "forecasting",

        ]:

            if task_name not in outputs:
                continue

            for target_type in [

                "continuous",

                "discrete",

            ]:

                if (
                    target_type
                    not in outputs[
                        task_name
                    ]
                ):
                    continue

                for tensor in outputs[
                    task_name
                ][
                    target_type
                ].values():

                    return tensor.device

        return torch.device(
            "cpu"
        )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        outputs: Dict,
        recognition_targets: Dict,
        forecast_targets: Dict,
    ) -> Dict[str, torch.Tensor]:

        device = self._infer_device(
            outputs
        )

        recognition_loss = torch.tensor(
            0.0,
            device=device,
        )

        forecasting_loss = torch.tensor(
            0.0,
            device=device,
        )

        # ====================================================
        # RECOGNITION CONTINUOUS
        # ====================================================

        for (
            name,
            pred,
        ) in outputs[
            "recognition"
        ][
            "continuous"
        ].items():

            if (
                name
                not in
                recognition_targets[
                    "continuous"
                ]
            ):
                continue

            recognition_loss += (

                self.continuous_loss(

                    pred,

                    recognition_targets[
                        "continuous"
                    ][
                        name
                    ],
                )
            )

        # ====================================================
        # RECOGNITION DISCRETE
        # ====================================================

        for (
            name,
            logits,
        ) in outputs[
            "recognition"
        ][
            "discrete"
        ].items():

            if (
                name
                not in
                recognition_targets[
                    "discrete"
                ]
            ):
                continue

            recognition_loss += (

                self.discrete_loss(

                    logits,

                    recognition_targets[
                        "discrete"
                    ][
                        name
                    ],
                )
            )

        # ====================================================
        # FORECAST CONTINUOUS
        # ====================================================

        for (
            name,
            pred,
        ) in outputs[
            "forecasting"
        ][
            "continuous"
        ].items():

            if (
                name
                not in
                forecast_targets[
                    "continuous"
                ]
            ):
                continue

            forecasting_loss += (

                self.continuous_loss(

                    pred,

                    forecast_targets[
                        "continuous"
                    ][
                        name
                    ],
                )
            )

        # ====================================================
        # FORECAST DISCRETE
        # ====================================================

        for (
            name,
            logits,
        ) in outputs[
            "forecasting"
        ][
            "discrete"
        ].items():

            if (
                name
                not in
                forecast_targets[
                    "discrete"
                ]
            ):
                continue

            forecasting_loss += (

                self.discrete_loss(

                    logits,

                    forecast_targets[
                        "discrete"
                    ][
                        name
                    ],
                )
            )

        # ====================================================
        # TOTAL
        # ====================================================

        total_loss = (

                    self.recognition_weight
                    *
                    recognition_loss

                    +

                    self.forecast_weight
                    *
                    forecasting_loss
                )

        # print("\n")
        # print("=" * 60)
        # print("LOSS DEBUG")
        # print("=" * 60)

        # print(
        #     "recognition_loss =",
        #     recognition_loss
        # )

        # print(
        #     "recognition_loss.requires_grad =",
        #     recognition_loss.requires_grad
        # )

        # print()

        # print(
        #     "forecasting_loss =",
        #     forecasting_loss
        # )

        # print(
        #     "forecasting_loss.requires_grad =",
        #     forecasting_loss.requires_grad
        # )

        # print()

        # print(
        #     "total_loss =",
        #     total_loss
        # )

        # print(
        #     "total_loss.requires_grad =",
        #     total_loss.requires_grad
        # )

        # print("=" * 60)
        # print()

        return {

            "total_loss":
                total_loss,

            "recognition_loss":
                recognition_loss,

            "forecasting_loss":
                forecasting_loss,
        }