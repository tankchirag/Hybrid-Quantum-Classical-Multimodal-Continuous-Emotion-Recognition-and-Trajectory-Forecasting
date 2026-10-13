"""
training.scheduler
==================

Learning Rate Scheduler Factory

Purpose
-------

Centralized scheduler creation for all models.

Supported Models
----------------

    BaselineB1

    BaselineB2

    BaselineB3

    QSAL

    MultimodalQEmotionModel

Supported Schedulers
--------------------

    none

    cosine

    cosine_warm_restarts

    step

    plateau

Design Goals
------------

✓ Dataset Agnostic

✓ Model Agnostic

✓ Fusion Agnostic

✓ Quantum Compatible

✓ Future Proof
"""

from __future__ import annotations

import torch


# ============================================================
# Scheduler Factory
# ============================================================

def build_scheduler(
    optimizer,
    config,
):
    """
    Build scheduler from config.

    Examples
    --------

    scheduler = "none"

    scheduler = "cosine"

    scheduler = "step"

    scheduler = "plateau"

    scheduler = "cosine_warm_restarts"
    """

    scheduler_name = getattr(
        config,
        "scheduler",
        "none",
    ).lower()

    # --------------------------------------------------------
    # No Scheduler
    # --------------------------------------------------------

    if scheduler_name == "none":
        return None

    # --------------------------------------------------------
    # Cosine Annealing
    # --------------------------------------------------------

    if scheduler_name == "cosine":

        t_max = getattr(
            config,
            "t_max",
            getattr(
                config,
                "epochs",
                100,
            ),
        )

        eta_min = getattr(
            config,
            "eta_min",
            1e-6,
        )

        return (
            torch.optim.lr_scheduler.
            CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=eta_min,
            )
        )

    # --------------------------------------------------------
    # Cosine Warm Restarts
    # --------------------------------------------------------

    if scheduler_name == "cosine_warm_restarts":

        t_0 = getattr(
            config,
            "t_0",
            10,
        )

        t_mult = getattr(
            config,
            "t_mult",
            2,
        )

        eta_min = getattr(
            config,
            "eta_min",
            1e-6,
        )

        return (
            torch.optim.lr_scheduler.
            CosineAnnealingWarmRestarts(
                optimizer,
                T_0=t_0,
                T_mult=t_mult,
                eta_min=eta_min,
            )
        )

    # --------------------------------------------------------
    # Step Scheduler
    # --------------------------------------------------------

    if scheduler_name == "step":

        step_size = getattr(
            config,
            "step_size",
            10,
        )

        gamma = getattr(
            config,
            "gamma",
            0.1,
        )

        return (
            torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma,
            )
        )

    # --------------------------------------------------------
    # Reduce On Plateau
    # --------------------------------------------------------

    if scheduler_name == "plateau":

        factor = getattr(
            config,
            "factor",
            0.5,
        )

        patience = getattr(
            config,
            "patience",
            5,
        )

        min_lr = getattr(
            config,
            "min_lr",
            1e-7,
        )

        return (
            torch.optim.lr_scheduler.
            ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            )
        )

    raise ValueError(
        f"Unsupported scheduler: "
        f"{scheduler_name}"
    )


# ============================================================
# Scheduler Step
# ============================================================

def step_scheduler(
    scheduler,
    validation_loss=None,
):
    """
    Step scheduler safely.

    Parameters
    ----------

    scheduler

        Scheduler instance.

    validation_loss

        Required for
        ReduceLROnPlateau.
    """

    if scheduler is None:
        return

    if isinstance(
        scheduler,
        torch.optim.lr_scheduler.
        ReduceLROnPlateau,
    ):

        if validation_loss is None:

            raise ValueError(
                "ReduceLROnPlateau "
                "requires validation_loss."
            )

        scheduler.step(
            validation_loss
        )

    else:

        scheduler.step()


# ============================================================
# Utilities
# ============================================================

def current_learning_rate(
    optimizer,
) -> float:
    """
    Return current learning rate.
    """

    return float(
        optimizer.param_groups[0]["lr"]
    )


def scheduler_summary(
    scheduler,
) -> dict:
    """
    Scheduler metadata.
    """

    if scheduler is None:

        return {
            "scheduler": "none"
        }

    return {

        "scheduler":
            scheduler.__class__.__name__
    }