"""
training.scheduler
==================

Universal Scheduler Factory

Supports

    • BaselineB1
    • BaselineB2
    • BaselineB3
    • QSAL

Design Goals
------------

✓ Config Driven

✓ Optimizer Agnostic

✓ Future Proof
"""

from __future__ import annotations

import torch


# ==========================================================
# Factory
# ==========================================================

def build_scheduler(
    optimizer,
    config,
):
    """
    Build scheduler from:

    scheduler:

        name:
    """

    scheduler_cfg = (
        config.scheduler
    )

    scheduler_name = (
        scheduler_cfg.name.lower()
    )

    # ------------------------------------------------------
    # None
    # ------------------------------------------------------

    if scheduler_name in [

        "none",

        "off",

        "disabled",
    ]:

        return None

    # ------------------------------------------------------
    # ReduceOnPlateau
    # ------------------------------------------------------

    if scheduler_name in [

        "plateau",

        "reduce_on_plateau",
    ]:

        return (
            torch.optim.lr_scheduler.
            ReduceLROnPlateau(

                optimizer,

                mode="min",

                factor=
                float(
                    scheduler_cfg.factor
                ),

                patience=
                int(
                    scheduler_cfg.patience
                ),

                min_lr=
                float(
                    getattr(
                        scheduler_cfg,
                        "min_lr",
                        1e-7,
                    )
                ),
            )
        )

    # ------------------------------------------------------
    # Cosine
    # ------------------------------------------------------

    if scheduler_name == "cosine":

        return (
            torch.optim.lr_scheduler.
            CosineAnnealingLR(

                optimizer,

                T_max=
                int(
                    getattr(
                        scheduler_cfg,
                        "t_max",
                        config.training.epochs,
                    )
                ),

                eta_min=
                float(
                    getattr(
                        scheduler_cfg,
                        "eta_min",
                        1e-7,
                    )
                ),
            )
        )

    # ------------------------------------------------------
    # Step
    # ------------------------------------------------------

    if scheduler_name == "step":

        return (
            torch.optim.lr_scheduler.
            StepLR(

                optimizer,

                step_size=
                int(
                    scheduler_cfg.step_size
                ),

                gamma=
                float(
                    scheduler_cfg.gamma
                ),
            )
        )

    raise ValueError(

        f"Unsupported scheduler: "
        f"{scheduler_name}"
    )


# ==========================================================
# Step
# ==========================================================

def step_scheduler(
    scheduler,
    validation_loss=None,
):
    """
    Safe scheduler stepping.
    """

    if scheduler is None:

        return

    if isinstance(

        scheduler,

        torch.optim.lr_scheduler.
        ReduceLROnPlateau,
    ):

        scheduler.step(
            validation_loss
        )

    else:

        scheduler.step()


# ==========================================================
# Utilities
# ==========================================================

def current_learning_rate(
    optimizer,
) -> float:

    return float(
        optimizer.param_groups[0]["lr"]
    )


def scheduler_summary(
    scheduler,
) -> dict:

    if scheduler is None:

        return {

            "scheduler":
                "none"
        }

    return {

        "scheduler":
            scheduler.__class__.__name__
    }