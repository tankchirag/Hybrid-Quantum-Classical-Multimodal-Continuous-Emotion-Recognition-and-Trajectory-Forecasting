"""
training.optimizer
==================

Universal Optimizer Factory

Purpose
-------

Build optimizers for:

    • BaselineB1
    • BaselineB2
    • BaselineB3
    • QSAL

Design Goals
------------

✓ Dataset Agnostic

✓ Model Agnostic

✓ Quantum Compatible

✓ Config Driven

✓ Future Proof
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ==========================================================
# Utilities
# ==========================================================

def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """
    Count trainable parameters.
    """

    return sum(

        parameter.numel()

        for parameter
        in model.parameters()

        if parameter.requires_grad
    )


# ==========================================================
# Optimizer Factory
# ==========================================================

def build_optimizer(
    model: nn.Module,
    config,
):
    """
    Build optimizer from config.

    Expected Config
    ---------------

    optimizer:

        name: adamw

        lr: 0.0001

        weight_decay: 0.00001
    """

    optimizer_cfg = config.optimizer

    optimizer_name = (
        optimizer_cfg.name.lower()
    )

    learning_rate = float(
        optimizer_cfg.lr
    )

    weight_decay = float(
        getattr(
            optimizer_cfg,
            "weight_decay",
            0.0,
        )
    )

    # ------------------------------------------------------
    # AdamW
    # ------------------------------------------------------

    if optimizer_name == "adamw":

        return torch.optim.AdamW(

            model.parameters(),

            lr=learning_rate,

            weight_decay=weight_decay,
        )

    # ------------------------------------------------------
    # Adam
    # ------------------------------------------------------

    if optimizer_name == "adam":

        return torch.optim.Adam(

            model.parameters(),

            lr=learning_rate,

            weight_decay=weight_decay,
        )

    # ------------------------------------------------------
    # SGD
    # ------------------------------------------------------

    if optimizer_name == "sgd":

        momentum = float(
            getattr(
                optimizer_cfg,
                "momentum",
                0.9,
            )
        )

        return torch.optim.SGD(

            model.parameters(),

            lr=learning_rate,

            momentum=momentum,

            weight_decay=weight_decay,
        )

    raise ValueError(

        f"Unsupported optimizer: "
        f"{optimizer_name}"
    )


# ==========================================================
# Quantum Optimizer
# ==========================================================

def build_quantum_optimizer(
    model: nn.Module,
    config,
):
    """
    Optional optimizer supporting
    separate quantum learning rate.
    """

    optimizer_cfg = (
        config.optimizer
    )

    quantum_lr = getattr(
        optimizer_cfg,
        "quantum_lr",
        None,
    )

    if quantum_lr is None:

        return build_optimizer(
            model=model,
            config=config,
        )

    classical_lr = (
        optimizer_cfg.lr
    )

    weight_decay = getattr(
        optimizer_cfg,
        "weight_decay",
        0.0,
    )

    quantum_params = []

    classical_params = []

    for (
        name,
        parameter,
    ) in model.named_parameters():

        if not parameter.requires_grad:

            continue

        lower_name = (
            name.lower()
        )

        is_quantum = any(

            keyword in lower_name

            for keyword in [

                "quantum",

                "qnode",

                "qubit",

                "theta",

                "phi",

                "pqc",
            ]
        )

        if is_quantum:

            quantum_params.append(
                parameter
            )

        else:

            classical_params.append(
                parameter
            )

    return torch.optim.AdamW(

        [
            {
                "params":
                    classical_params,

                "lr":
                    classical_lr,
            },

            {
                "params":
                    quantum_params,

                "lr":
                    quantum_lr,
            },
        ],

        weight_decay=
        weight_decay,
    )


# ==========================================================
# Summary
# ==========================================================

def optimizer_summary(
    optimizer,
) -> dict:

    return {

        "optimizer":
            optimizer.__class__.__name__,

        "parameter_groups":
            len(
                optimizer.param_groups
            ),

        "learning_rates":
            [

                group["lr"]

                for group
                in optimizer.param_groups
            ],
    }
