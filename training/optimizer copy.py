"""
training.optimizer
==================

Optimizer Factory

Purpose
-------

Centralized optimizer construction for all
training pipelines.

Supported Models

    Baseline B1

    Baseline B2

    Baseline B3

    QSAL

    MultimodalQEmotionModel

Supported Optimizers
--------------------

    Adam

    AdamW

    SGD

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
import torch.nn as nn


# ============================================================
# Parameter Utilities
# ============================================================

def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """
    Count trainable parameters.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Optimizer Factory
# ============================================================

def build_optimizer(
    model: nn.Module,
    config,
):
    """
    Build optimizer from configuration.

    Expected Config
    ---------------

    optimizer = "adamw"

    learning_rate = 1e-4

    weight_decay = 1e-5

    optimizer_name = "adamw"

    Returns
    -------

    torch.optim.Optimizer
    """

    optimizer_name = getattr(
        config,
        "optimizer",
        getattr(
            config,
            "optimizer_name",
            "adamw",
        ),
    ).lower()

    learning_rate = getattr(
        config,
        "learning_rate",
        1e-4,
    )

    weight_decay = getattr(
        config,
        "weight_decay",
        1e-5,
    )

    # --------------------------------------------------------
    # AdamW
    # --------------------------------------------------------

    if optimizer_name == "adamw":

        return torch.optim.AdamW(

            model.parameters(),

            lr=learning_rate,

            weight_decay=weight_decay,

        )

    # --------------------------------------------------------
    # Adam
    # --------------------------------------------------------

    if optimizer_name == "adam":

        return torch.optim.Adam(

            model.parameters(),

            lr=learning_rate,

            weight_decay=weight_decay,

        )

    # --------------------------------------------------------
    # SGD
    # --------------------------------------------------------

    if optimizer_name == "sgd":

        momentum = getattr(
            config,
            "momentum",
            0.9,
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


# ============================================================
# Quantum Optimizer (Optional)
# ============================================================

def build_quantum_optimizer(
    model: nn.Module,
    config,
):
    """
    Optional advanced optimizer builder.

    Supports separate learning rates for:

        Classical Parameters

        Quantum Parameters

    If quantum_lr is missing,
    falls back to build_optimizer().
    """

    quantum_lr = getattr(
        config,
        "quantum_learning_rate",
        None,
    )

    if quantum_lr is None:

        return build_optimizer(
            model,
            config,
        )

    learning_rate = getattr(
        config,
        "learning_rate",
        1e-4,
    )

    weight_decay = getattr(
        config,
        "weight_decay",
        1e-5,
    )

    quantum_params = []

    classical_params = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        lower_name = name.lower()

        if any(
            keyword in lower_name
            for keyword in [

                "pqc",

                "quantum",

                "theta",

                "phi",

                "qnode",

                "qubit",
            ]
        ):

            quantum_params.append(
                param
            )

        else:

            classical_params.append(
                param
            )

    parameter_groups = [

        {
            "params":
                classical_params,

            "lr":
                learning_rate,
        },

        {
            "params":
                quantum_params,

            "lr":
                quantum_lr,
        },
    ]

    return torch.optim.AdamW(

        parameter_groups,

        weight_decay=weight_decay,

    )


# ============================================================
# Summary Utility
# ============================================================

def optimizer_summary(
    optimizer,
) -> dict:
    """
    Create optimizer summary.
    """

    learning_rates = [

        group["lr"]
        for group in optimizer.param_groups
    ]

    return {

        "optimizer":
            optimizer.__class__.__name__,

        "parameter_groups":
            len(
                optimizer.param_groups
            ),

        "learning_rates":
            learning_rates,
    }