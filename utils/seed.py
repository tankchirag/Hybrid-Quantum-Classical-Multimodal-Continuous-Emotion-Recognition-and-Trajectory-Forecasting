"""
utils.seed
==========

Reproducibility Utilities

Purpose
-------

Provide deterministic and reproducible
training across:

    • Baseline B1
    • Baseline B2
    • Baseline B3
    • QSAL

Controls

    • Python RNG
    • NumPy RNG
    • PyTorch CPU RNG
    • PyTorch CUDA RNG
    • CuDNN Determinism

Design Goals
------------

✓ Reproducible

✓ Thesis Ready

✓ Config Driven

✓ Multi-GPU Safe

✓ Future Proof
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


# ==========================================================
# Set Seed
# ==========================================================

def set_seed(
    seed: int = 42,
    deterministic: bool = True,
):
    """
    Configure all supported random generators.

    Parameters
    ----------
    seed

        Global experiment seed.

    deterministic

        If True:
            maximize reproducibility

        If False:
            maximize speed
    """

    # ------------------------------------------------------
    # Python
    # ------------------------------------------------------

    random.seed(seed)

    # ------------------------------------------------------
    # NumPy
    # ------------------------------------------------------

    np.random.seed(seed)

    # ------------------------------------------------------
    # PyTorch CPU
    # ------------------------------------------------------

    torch.manual_seed(seed)

    # ------------------------------------------------------
    # PyTorch CUDA
    # ------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------
    # Python Hash Seed
    # ------------------------------------------------------

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    # ------------------------------------------------------
    # CuDNN
    # ------------------------------------------------------

    torch.backends.cudnn.deterministic = (
        deterministic
    )

    torch.backends.cudnn.benchmark = (
        not deterministic
    )

    # ------------------------------------------------------
    # PyTorch Determinism
    # ------------------------------------------------------

    try:

        torch.use_deterministic_algorithms(
            deterministic
        )

    except Exception:

        pass


# ==========================================================
# DataLoader Workers
# ==========================================================

def seed_worker(
    worker_id: int,
):
    """
    Deterministic DataLoader workers.

    Use with:

        DataLoader(
            ...
            worker_init_fn=seed_worker
        )
    """

    worker_seed = (
        torch.initial_seed()
        % (2**32)
    )

    np.random.seed(
        worker_seed
    )

    random.seed(
        worker_seed
    )


# ==========================================================
# Generator
# ==========================================================

def build_generator(
    seed: int = 42,
):
    """
    Deterministic generator for DataLoader.

    Example
    -------

    generator = build_generator(42)

    DataLoader(
        dataset,
        generator=generator,
    )
    """

    g = torch.Generator()

    g.manual_seed(seed)

    return g


# ==========================================================
# Metadata
# ==========================================================

def seed_report(
    seed: int,
    deterministic: bool,
):
    """
    Experiment metadata.
    """

    return {

        "seed":
            seed,

        "deterministic":
            deterministic,

        "cuda_available":
            torch.cuda.is_available(),

        "torch_version":
            torch.__version__,
    }


# ==========================================================
# Pretty Print
# ==========================================================

def print_seed_report(
    seed: int,
    deterministic: bool,
):

    print()

    print("=" * 70)

    print(
        "REPRODUCIBILITY SETTINGS"
    )

    print("=" * 70)

    print(
        f"Seed          : {seed}"
    )

    print(
        f"Deterministic : "
        f"{deterministic}"
    )

    print(
        f"CUDA Available: "
        f"{torch.cuda.is_available()}"
    )

    print(
        f"PyTorch       : "
        f"{torch.__version__}"
    )

    print("=" * 70)

    print()