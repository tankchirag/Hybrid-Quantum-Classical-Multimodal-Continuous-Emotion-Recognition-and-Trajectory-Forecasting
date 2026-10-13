"""
utils.flops
===========

Model Complexity Analysis

Provides:

    • Parameters
    • FLOPs
    • MACs
    • Memory Estimates

Supports

    • BaselineB1
    • BaselineB2
    • BaselineB3
    • QSAL

Thesis Ready
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn


# ==========================================================
# Formatting
# ==========================================================

def human_readable(
    value: float,
) -> str:

    if value >= 1e12:
        return f"{value/1e12:.2f}T"

    if value >= 1e9:
        return f"{value/1e9:.2f}G"

    if value >= 1e6:
        return f"{value/1e6:.2f}M"

    if value >= 1e3:
        return f"{value/1e3:.2f}K"

    return str(value)


# ==========================================================
# Parameter Count
# ==========================================================

def count_parameters(
    model: nn.Module,
):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {

        "total": total,

        "trainable": trainable,
    }


# ==========================================================
# FLOP Estimation
# ==========================================================

def compute_flops(
    model: nn.Module,
    example_inputs,
):
    """
    Uses THOP if available.

    Returns
    -------
    dict
    """

    try:

        # Import thop dynamically to avoid hard dependency at module import time.
        import importlib

        try:
            thop = importlib.import_module("thop")
            profile = getattr(thop, "profile")
        except Exception:
            raise ImportError(
                "thop is not installed. Install it with `pip install thop` to enable FLOP computation."
            )

        macs, params = profile(

            model,

            inputs=example_inputs,

            verbose=False,
        )

        flops = 2 * macs

        return {

            "success": True,

            "macs": int(macs),

            "flops": int(flops),

            "params": int(params),
        }

    except Exception as e:

        return {

            "success": False,

            "error": str(e),

            "macs": None,

            "flops": None,

            "params": None,
        }


# ==========================================================
# Complete Report
# ==========================================================

def build_complexity_report(
    model: nn.Module,
    example_inputs=None,
):
    """
    Full complexity report.
    """

    param_info = (
        count_parameters(
            model
        )
    )

    report = {

        "model":
            model.__class__.__name__,

        "total_parameters":
            param_info["total"],

        "trainable_parameters":
            param_info["trainable"],

        "total_parameters_readable":
            human_readable(
                param_info["total"]
            ),

        "trainable_parameters_readable":
            human_readable(
                param_info["trainable"]
            ),
    }

    if example_inputs is not None:

        flop_info = compute_flops(

            model,

            example_inputs,
        )

        report.update({

            "flops":
                flop_info["flops"],

            "macs":
                flop_info["macs"],

            "flops_readable":
                human_readable(
                    flop_info["flops"]
                )
                if flop_info["flops"]
                else None,

            "macs_readable":
                human_readable(
                    flop_info["macs"]
                )
                if flop_info["macs"]
                else None,

            "flop_success":
                flop_info["success"],
        })

    return report


# ==========================================================
# Console Display
# ==========================================================

def print_complexity_report(
    model: nn.Module,
    example_inputs=None,
):

    report = (
        build_complexity_report(
            model,
            example_inputs,
        )
    )

    print()

    print("=" * 70)

    print(
        "MODEL COMPLEXITY REPORT"
    )

    print("=" * 70)

    print(
        f"Model      : "
        f"{report['model']}"
    )

    print(
        f"Parameters : "
        f"{report['total_parameters_readable']}"
    )

    print(
        f"Trainable  : "
        f"{report['trainable_parameters_readable']}"
    )

    if report.get(
        "flops_readable"
    ):

        print(
            f"FLOPs      : "
            f"{report['flops_readable']}"
        )

        print(
            f"MACs       : "
            f"{report['macs_readable']}"
        )

    print("=" * 70)

    print()


# ==========================================================
# Save JSON
# ==========================================================

def save_complexity_report(
    model: nn.Module,
    output_path: str,
    example_inputs=None,
):

    report = (
        build_complexity_report(
            model,
            example_inputs,
        )
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as fp:

        json.dump(

            report,

            fp,

            indent=4,
        )