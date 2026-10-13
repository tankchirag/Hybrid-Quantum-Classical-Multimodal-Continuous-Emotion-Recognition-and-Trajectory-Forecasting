"""
utils.params
============

Parameter Analysis Utilities

Purpose
-------

Provides robust parameter accounting for:

    • Baseline B1
    • Baseline B2
    • Baseline B3
    • QSAL

Outputs

    • Total Parameters

    • Trainable Parameters

    • Frozen Parameters

    • Parameter Breakdown

    • Memory Footprint

    • Thesis Reporting

Design Goals
------------

✓ Reproducible

✓ Thesis Ready

✓ JSON Compatible

✓ Model Agnostic

✓ Future Proof
"""

from __future__ import annotations

from collections import defaultdict

import json

import torch.nn as nn


# ==========================================================
# Formatting
# ==========================================================

def format_count(
    value: int,
) -> str:
    """
    Human-readable parameter count.
    """

    if value >= 1_000_000_000:

        return (
            f"{value / 1_000_000_000:.2f}B"
        )

    if value >= 1_000_000:

        return (
            f"{value / 1_000_000:.2f}M"
        )

    if value >= 1_000:

        return (
            f"{value / 1_000:.2f}K"
        )

    return str(value)


# ==========================================================
# Core Counts
# ==========================================================

def count_parameters(
    model: nn.Module,
) -> dict:
    """
    Count model parameters.
    """

    total = 0

    trainable = 0

    frozen = 0

    for p in model.parameters():

        n = p.numel()

        total += n

        if p.requires_grad:

            trainable += n

        else:

            frozen += n

    return {

        "total": total,

        "trainable": trainable,

        "frozen": frozen,
    }


# ==========================================================
# Memory Estimate
# ==========================================================

def parameter_memory_mb(
    model: nn.Module,
) -> float:
    """
    Parameter memory only.

    FP32 assumption.
    """

    total = sum(

        p.numel()

        for p in model.parameters()
    )

    bytes_used = total * 4

    return bytes_used / (
        1024 ** 2
    )


# ==========================================================
# Module Breakdown
# ==========================================================

def parameter_breakdown(
    model: nn.Module,
) -> dict:
    """
    Count parameters per top module.
    """

    breakdown = defaultdict(int)

    total = 0

    for name, param in model.named_parameters():

        module_name = (
            name.split(".")[0]
        )

        count = param.numel()

        breakdown[
            module_name
        ] += count

        total += count

    results = {}

    for (
        module,
        count,
    ) in sorted(
        breakdown.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        results[module] = {

            "parameters":
                count,

            "percent":
                round(
                    (
                        count
                        /
                        total
                    )
                    * 100.0,
                    2,
                ),
        }

    return results


# ==========================================================
# Full Report
# ==========================================================

def build_parameter_report(
    model: nn.Module,
    model_name: str | None = None,
) -> dict:
    """
    Full thesis report.
    """

    counts = count_parameters(
        model
    )

    report = {

        "model":
            model_name
            or model.__class__.__name__,

        "total_parameters":
            counts["total"],

        "trainable_parameters":
            counts["trainable"],

        "frozen_parameters":
            counts["frozen"],

        "total_parameters_readable":
            format_count(
                counts["total"]
            ),

        "trainable_parameters_readable":
            format_count(
                counts["trainable"]
            ),

        "memory_mb":
            round(
                parameter_memory_mb(
                    model
                ),
                2,
            ),

        "module_breakdown":
            parameter_breakdown(
                model
            ),
    }

    return report


# ==========================================================
# Pretty Printing
# ==========================================================

def print_parameter_report(
    model: nn.Module,
    model_name: str | None = None,
):
    """
    Console report.
    """

    report = build_parameter_report(
        model,
        model_name,
    )

    print()

    print("=" * 70)

    print(
        "PARAMETER REPORT"
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

    print(
        f"Memory MB  : "
        f"{report['memory_mb']:.2f}"
    )

    print()

    print(
        "Module Breakdown"
    )

    print("-" * 70)

    for (
        module,
        info,
    ) in report[
        "module_breakdown"
    ].items():

        print(

            f"{module:<30}"

            f"{format_count(info['parameters']):>12}"

            f"{info['percent']:>10.2f}%"
        )

    print("=" * 70)

    print()


# ==========================================================
# JSON Export
# ==========================================================

def save_parameter_report(
    model: nn.Module,
    filepath: str,
    model_name: str | None = None,
):
    """
    Save report to JSON.
    """

    report = build_parameter_report(
        model,
        model_name,
    )

    with open(
        filepath,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
        )