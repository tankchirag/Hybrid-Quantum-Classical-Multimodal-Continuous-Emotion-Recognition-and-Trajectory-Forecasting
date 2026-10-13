"""
utils.latency
=============

Runtime and Latency Analysis

Purpose
-------

Measure:

    • Inference Latency

    • Throughput

    • Forward Pass Time

    • GPU Runtime

    • CPU Runtime

    • Thesis Comparisons

Supports

    • BaselineB1
    • BaselineB2
    • BaselineB3
    • QSAL

Thesis Ready
"""

from __future__ import annotations

import json
import time

import torch
import torch.nn as nn


# ==========================================================
# Formatting
# ==========================================================

def format_ms(
    milliseconds: float,
) -> str:

    return f"{milliseconds:.3f} ms"


# ==========================================================
# Latency
# ==========================================================

@torch.no_grad()
def measure_latency(
    model: nn.Module,
    example_inputs,
    warmup_runs: int = 20,
    measurement_runs: int = 100,
):
    """
    Measure inference latency.

    Returns
    -------
    dict
    """

    device = next(
        model.parameters()
    ).device

    model.eval()

    # --------------------------------------
    # Warmup
    # --------------------------------------

    for _ in range(
        warmup_runs
    ):

        _ = model(
            *example_inputs
        )

    # --------------------------------------
    # GPU Timing
    # --------------------------------------

    if device.type == "cuda":

        torch.cuda.synchronize()

    timings = []

    for _ in range(
        measurement_runs
    ):

        if device.type == "cuda":

            torch.cuda.synchronize()

        start = time.perf_counter()

        _ = model(
            *example_inputs
        )

        if device.type == "cuda":

            torch.cuda.synchronize()

        end = time.perf_counter()

        timings.append(
            end - start
        )

    avg_seconds = (
        sum(timings)
        /
        len(timings)
    )

    min_seconds = min(
        timings
    )

    max_seconds = max(
        timings
    )

    latency_ms = (
        avg_seconds * 1000.0
    )

    throughput = (
        1.0 / avg_seconds
    )

    return {

        "device":
            device.type,

        "warmup_runs":
            warmup_runs,

        "measurement_runs":
            measurement_runs,

        "avg_latency_ms":
            latency_ms,

        "min_latency_ms":
            min_seconds * 1000.0,

        "max_latency_ms":
            max_seconds * 1000.0,

        "throughput_samples_per_sec":
            throughput,
    }


# ==========================================================
# Training Step Latency
# ==========================================================

def measure_training_step(
    model: nn.Module,
    optimizer,
    criterion,
    example_inputs,
    target,
):
    """
    Single training step timing.
    """

    model.train()

    device = next(
        model.parameters()
    ).device

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    optimizer.zero_grad()

    outputs = model(
        *example_inputs
    )

    loss = criterion(
        outputs,
        target,
    )

    loss.backward()

    optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        -
        start
    )

    return {

        "training_step_ms":
            elapsed * 1000.0
    }


# ==========================================================
# Report
# ==========================================================

def print_latency_report(
    model,
    example_inputs,
):
    """
    Pretty console report.
    """

    report = (
        measure_latency(
            model,
            example_inputs,
        )
    )

    print()

    print("=" * 70)

    print(
        "LATENCY REPORT"
    )

    print("=" * 70)

    print(
        f"Model      : "
        f"{model.__class__.__name__}"
    )

    print(
        f"Device     : "
        f"{report['device']}"
    )

    print(
        f"Average    : "
        f"{report['avg_latency_ms']:.3f} ms"
    )

    print(
        f"Minimum    : "
        f"{report['min_latency_ms']:.3f} ms"
    )

    print(
        f"Maximum    : "
        f"{report['max_latency_ms']:.3f} ms"
    )

    print(
        f"Throughput : "
        f"{report['throughput_samples_per_sec']:.2f}"
        f" samples/sec"
    )

    print("=" * 70)

    print()


# ==========================================================
# Save
# ==========================================================

def save_latency_report(
    model,
    example_inputs,
    output_path,
):
    """
    Save latency report JSON.
    """

    report = (
        measure_latency(
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