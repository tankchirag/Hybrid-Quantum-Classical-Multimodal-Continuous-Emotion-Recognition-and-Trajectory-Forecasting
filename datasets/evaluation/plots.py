"""
datasets.evaluation.plots
=========================

Visualization Module

Generates visualizations for validating
synthetic multimodal datasets.

Purpose
-------

This module creates:

    • TSST protocol plots

    • Continuous label trajectories

    • Discrete label distributions

    • Physiological trajectories

    • Sensor visualizations

    • Correlation heatmaps

    • Subject variability plots

All generated figures are intended for:

    • Dataset debugging

    • Research validation

    • Thesis figures

    • Publication figures

Design Goals
------------

✓ Publication quality

✓ Thesis ready

✓ Dataset agnostic

✓ Label agnostic

✓ Modality agnostic

✓ Reusable
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


# ==========================================================
# Helper
# ==========================================================

def _ensure_dir(
    output_dir: str,
) -> Path:

    path = Path(output_dir)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


# ==========================================================
# Label Trajectories
# ==========================================================

def plot_continuous_labels(
    sample: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Plot all continuous trajectories.
    """

    save_dir = _ensure_dir(
        output_dir
    )

    labels = sample[
        "continuous_labels"
    ]

    plt.figure(
        figsize=(14, 6)
    )

    for (
        name,
        signal,
    ) in labels.items():

        plt.plot(
            signal.cpu().numpy(),
            label=name,
            linewidth=2,
        )

    plt.title(
        "Continuous Label Trajectories"
    )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "Value"
    )

    plt.grid(True)

    plt.legend()

    path = (
        save_dir
        / "continuous_labels.png"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
    )

    plt.close()

    return str(path)


# ==========================================================
# Discrete Labels
# ==========================================================

def plot_discrete_labels(
    sample: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Plot discrete label distributions.
    """

    labels = sample[
        "discrete_labels"
    ]

    save_dir = _ensure_dir(
        output_dir
    )

    n = len(labels)

    fig, axes = plt.subplots(
        n,
        1,
        figsize=(12, 3 * n),
    )

    if n == 1:
        axes = [axes]

    for ax, (
        name,
        signal,
    ) in zip(
        axes,
        labels.items(),
    ):

        values = (
            signal.cpu()
            .numpy()
            .astype(int)
        )

        unique, counts = np.unique(
            values,
            return_counts=True,
        )

        ax.bar(
            unique,
            counts,
        )

        ax.set_title(name)

        ax.grid(True)

    path = (
        save_dir
        / "discrete_labels.png"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
    )

    plt.close()

    return str(path)


# ==========================================================
# Sensor Visualization
# ==========================================================

def plot_modalities(
    sample: Dict[str, Any],
    output_dir: str,
    max_modalities: int = 8,
) -> str:
    """
    Plot first channels of modalities.
    """

    save_dir = _ensure_dir(
        output_dir
    )

    modalities = sample[
        "modalities"
    ]

    names = list(
        modalities.keys()
    )[:max_modalities]

    n = len(names)

    fig, axes = plt.subplots(
        n,
        1,
        figsize=(14, 3 * n),
    )

    if n == 1:
        axes = [axes]

    for ax, name in zip(
        axes,
        names,
    ):

        x = modalities[name]

        x = x.cpu().numpy()

        if x.ndim == 2:
            x = x[:, 0]

        ax.plot(
            x
        )

        ax.set_title(
            name.upper()
        )

        ax.grid(True)

    path = (
        save_dir
        / "modalities.png"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
    )

    plt.close()

    return str(path)


# ==========================================================
# Physiology
# ==========================================================

def plot_physiology(
    physiology,
    output_dir: str,
) -> str:
    """
    Plot physiological trajectories.
    """

    save_dir = _ensure_dir(
        output_dir
    )

    signals = {

        "heart_rate":
            physiology.heart_rate,

        "hrv":
            physiology.hrv,

        "respiration_rate":
            physiology.respiration_rate,

        "muscle_tension":
            physiology.muscle_tension,

        "sympathetic":
            physiology.sympathetic,
    }

    fig, axes = plt.subplots(
        len(signals),
        1,
        figsize=(14, 12),
    )

    for ax, (
        name,
        signal,
    ) in zip(
        axes,
        signals.items(),
    ):

        ax.plot(
            signal.cpu().numpy()
        )

        ax.set_title(
            name
        )

        ax.grid(True)

    path = (
        save_dir
        / "physiology.png"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
    )

    plt.close()

    return str(path)


# ==========================================================
# Correlation Heatmap
# ==========================================================

def plot_correlation_matrix(
    names,
    matrix,
    output_dir: str,
) -> str:
    """
    Plot correlation heatmap.
    """

    save_dir = _ensure_dir(
        output_dir
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        matrix,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    plt.colorbar()

    plt.xticks(
        range(len(names)),
        names,
        rotation=45,
    )

    plt.yticks(
        range(len(names)),
        names,
    )

    plt.title(
        "Label Correlation Matrix"
    )

    path = (
        save_dir
        / "correlation_matrix.png"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
    )

    plt.close()

    return str(path)


# ==========================================================
# Subject Variability
# ==========================================================

def plot_subject_variability(
    profiles,
    output_dir: str,
) -> str:
    """
    Visualize parameter variation
    across subjects.
    """

    save_dir = _ensure_dir(
        output_dir
    )

    stress_gain = [

        p.stress_gain

        for p in profiles
    ]

    recovery = [

        p.recovery_rate

        for p in profiles
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    axes[0].hist(
        stress_gain,
        bins=20,
    )

    axes[0].set_title(
        "Stress Gain"
    )

    axes[1].hist(
        recovery,
        bins=20,
    )

    axes[1].set_title(
        "Recovery Rate"
    )

    path = (
        save_dir
        / "subject_variability.png"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
    )

    plt.close()

    return str(path)


# ==========================================================
# Dashboard
# ==========================================================

def generate_all_figures(
    sample,
    physiology,
    corr_names,
    corr_matrix,
    subject_profiles,
    output_dir,
):
    """
    Generate all visual outputs.
    """

    return {

        "continuous_labels":
            plot_continuous_labels(
                sample,
                output_dir,
            ),

        "discrete_labels":
            plot_discrete_labels(
                sample,
                output_dir,
            ),

        "modalities":
            plot_modalities(
                sample,
                output_dir,
            ),

        "physiology":
            plot_physiology(
                physiology,
                output_dir,
            ),

        "correlation_matrix":
            plot_correlation_matrix(
                corr_names,
                corr_matrix,
                output_dir,
            ),

        "subject_variability":
            plot_subject_variability(
                subject_profiles,
                output_dir,
            ),
    }