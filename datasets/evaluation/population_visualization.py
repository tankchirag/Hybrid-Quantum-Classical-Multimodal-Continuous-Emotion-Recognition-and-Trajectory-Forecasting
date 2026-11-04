"""
population_visualization.py
===========================

Population Visualization Framework

Purpose
-------
Generate publication-quality visualizations
from population validation outputs.

Inputs
------

sample_variability.csv

subject_variability.csv

session_variability.csv

Outputs
-------

valence_distribution.png

arousal_distribution.png

stress_distribution.png

workload_distribution.png

label_correlation_matrix.png

subject_variability.png

session_variability.png

Design Goals
------------

✓ Thesis Ready

✓ Supervisor Ready

✓ Publication Ready

✓ Dataset Agnostic

✓ Population Focused
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    sns = None


class PopulationVisualizer:

    def __init__(
        self,
        validation_dir,
    ):

        self.validation_dir = Path(
            validation_dir
        )

        self.plots_dir = (
            self.validation_dir
            / "plots"
        )

        self.plots_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.sample_df = pd.read_csv(
            self.validation_dir
            / "sample_variability.csv"
        )

        self.subject_df = pd.read_csv(
            self.validation_dir
            / "subject_variability.csv"
        )

        self.session_df = pd.read_csv(
            self.validation_dir
            / "session_variability.csv"
        )

    # ======================================================
    # Utilities
    # ======================================================

    def _save(
        self,
        filename,
    ):

        plt.tight_layout()

        plt.savefig(
            self.plots_dir / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    # ======================================================
    # Distributions
    # ======================================================

    def plot_distribution(
        self,
        label_name,
    ):

        if label_name not in self.sample_df:
            return

        plt.figure(
            figsize=(8, 5)
        )

        sns.histplot(

            self.sample_df[
                label_name
            ],

            bins=30,

            kde=True,
        )

        plt.title(
            f"{label_name.capitalize()} Distribution"
        )

        plt.xlabel(
            label_name.capitalize()
        )

        plt.ylabel(
            "Count"
        )

        self._save(
            f"{label_name}_distribution.png"
        )

    # ======================================================
    # Correlation Matrix
    # ======================================================

    def plot_correlation_matrix(
        self,
    ):

        numeric = (

            self.sample_df

            .select_dtypes(
                include=np.number
            )
        )

        corr = numeric.corr()

        plt.figure(
            figsize=(10, 8)
        )

        sns.heatmap(

            corr,

            annot=True,

            cmap="coolwarm",

            center=0,
        )

        plt.title(
            "Label Correlation Matrix"
        )

        self._save(
            "label_correlation_matrix.png"
        )

    # ======================================================
    # Subject Variability
    # ======================================================

    def plot_subject_variability(
        self,
    ):

        if len(
            self.subject_df
        ) == 0:

            return

        if (
            "subject_id"
            not in self.subject_df
        ):

            return

        labels = [

            c

            for c in self.subject_df.columns

            if c != "subject_id"
        ]

        if len(labels) == 0:
            return

        means = []

        for label in labels:

            means.append(

                self.subject_df[
                    label
                ].std()
            )

        plt.figure(
            figsize=(12, 6)
        )

        plot_df = self.subject_df[
            labels
        ]

        if sns is not None:

            sns.boxplot(
                data=plot_df
            )

        else:

            plot_df.boxplot()

        plt.ylabel(
            "Value"
        )

        plt.title(
            "Subject-Level Label Distribution"
        )

        plt.xticks(
            rotation=30
        )

        self._save(
            "subject_variability.png"
        )


    # ======================================================
    # Subject Count Distribution
    # ======================================================

    def plot_subject_sample_counts(
        self,
    ):

        if (
            "subject_id"
            not in self.session_df
        ):
            return

        counts = (

            self.session_df

            .groupby(
                "subject_id"
            )

            .size()
        )

        plt.figure(
            figsize=(10, 5)
        )

        plt.bar(

            counts.index,

            counts.values,
        )

        plt.title(
            "Samples per Subject"
        )

        plt.xlabel(
            "Subject ID"
        )

        plt.ylabel(
            "Number of Samples"
        )

        self._save(
            "subject_sample_counts.png"
        )


    # ======================================================
    # Session Variability
    # ======================================================

    def plot_session_variability(
        self,
    ):

        if len(
            self.session_df
        ) == 0:

            return

        labels = [

            c

            for c in self.session_df.columns

            if c not in [

                "subject_id",

                "session_id",
            ]
        ]

        if len(labels) == 0:
            return

        variability = []

        for label in labels:

            variability.append(

                self.session_df[
                    label
                ].std()
            )

        plt.figure(
            figsize=(12, 6)
        )

        plot_df = self.session_df[
            labels
        ]

        if sns is not None:

            sns.boxplot(
                data=plot_df
            )

        else:

            plot_df.boxplot()

        plt.ylabel(
            "Value"
        )

        plt.title(
            "Session-Level Label Distribution"
        )

        plt.xticks(
            rotation=30
        )

        self._save(
            "session_variability.png"
        )

    # ======================================================
    # Main
    # ======================================================

    def generate_all(
        self,
    ):

        self.plot_distribution(
            "valence"
        )

        self.plot_distribution(
            "arousal"
        )

        self.plot_distribution(
            "stress"
        )

        self.plot_distribution(
            "workload"
        )

        self.plot_correlation_matrix()

        self.plot_subject_variability()

        self.plot_session_variability()

        self.plot_subject_sample_counts()

        print(
            "\nPopulation visualizations saved to:"
        )

        print(
            self.plots_dir
        )