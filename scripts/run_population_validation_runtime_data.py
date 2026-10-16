"""
run_population_validation.py
============================

Population-Level Validation Pipeline

Purpose
-------
Evaluate dataset-wide quality of the synthetic
multimodal dataset.

Unlike:

    run_physiological_validation.py

which validates a SINGLE sample,

this pipeline validates a LARGE POPULATION
of generated samples.

Research Questions
------------------

1. Sample Variability

    Are generated samples different?

2. Subject Variability

    Are generated subjects different?

3. Session Variability

    Are sessions of the same subject
    non-identical?

4. Controllability

    Does generated output follow
    target emotional conditions?

Outputs
-------

results/
└── population_validation/
    ├── population_validation.json
    ├── population_report.md
    ├── sample_variability.csv
    ├── subject_variability.csv
    ├── session_variability.csv
    └── controllability.csv

Usage
-----

python -m scripts.run_population_validation

or

python scripts/run_population_validation.py

Notes
-----

This script should be executed with
significantly more samples than the
single-sample validation pipeline.

Recommended:

    num_samples = 100

or

    num_samples = 500

or larger.

The resulting files can be used in:

    thesis

    supervisor review

    ablation analysis

    publication tables
"""

from __future__ import annotations

from pathlib import Path
import sys


# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent.parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ==========================================================
# Imports
# ==========================================================

from datasets.ulm_tsst_synthetic import (
    SyntheticULMTSSTDataset,
)

from datasets.evaluation.population_validation import (
    PopulationValidator,
)


# ==========================================================
# Configuration
# ==========================================================

OUTPUT_ROOT = Path(
    "results/population_validation"
)

NUM_SAMPLES = 100

NUM_SUBJECTS = 30

NUM_SESSIONS = 5

DURATION_SECONDS = 300


# ==========================================================
# Modalities
# ==========================================================

MODALITIES = {

    "ecg": True,
    "eda": True,
    "resp": True,
    "bpm": True,
    "eeg": True,
    "emg": True,

    "eye": True,
    "audio": True,
    "video": True,
    "text": True,
}


# ==========================================================
# Continuous Labels
# ==========================================================

CONTINUOUS_LABELS = {

    "valence": True,
    "arousal": True,
    "dominance": True,

    "stress": True,
    "workload": True,
    "fatigue": True,

    "engagement": True,
}


# ==========================================================
# Discrete Labels
# ==========================================================

DISCRETE_LABELS = {

    "emotion_class": True,

    "stress_class": True,

    "workload_class": True,

    "alertness_class": True,

    "drowsiness_class": True,
}


# ==========================================================
# Sampling Rates
# ==========================================================

SAMPLING_RATES = {

    "ecg": 1000,

    "eda": 128,

    "resp": 64,

    "bpm": 25,

    "eeg": 256,

    "emg": 1000,

    "eye": 30,

    "audio": 16000,

    "video": 30,

    "text": 25,
}


# ==========================================================
# Dataset Factory
# ==========================================================

def build_dataset():
    """
    Create synthetic dataset.

    Modify parameters as needed
    for large-scale evaluation.
    """

    return SyntheticULMTSSTDataset(

        num_samples=
        NUM_SAMPLES,

        duration_seconds=
        DURATION_SECONDS,

        num_subjects=
        NUM_SUBJECTS,

        num_sessions=
        NUM_SESSIONS,

        device="cpu",

        modalities=
        MODALITIES,

        continuous_labels=
        CONTINUOUS_LABELS,

        discrete_labels=
        DISCRETE_LABELS,

        sampling_rates=
        SAMPLING_RATES,
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "POPULATION VALIDATION"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    print(
        "\n[1/3] Building dataset..."
    )

    dataset = build_dataset()

    print(
        f"Dataset Size: "
        f"{len(dataset)} samples"
    )

    # --------------------------------------------------
    # Population Validation
    # --------------------------------------------------

    print(
        "\n[2/3] Running population validation..."
    )

    validator = (
        PopulationValidator()
    )

    summary = (
        validator.validate(
            dataset=dataset,
            output_dir=
            OUTPUT_ROOT,
        )
    )

    # --------------------------------------------------
    # Finished
    # --------------------------------------------------

    print(
        "\n[3/3] Validation completed."
    )

    print(
        "\nSummary"
    )

    print(
        "-" * 40
    )

    for key, value in (
        summary.items()
    ):

        print(
            f"{key}: {value}"
        )

    print(
        "\nResults written to:"
    )

    print(
        OUTPUT_ROOT.resolve()
    )

    print(
        "\nGenerated Files:"
    )

    print(
        "  population_validation.json"
    )

    print(
        "  population_report.md"
    )

    print(
        "  sample_variability.csv"
    )

    print(
        "  subject_variability.csv"
    )

    print(
        "  session_variability.csv"
    )

    print(
        "  controllability.csv"
    )

    print(
        "\nDone."
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()