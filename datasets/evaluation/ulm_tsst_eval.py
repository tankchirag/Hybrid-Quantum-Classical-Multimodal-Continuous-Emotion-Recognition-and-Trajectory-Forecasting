"""
datasets.evaluation.ulm_tsst_eval
=================================

Synthetic Dataset Evaluation Pipeline

This script performs a complete validation of the
Synthetic Multimodal Human-State Framework.

Validation Levels
-----------------

1. Dataset Integrity

2. Dataset Statistics

3. Continuous Label Analysis

4. Discrete Label Analysis

5. Correlation Analysis

6. Scientific Validation

7. Figure Generation

8. Report Generation

Outputs
-------

results/dataset_validation/

    continuous_labels.png

    discrete_labels.png

    modalities.png

    physiology.png

    correlation_matrix.png

    subject_variability.png

    validation_report.md

Usage
-----

python datasets/evaluation/ulm_tsst_eval.py
"""

from __future__ import annotations

# ==========================================================
# PROJECT ROOT SETUP
# ==========================================================

import sys
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

# ==========================================================
# IMPORTS
# ==========================================================

from datasets.ulm_tsst_synthetic import (
    SyntheticULMTSSTDataset,
)

from datasets.evaluation.statistics import (
    DatasetStatistics,
    print_statistics_report,
)

from datasets.evaluation.correlations import (
    CorrelationAnalyzer,
    print_correlation_report,
)

from datasets.evaluation.plots import (
    generate_all_figures,
)

from datasets.evaluation.report import (
    ValidationReportGenerator,
)

from datasets.generators.subject_model import (
    SubjectModelGenerator,
)

from datasets.generators.physiological_generator import (
    PhysiologicalGenerator,
)

from datasets.generators.tsst_protocol import (
    TSSTProtocolGenerator,
)

from datasets.generators.latent_state_generator import (
    LatentStateGenerator,
)


# ==========================================================
# Configuration
# ==========================================================

OUTPUT_DIR = (
    "results/dataset_validation"
)

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

CONTINUOUS_LABELS = {

    "valence": True,
    "arousal": True,

    "stress": True,
    "workload": True,

    "fatigue": True,
    "dominance": True,
    "engagement": True,
}

DISCRETE_LABELS = {

    "emotion_class": True,

    "stress_class": True,

    "workload_class": True,

    "alertness_class": True,

    "drowsiness_class": True,

    "kss_score": True,
}

SAMPLING_RATES = {

    "ecg": 1000,
    "eda": 128,
    "resp": 64,
    "bpm": 4,

    "eeg": 256,
    "emg": 1000,

    "eye": 120,

    "audio": 100,
    "video": 25,
    "text": 5,
}


# ==========================================================
# Dataset Creation
# ==========================================================

def build_dataset():

    return SyntheticULMTSSTDataset(

        num_samples=10,

        duration_seconds=600,

        modalities=MODALITIES,

        continuous_labels=
            CONTINUOUS_LABELS,

        discrete_labels=
            DISCRETE_LABELS,

        sampling_rates=
            SAMPLING_RATES,

        save_dir=None,

        device="cpu",
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print("\n")
    print("=" * 70)
    print("ULM-TSST SYNTHETIC DATASET VALIDATION")
    print("=" * 70)

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------

    dataset = build_dataset()

    sample = dataset[0]

    print(
        "\nDataset instantiated successfully."
    )

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    stats = DatasetStatistics(
        dataset
    )

    statistics_report = (
        stats.generate_report()
    )

    print_statistics_report(
        statistics_report
    )

    # ------------------------------------------------------
    # Correlations
    # ------------------------------------------------------

    analyzer = CorrelationAnalyzer(
        sample
    )

    print_correlation_report(
        analyzer
    )

    corr_names, corr_matrix = (
        analyzer.correlation_matrix()
    )

    correlation_results = (
        analyzer.expected_relationships()
    )

    validation_checks = (
        analyzer.validation_checks()
    )

    # ------------------------------------------------------
    # Create physiology sample for plotting
    # ------------------------------------------------------

    subject_generator = (
        SubjectModelGenerator()
    )

    protocol_generator = (
        TSSTProtocolGenerator(
            duration_seconds=600
        )
    )

    latent_generator = (
        LatentStateGenerator()
    )

    physiology_generator = (
        PhysiologicalGenerator()
    )

    subject = (
        subject_generator.generate(
            0
        )
    )

    protocol = (
        protocol_generator.generate()
    )

    latent = (
        latent_generator.generate(
            subject,
            protocol,
        )
    )

    physiology = (
        physiology_generator.generate(
            latent,
            subject,
        )
    )

    # ------------------------------------------------------
    # Subject variability
    # ------------------------------------------------------

    profiles = [

        subject_generator.generate(i)

        for i in range(30)
    ]

    # ------------------------------------------------------
    # Figures
    # ------------------------------------------------------

    figures = generate_all_figures(

        sample=sample,

        physiology=physiology,

        corr_names=corr_names,

        corr_matrix=corr_matrix,

        subject_profiles=profiles,

        output_dir=OUTPUT_DIR,
    )

    print(
        "\nFigures generated successfully."
    )

    # ------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------

    report_generator = (
        ValidationReportGenerator(
            output_dir=OUTPUT_DIR
        )
    )

    report_file = (
        report_generator.save_report(

            statistics_report=
                statistics_report,

            correlation_results=
                correlation_results,

            validation_checks=
                validation_checks,

            generated_figures=
                figures,
        )
    )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        "DATASET VALIDATION COMPLETE"
    )

    print("=" * 70)

    print(
        f"\nReport:\n  {report_file}"
    )

    print(
        f"\nFigures:\n  {OUTPUT_DIR}"
    )

    print(
        "\nValidation successful."
    )

    print("=" * 70)
    print()


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":
    main()