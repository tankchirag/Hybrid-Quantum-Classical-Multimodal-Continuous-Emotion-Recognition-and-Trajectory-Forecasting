"""
run_physiological_validation.py
===============================

End-to-End Physiological Validation Pipeline

Purpose
-------
Generate all validation artifacts required
for supervisor review.

Pipeline
--------

Dataset Sample
      ↓

PhysiologicalValidator
      ↓

EmotionTimelinePlotter
      ↓

ECGValidator
      ↓

EDAValidator
      ↓

EMGValidator
      ↓

SignalReportGenerator
      ↓

Validation Artifacts

Outputs
-------

results/
└── physiological_validation/
    └── sample_000/
        ├── emotion_summary.json
        ├── emotion_timeline.png
        ├── ecg_validation.png
        ├── ecg_summary.json
        ├── eda_validation.png
        ├── eda_summary.json
        ├── emg_validation.png
        ├── emg_summary.json
        └── validation_report.md

Usage
-----

python scripts/run_physiological_validation.py
"""

from pathlib import Path

import sys

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent.parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from datasets.ulm_tsst_synthetic import (
    SyntheticULMTSSTDataset,
)

from datasets.evaluation.physiological_validation import (
    PhysiologicalValidator,
)

from datasets.evaluation.emotion_timeline import (
    EmotionTimelinePlotter,
)

from datasets.evaluation.ecg_validation import (
    ECGValidator,
)

from datasets.evaluation.eda_validation import (
    EDAValidator,
)

from datasets.evaluation.emg_validation import (
    EMGValidator,
)

from datasets.evaluation.signal_report import (
    SignalReportGenerator,
)

from datasets.evaluation.scenario_validation import (
    ScenarioValidator,
)

from datasets.evaluation.multimodal_alignment import (
    MultimodalAlignment,
)

# ==========================================================
# Configuration
# ==========================================================

OUTPUT_ROOT = Path(
    "results/physiological_validation"
)

SAMPLE_INDEX = 0 # Index of the sample to validate


# ==========================================================
# Dataset Factory
# ==========================================================

def build_dataset():
    """
    Create dataset instance.

    Replace with your project's
    actual configuration.
    """

    return SyntheticULMTSSTDataset(

        num_samples=1,

        duration_seconds=300,

        num_subjects=30,

        num_sessions=5,

        device="cpu",

        modalities={
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
        },

        continuous_labels={
            "valence": True,
            "arousal": True,
            "dominance": True,
            "stress": True,
            "workload": True,
            "fatigue": True,
            "engagement": True,
        },

        discrete_labels={
            "emotion_class": True,
            "stress_class": True,
            "workload_class": True,
            "alertness_class": True,
            "drowsiness_class": True,
        },

        sampling_rates={
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
        },
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print(
        "\n[1/9] Loading dataset..."
    )

    dataset = build_dataset()

    sample = dataset[
        SAMPLE_INDEX
    ]

    sample_dir = (
        OUTPUT_ROOT
        / f"sample_{SAMPLE_INDEX:03d}"
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print(
        "\n[2/9] Creating summary..."
    )

    validator = (
        PhysiologicalValidator()
    )

    validator.validate(
        sample=sample,
        output_json=
        sample_dir
        / "emotion_summary.json",
    )

    # --------------------------------------------------
    # Emotion Timeline
    # --------------------------------------------------

    print(
        "\n[3/9] Creating emotion timeline..."
    )

    timeline = (
        EmotionTimelinePlotter()
    )

    timeline.generate(
        sample=sample,
        output_directory=
        sample_dir,
    )

    # --------------------------------------------------
    # Scenario Validation
    # --------------------------------------------------

    print(
        "\n[4/9] Scenario validation..."
    )

    scenario_summary = (
        ScenarioValidator()
        .validate(
            sample=sample,
            output_dir=sample_dir,
        )
    )

    # --------------------------------------------------
    # ECG
    # --------------------------------------------------

    print(
        "\n[5/9] ECG validation..."
    )

    ecg_summary = (
        ECGValidator().validate(
            sample=sample,
            output_dir=sample_dir,
        )
    )

    # --------------------------------------------------
    # EDA
    # --------------------------------------------------

    print(
        "\n[6/9] EDA validation..."
    )

    eda_summary = (
        EDAValidator().validate(
            sample=sample,
            output_dir=sample_dir,
        )
    )

    # --------------------------------------------------
    # EMG
    # --------------------------------------------------

    print(
        "\n[7/9] EMG validation..."
    )

    emg_summary = (
        EMGValidator().validate(
            sample=sample,
            output_dir=sample_dir,
        )
    )

    # --------------------------------------------------
    # Multimodal Alignment
    # --------------------------------------------------

    print(
        "\n[8/9] Multimodal alignment..."
    )

    MultimodalAlignment().generate(
        sample=sample,
        sampling_rates=dataset.sampling_rates,
        output_dir=sample_dir,
    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print(
        "\n[9/9] Generating report..."
    )

    report = (
        SignalReportGenerator()
    )

    report.generate(

        emotion_summary_file=
        sample_dir
        / "emotion_summary.json",

        ecg_summary_file=
        sample_dir
        / "ecg_summary.json",

        eda_summary_file=
        sample_dir
        / "eda_summary.json",

        emg_summary_file=
        sample_dir
        / "emg_summary.json",

        output_file=
        sample_dir
        / "validation_report.md",
    )

    print(
        "\nValidation Complete"
    )

    print(
        f"Output Directory:\n"
        f"{sample_dir}"
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()