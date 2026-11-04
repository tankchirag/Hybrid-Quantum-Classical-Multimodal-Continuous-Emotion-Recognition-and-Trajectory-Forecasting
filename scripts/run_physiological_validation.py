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

import yaml

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent.parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from datasets.loaders.ulm_tsst_disk_dataset import (
    ULMTSSTDiskDataset,
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

from datasets.evaluation.physiological_timeline import (
    PhysiologicalTimelinePlotter,
)

# ==========================================================
# Configuration
# ==========================================================

# ==========================================================
# Configuration
# ==========================================================

DATASET_ROOT = Path(
    "data/generated/ulm_tsst_v1"
)

CONFIG_FILE = Path(
    "configs/datasets/ulm_tsst_v1.yaml"
)

DATASET_SPLIT = "raw"

OUTPUT_ROOT = Path(
    "results/physiological_validation"
)

SAMPLE_INDEX = 0 # Index of the sample to validate


# ==========================================================
# Load Configuration
# ==========================================================

def load_config():
    """
    Load official dataset configuration.

    This guarantees that validation
    uses the same modality and
    sampling-rate configuration
    that was used during dataset
    generation.
    """

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8",
    ) as fp:

        return yaml.safe_load(fp)


# ==========================================================
# Load Dataset 
# ==========================================================

def load_dataset():
    """
    Load released dataset artifact.
    """

    return ULMTSSTDiskDataset(

        dataset_root=
        DATASET_ROOT,

        split=
        DATASET_SPLIT,

        device="cpu",
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print(
        "\n[1/10] Loading released dataset..."
    )

    dataset = load_dataset()

    config = load_config()

    print(
        f"Dataset Root : "
        f"{DATASET_ROOT}"
    )

    print(
        f"Dataset Split : "
        f"{DATASET_SPLIT}"
    )

    print(
        f"Dataset Size : "
        f"{len(dataset)} samples"
    )

    print(
        f"Sample Index  : "
        f"{SAMPLE_INDEX}"
    )

    sampling_rates = (
    config["sampling_rates"]
    )

    sample = dataset[
        SAMPLE_INDEX
    ]

    print(
        f"Clip File     : "
        f"{sample['metadata']['file_name']}"
    )

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
        "\n[2/10] Creating summary..."
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
        "\n[3/10] Creating emotion timeline..."
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
    # Physiological Timeline
    # --------------------------------------------------


    print(
        "\n[4/10] Creating physiological timeline..."
    )

    PhysiologicalTimelinePlotter().generate(
        sample=sample,
        output_directory=sample_dir,
    )

    # --------------------------------------------------
    # Scenario Validation
    # --------------------------------------------------

    print(
        "\n[5/10] Scenario validation..."
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
        "\n[6/10] ECG validation..."
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
        "\n[7/10] EDA validation..."
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
        "\n[8/10] EMG validation..."
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
        "\n[9/10] Multimodal alignment..."
    )

    MultimodalAlignment().generate(
        sample=sample,
        sampling_rates=sampling_rates,
        output_dir=sample_dir,
    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print(
        "\n[10/10] Generating report..."
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