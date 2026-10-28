# tests/test_case_training_dataset.py

from pathlib import Path
import sys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from datasets.wrappers.case_training_dataset import (  # type: ignore[reportMissingImports]
    CaseTrainingDataset,
)

DATASET_ROOT = Path(
    r"D:\Datasets\CASE\processed"
)


def main():

    dataset = (
        CaseTrainingDataset(
            dataset_root=
            DATASET_ROOT,

            split="train",
        )
    )

    print()
    print("=" * 60)
    print(
        "CASE TRAINING DATASET"
    )
    print("=" * 60)

    print()
    print(
        "Dataset Length:"
    )

    print(
        len(dataset)
    )

    sample = dataset[0]

    print()
    print(
        "Top-Level Keys:"
    )

    print(
        list(
            sample.keys()
        )
    )

    print()
    print(
        "Modalities:"
    )

    print(
        sample[
            "modalities"
        ].keys()
    )

    print()
    print(
        "Recognition Targets:"
    )

    print(
        sample[
            "recognition_targets"
        ][
            "continuous"
        ].keys()
    )

    print()
    print(
        "Forecast Targets:"
    )

    print(
        sample[
            "forecast_targets"
        ][
            "continuous"
        ].keys()
    )

    print()
    print(
        "Physiology Shape:"
    )

    print(
        sample[
            "modalities"
        ][
            "physiology"
        ].shape
    )

    print()
    print(
        "Valence Forecast:"
    )

    print(
        sample[
            "forecast_targets"
        ][
            "continuous"
        ][
            "valence"
        ].shape
    )

    assert (
        "modalities"
        in sample
    )

    assert (
        "recognition_targets"
        in sample
    )

    assert (
        "forecast_targets"
        in sample
    )

    print()
    print(
        "✅ CASE TRAINING DATASET PASSED"
    )


if __name__ == "__main__":
    main()