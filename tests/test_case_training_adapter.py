from pathlib import Path

from datasets.loaders.case_loader import (
    CaseDataset,
)

from datasets.adapters.case_training_adapter import (
    CaseTrainingAdapter,
)

DATASET_ROOT = Path(
    r"D:\Datasets\CASE\processed"
)


def main():

    dataset = CaseDataset(
        DATASET_ROOT,
        split="train",
    )

    adapter = (
        CaseTrainingAdapter()
    )

    sample = dataset[0]

    adapted = adapter(
        sample
    )

    print()
    print("=" * 60)
    print(
        "CASE TRAINING ADAPTER"
    )
    print("=" * 60)

    print()

    print(
        adapted.keys()
    )

    print()

    print(
        adapted["modalities"][
            "physiology"
        ].shape
    )

    print()

    print(
        adapted[
            "recognition_targets"
        ][
            "continuous"
        ][
            "valence"
        ].shape
    )

    print()

    print(
        adapted[
            "forecast_targets"
        ][
            "continuous"
        ][
            "valence"
        ].shape
    )

    print()
    print(
        "✅ ADAPTER PASSED"
    )


if __name__ == "__main__":
    main()