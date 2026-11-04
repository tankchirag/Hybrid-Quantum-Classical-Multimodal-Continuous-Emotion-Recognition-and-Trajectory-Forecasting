from pathlib import Path

from datasets.loaders.case_loader import (
    CaseDataset,
)

DATASET_ROOT = Path(
    r"D:\Datasets\CASE\processed"
)


def main():

    print()
    print("=" * 60)
    print(
        "CASE SPLIT LOADER TEST"
    )
    print("=" * 60)

    train_ds = CaseDataset(
        DATASET_ROOT,
        split="train",
    )

    val_ds = CaseDataset(
        DATASET_ROOT,
        split="validation",
    )

    test_ds = CaseDataset(
        DATASET_ROOT,
        split="test",
    )

    all_ds = CaseDataset(
        DATASET_ROOT,
        split="all",
    )

    print()

    print(
        "Train:",
        len(train_ds)
    )

    print(
        "Validation:",
        len(val_ds)
    )

    print(
        "Test:",
        len(test_ds)
    )

    print(
        "All:",
        len(all_ds)
    )

    assert len(train_ds) == 160
    assert len(val_ds) == 40
    assert len(test_ds) == 40
    assert len(all_ds) == 240

    print()
    print(
        "✅ CASE SPLIT LOADER TEST PASSED"
    )


if __name__ == "__main__":
    main()