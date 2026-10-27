from pathlib import Path

from datasets.loaders.case_loader import (
    CaseDataset,
)

DATASET_ROOT = Path(
    r"D:\Datasets\CASE\processed"
)


def collect_subjects(
    dataset,
):

    subjects = set()

    for idx in range(
        len(dataset)
    ):

        metadata = (
            dataset.get_metadata(
                idx
            )
        )

        subjects.add(
            metadata[
                "subject_id"
            ]
        )

    return subjects


def main():

    print()
    print("=" * 60)
    print(
        "CASE SPLIT INTEGRITY TEST"
    )
    print("=" * 60)

    train_dataset = (
        CaseDataset(
            DATASET_ROOT,
            split="train",
        )
    )

    validation_dataset = (
        CaseDataset(
            DATASET_ROOT,
            split="validation",
        )
    )

    test_dataset = (
        CaseDataset(
            DATASET_ROOT,
            split="test",
        )
    )

    train_subjects = (
        collect_subjects(
            train_dataset
        )
    )

    validation_subjects = (
        collect_subjects(
            validation_dataset
        )
    )

    test_subjects = (
        collect_subjects(
            test_dataset
        )
    )

    print()

    print(
        "Train Subjects:"
    )
    print(
        sorted(
            train_subjects
        )
    )

    print()

    print(
        "Validation Subjects:"
    )
    print(
        sorted(
            validation_subjects
        )
    )

    print()

    print(
        "Test Subjects:"
    )
    print(
        sorted(
            test_subjects
        )
    )

    assert (
        train_subjects
        .isdisjoint(
            validation_subjects
        )
    )

    assert (
        train_subjects
        .isdisjoint(
            test_subjects
        )
    )

    assert (
        validation_subjects
        .isdisjoint(
            test_subjects
        )
    )

    assert (
        len(
            train_subjects
        )
        == 20
    )

    assert (
        len(
            validation_subjects
        )
        == 5
    )

    assert (
        len(
            test_subjects
        )
        == 5
    )

    print()
    print(
        "✅ No subject leakage"
    )

    print(
        "✅ CASE SPLIT INTEGRITY TEST PASSED"
    )


if __name__ == "__main__":
    main()