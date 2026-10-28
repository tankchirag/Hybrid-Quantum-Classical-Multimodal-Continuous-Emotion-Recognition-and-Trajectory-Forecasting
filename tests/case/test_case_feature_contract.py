from pathlib import Path

from datasets.loaders.case_loader import (
    CaseDataset,
)

DATASET_ROOT = Path(
    r"D:\Datasets\CASE\processed"
)


def main():

    dataset = (
        CaseDataset(
            DATASET_ROOT,
            split="train",
        )
    )

    sample = dataset[0]

    print()
    print(
        "Feature Names:"
    )

    print(
        sample[
            "feature_names"
        ]
    )

    print()
    print(
        "Target Names:"
    )

    print(
        sample[
            "target_names"
        ]
    )

    assert len(
        sample[
            "feature_names"
        ]
    ) == 8

    assert len(
        sample[
            "target_names"
        ]
    ) == 2

    print()
    print(
        "✅ FEATURE CONTRACT PASSED"
    )


if __name__ == "__main__":
    main()