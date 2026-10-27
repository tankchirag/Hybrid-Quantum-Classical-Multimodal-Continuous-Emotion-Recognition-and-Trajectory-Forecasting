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
            DATASET_ROOT
        )
    )

    print(
        "\nLength:"
    )

    print(
        len(dataset)
    )

    sample = dataset[0]

    print(
        "\nX Shape:"
    )

    print(
        sample["X"].shape
    )

    print(
        "\ny Shape:"
    )

    print(
        sample["y"].shape
    )

    print(
        "\nMetadata:"
    )

    print(
        sample["metadata"]
    )

    print(
        "\n✅ CASE LOADER TEST PASSED"
    )


if __name__ == "__main__":
    main()