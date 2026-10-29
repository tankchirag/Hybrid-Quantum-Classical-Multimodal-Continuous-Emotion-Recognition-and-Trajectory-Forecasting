import numpy as np

from datasets.wrappers.case_training_dataset import (
    CaseTrainingDataset,
)


def main():

    dataset = CaseTrainingDataset(
        dataset_root=
        r"D:\Datasets\CASE\processed",
        split="all",
    )

    print()
    print("=" * 80)
    print("DATASET STATS")
    print("=" * 80)

    print(
        "num files:",
        len(
            dataset.base_dataset
        )
    )

    print(
        "num windows:",
        len(
            dataset
        )
    )

    print()

    windows_per_file = []

    for file_idx in range(
        len(
            dataset.base_dataset
        )
    ):

        sample = (
            dataset.base_dataset[
                file_idx
            ]
        )

        num_windows = (
            sample["X"].shape[0]
        )

        windows_per_file.append(
            num_windows
        )

    windows_per_file = np.array(
        windows_per_file
    )

    print(
        "min windows/file:",
        windows_per_file.min()
    )

    print(
        "max windows/file:",
        windows_per_file.max()
    )

    print(
        "avg windows/file:",
        windows_per_file.mean()
    )

    print()

    print(
        "first 10 files:"
    )

    print(
        windows_per_file[:10]
    )

    print()

    sample = (
        dataset.base_dataset[
            0
        ]
    )

    print("=" * 80)
    print("FIRST NPZ")
    print("=" * 80)

    print(
        "X:",
        sample["X"].shape
    )

    print(
        "label_history:",
        sample[
            "label_history"
        ].shape
    )

    print(
        "y:",
        sample["y"].shape
    )


if __name__ == "__main__":
    main()