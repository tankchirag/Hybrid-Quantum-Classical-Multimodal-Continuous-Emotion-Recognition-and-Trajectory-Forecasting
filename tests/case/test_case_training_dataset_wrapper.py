from pathlib import Path

from datasets.wrappers.case_training_dataset import (
    CaseTrainingDataset,
)

DATASET_ROOT = Path(
    r"D:\Datasets\CASE\processed"
)


def main():

    ds = CaseTrainingDataset(
        dataset_root=DATASET_ROOT,
        split="train",
    )

    print()
    print(
        "LEN:",
        len(ds)
    )

    sample = ds[0]

    print()
    print(
        "KEYS:"
    )

    print(
        sample.keys()
    )

    print()

    for k, v in sample.items():

        try:

            print(
                k,
                v.shape,
            )

        except Exception:

            print(
                k,
                type(v),
            )

    print()
    print(
        "✅ CASE TRAINING WRAPPER PASSED"
    )


if __name__ == "__main__":
    main()