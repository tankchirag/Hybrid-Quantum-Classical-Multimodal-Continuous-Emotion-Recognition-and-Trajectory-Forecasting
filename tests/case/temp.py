from datasets.wrappers.case_training_dataset import (
    CaseTrainingDataset,
)


import sys
from pathlib import Path

DATASET_ROOT = Path(
    r"D:\Datasets\CASE\processed"
)

def main():

    ds = CaseTrainingDataset(
        DATASET_ROOT,
        "train",
    )

    sample = ds[0]

    for k, v in sample["modalities"].items():
        print(
            k,
            tuple(v.shape),
        )



    print()
    print(sample.keys())

    print()
    print("MODALITIES KEYS")
    print(
        sample["modalities"].keys()
    )

    for k, v in sample["modalities"].items():

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


if __name__ == "__main__":
    main()