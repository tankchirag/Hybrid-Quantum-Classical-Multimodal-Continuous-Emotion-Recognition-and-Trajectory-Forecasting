# tests/case/check_case_npz.py

from datasets.loaders.case_loader import (
    CaseDataset,
)

ds = CaseDataset(
    dataset_root=r"D:/Datasets/CASE/processed",
    split="train",
)

sample = ds[0]

print()
print("=" * 80)
print("TOP LEVEL KEYS")
print("=" * 80)

print(
    sample.keys()
)

print()

for key, value in sample.items():

    print("-" * 60)

    print(
        "KEY:",
        key
    )

    try:

        print(
            "SHAPE:",
            value.shape
        )

    except AttributeError:

        print(
            "TYPE:",
            type(value)
        )

        if isinstance(
            value,
            dict,
        ):
            print(
                "DICT KEYS:",
                value.keys()
            )