from pathlib import Path
import sys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]),
)

from datasets.loaders.case_loader import (
    CaseDataset,
)

DATASET_ROOT = Path(
    r"D:\Datasets\CASE\processed"
)


def main():

    ds = CaseDataset(
        DATASET_ROOT,
        split="train",
    )

    sample = ds[0]

    print()
    print("=" * 60)
    print("SAMPLE KEYS")
    print("=" * 60)

    for k in sample.keys():

        print(k)

    print()

    for k, v in sample.items():

        try:
            print(
                k,
                v.shape,
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()