from pathlib import Path
import numpy as np

DATASET_ROOT = Path(
    r"D:\Datasets\CASE\processed\samples"
)


def main():

    first_file = sorted(
        DATASET_ROOT.glob("*.npz")
    )[0]

    print()
    print("=" * 60)
    print("CASE NPZ SCHEMA")
    print("=" * 60)

    print()
    print("File:")
    print(first_file.name)

    data = np.load(
        first_file,
        allow_pickle=True,
    )

    print()
    print("Keys:")

    for key in data.files:
        print(" -", key)

    print()
    print("✅ NPZ SCHEMA TEST PASSED")


if __name__ == "__main__":
    main()