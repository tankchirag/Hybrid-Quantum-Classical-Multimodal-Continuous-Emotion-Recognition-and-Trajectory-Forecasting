from pathlib import Path

import numpy as np


def main():

    sample_dir = Path(
        r"D:/Datasets/CASE/processed/samples"
    )

    first_file = sorted(
        sample_dir.glob("*.npz")
    )[0]

    print()
    print(
        "FILE:",
        first_file.name
    )

    data = np.load(
        first_file,
        allow_pickle=True,
    )

    print()
    print(
        "=" * 80
    )
    print(
        "NPZ KEYS"
    )
    print(
        "=" * 80
    )

    print(
        data.files
    )

    print()

    if (
        "label_history"
        in data.files
    ):

        print(
            "label_history shape:",
            data[
                "label_history"
            ].shape
        )

    else:

        print(
            "❌ label_history NOT found"
        )

    print()

    print(
        "X shape:",
        data["X"].shape
    )

    print(
        "Y shape:",
        data["y"].shape
    )


if __name__ == "__main__":
    main()