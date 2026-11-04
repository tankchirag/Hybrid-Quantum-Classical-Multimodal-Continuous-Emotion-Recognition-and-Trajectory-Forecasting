# tests/case/test_raw_case_loading.py

from pathlib import Path

import pandas as pd
import numpy as np


DATA_ROOT = Path(
    r"D:\Datasets\CASE\data\raw"
)


def main() -> None:

    physiology_file = (
        DATA_ROOT
        / "physiological"
        / "sub1_DAQ.txt"
    )

    annotation_file = (
        DATA_ROOT
        / "annotations"
        / "sub1_joystick.txt"
    )

    print(
        f"Physiology file: "
        f"{physiology_file}"
    )

    print(
        f"Annotation file: "
        f"{annotation_file}"
    )

    assert physiology_file.exists()
    assert annotation_file.exists()

    physiology = pd.read_csv(
        physiology_file,
        sep="\t",
        header=None,
    )

    annotations = pd.read_csv(
        annotation_file,
        sep="\t",
        header=None,
    )

    print("\n=== PHYSIOLOGY ===")
    print(
        "Shape:",
        physiology.shape
    )

    print(
        "First rows:"
    )

    print(
        physiology.head()
    )

    print(
        "\nColumns:",
        physiology.shape[1]
    )

    print("\n=== ANNOTATIONS ===")

    print(
        "Shape:",
        annotations.shape
    )

    print(
        "First rows:"
    )

    print(
        annotations.head()
    )

    print(
        "\nColumns:",
        annotations.shape[1]
    )

    #
    # CASE expectations
    #

    assert (
        physiology.shape[1] == 9
    ), (
        "Expected 9 physiology columns "
        "(time + 8 channels)"
    )

    assert (
        annotations.shape[1] == 3
    ), (
        "Expected 3 annotation columns "
        "(time + valence + arousal)"
    )

    print(
        "\n✅ Raw CASE files loaded."
    )

    print(
        "✅ Physiology format valid."
    )

    print(
        "✅ Annotation format valid."
    )


if __name__ == "__main__":
    main()