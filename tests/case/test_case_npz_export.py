from pathlib import Path

import numpy as np

from datasets.processors.case_processor import (
    CaseProcessor,
)


DATA_ROOT = Path(
    r"D:\Datasets\CASE\data\raw"
)

METADATA_ROOT = Path(
    r"D:\Datasets\CASE\metadata"
)

OUTPUT_DIR = Path(
    r"D:\Datasets\CASE\temp"
)


def main():

    print(
        "\n========================================"
    )

    print(
        "CASE NPZ EXPORT TEST"
    )

    print(
        "========================================"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    processor = (
        CaseProcessor(
            data_root=
            DATA_ROOT,

            metadata_root=
            METADATA_ROOT,
        )
    )

    processed = (
        processor.process_video(
            subject_id=1,
            video_id=3,
        )
    )

    output_file = (
        OUTPUT_DIR
        / "sub_01_vid_03.npz"
    )

    np.savez_compressed(

        output_file,

        X=processed.X,

        y=processed.y,

        subject_id=
        processed.subject_id,

        video_id=
        processed.video_id,
    )

    print(
        "\nSaved:"
    )

    print(
        output_file
    )

    loaded = (
        np.load(
            output_file,
            allow_pickle=True,
        )
    )

    X = loaded["X"]

    y = loaded["y"]

    print(
        "\nReloaded Shapes"
    )

    print(
        "X:",
        X.shape
    )

    print(
        "y:",
        y.shape
    )

    print(
        "\nMetadata"
    )

    print(
        "subject_id:",
        int(
            loaded[
                "subject_id"
            ]
        )
    )

    print(
        "video_id:",
        int(
            loaded[
                "video_id"
            ]
        )
    )

    assert (
        X.shape
        ==
        processed.X.shape
    )

    assert (
        y.shape
        ==
        processed.y.shape
    )

    print(
        "\n✅ NPZ export works"
    )

    print(
        "✅ CASE NPZ EXPORT TEST PASSED"
    )


if __name__ == "__main__":
    main()