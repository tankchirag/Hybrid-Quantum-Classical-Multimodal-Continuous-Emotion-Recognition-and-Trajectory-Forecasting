from pathlib import Path

from datasets.processors.case_processor import (
    CaseProcessor,
)


DATA_ROOT = Path(
    r"D:\Datasets\CASE\data\raw"
)

METADATA_ROOT = Path(
    r"D:\Datasets\CASE\metadata"
)


def main():

    print(
        "\n========================================"
    )

    print(
        "CASE VIDEO PROCESSOR TEST"
    )

    print(
        "========================================"
    )

    processor = (
        CaseProcessor(
            data_root=
            DATA_ROOT,

            metadata_root=
            METADATA_ROOT,
        )
    )

    result = (
        processor.process_video(
            subject_id=1,

            video_id=1,
        )
    )

    print(
        "\nSubject:",
        result.subject_id
    )

    print(
        "Video:",
        result.video_id
    )

    print(
        "X Shape:",
        result.X.shape
    )

    print(
        "y Shape:",
        result.y.shape
    )

    assert (
        result.subject_id
        == 1
    )

    assert (
        result.video_id
        == 1
    )

    print(
        "\n✅ Video assignment validated"
    )

    print(
        "✅ CASE VIDEO PROCESSOR TEST PASSED"
    )


if __name__ == "__main__":
    main()