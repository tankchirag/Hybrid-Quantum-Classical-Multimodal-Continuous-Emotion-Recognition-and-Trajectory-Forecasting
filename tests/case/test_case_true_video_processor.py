"""
CASE True Video Processor Test.

Purpose
-------
Verify that:

    process_video()

returns only a single video segment
rather than the entire subject
recording.
"""

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
        "CASE TRUE VIDEO PROCESSOR TEST"
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

    subject = (
        processor.process_subject(
            subject_id=1
        )
    )

    video = (
        processor.process_video(
            subject_id=1,
            video_id=3,
        )
    )

    print(
        "\nSubject Shape:"
    )

    print(
        subject.X.shape
    )

    print(
        "\nVideo Shape:"
    )

    print(
        video.X.shape
    )

    print(
        "\nSubject Labels:"
    )

    print(
        subject.y.shape
    )

    print(
        "\nVideo Labels:"
    )

    print(
        video.y.shape
    )

    assert (
        video.X.shape[0]
        <
        subject.X.shape[0]
    )

    assert (
        video.y.shape[0]
        <
        subject.y.shape[0]
    )

    print(
        "\nCompression Ratio:"
    )

    print(
        f"{video.X.shape[0] / subject.X.shape[0]:.4f}"
    )

    print(
        "\n✅ process_video performs real filtering"
    )

    print(
        "✅ CASE TRUE VIDEO PROCESSOR TEST PASSED"
    )


if __name__ == "__main__":
    main()