"""
CASE Segment Extraction Test.

Purpose
-------
Compute temporal boundaries for each
video stimulus using:

    subject sequence
    +
    video durations

This validates that true video-level
segmentation is possible.
"""

from pathlib import Path

from datasets.preprocessing.case_metadata import (
    CaseMetadataManager,
)


METADATA_ROOT = Path(
    r"D:\Datasets\CASE\metadata"
)


def main():

    print(
        "\n========================================"
    )

    print(
        "CASE SEGMENT EXTRACTOR TEST"
    )

    print(
        "========================================"
    )

    metadata = (
        CaseMetadataManager(
            METADATA_ROOT
        )
    )

    metadata.load()

    sequence = (
        metadata.get_subject_sequence(
            1
        )
    )

    timeline_ms = 0

    print(
        "\nSubject 1 Timeline"
    )

    print(
        "\nVideo | Start(ms) | End(ms)"
    )

    print(
        "-" * 40
    )

    for video_id in sequence:

        video_info = (
            metadata.video_infos[
                video_id
            ]
        )

        start_ms = timeline_ms

        end_ms = (
            start_ms
            +
            video_info.duration_ms
        )

        print(
            f"{video_id:>5} | "
            f"{int(start_ms):>10} | "
            f"{int(end_ms):>8}"
        )

        timeline_ms = end_ms

    print(
        "\nTotal Recording Length:"
    )

    print(
        f"{timeline_ms/1000:.2f} sec"
    )

    assert (
        timeline_ms > 0
    )

    print(
        "\n✅ Timeline generated"
    )

    print(
        "✅ CASE SEGMENT EXTRACTOR TEST PASSED"
    )


if __name__ == "__main__":
    main()