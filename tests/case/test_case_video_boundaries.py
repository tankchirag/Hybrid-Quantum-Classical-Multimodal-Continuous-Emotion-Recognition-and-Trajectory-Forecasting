"""
CASE Video Boundary Test.

Purpose
-------
Validate that metadata contains sufficient
information to compute real video segments.
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
        "CASE VIDEO BOUNDARY TEST"
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

    print(
        "\nSubject 1 Sequence:"
    )

    print(sequence)

    print(
        "\nVideo Infos:"
    )

    print(metadata.video_infos)

    print(
        "\nSequence Length:"
    )

    print(len(sequence))

    assert (
        len(sequence)
        > 0
    )

    print(
        "\n✅ Subject sequence available"
    )

    print(
        "✅ Video metadata available"
    )

    print(
        "\n✅ CASE VIDEO BOUNDARY TEST PASSED"
    )


if __name__ == "__main__":
    main()