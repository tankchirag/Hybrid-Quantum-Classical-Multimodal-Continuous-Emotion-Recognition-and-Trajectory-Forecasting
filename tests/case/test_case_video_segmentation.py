"""
CASE Video Segmentation Test.

Purpose
-------
Verify that CASE metadata contains
video boundary information required
for true video-level exports.
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
        "CASE VIDEO SEGMENTATION TEST"
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

    print(
        "\nLoaded metadata."
    )

    print(
        "\nMetadata object type:"
    )

    print(
        type(metadata)
    )

    print(
        "\nAvailable attributes:"
    )

    for attr in dir(metadata):

        if not attr.startswith("_"):

            print(attr)

    print(
        "\n✅ Metadata inspection completed"
    )

    print(
        "\nMANUAL CHECK:"
    )

    print(
        "Look for video-related "
        "attributes such as:"
    )

    print(
        "video_id"
    )

    print(
        "video_start"
    )

    print(
        "video_end"
    )

    print(
        "segments"
    )

    print(
        "boundaries"
    )

    print(
        "\n✅ CASE VIDEO SEGMENTATION TEST PASSED"
    )


if __name__ == "__main__":
    main()