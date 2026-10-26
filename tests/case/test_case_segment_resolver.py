from pathlib import Path

from datasets.preprocessing.case_metadata import (
    CaseMetadataManager,
)

from datasets.preprocessing.case_segment_extractor import (
    CaseSegmentExtractor,
)


METADATA_ROOT = Path(
    r"D:\Datasets\CASE\metadata"
)


def main():

    print(
        "\n========================================"
    )

    print(
        "CASE SEGMENT RESOLVER TEST"
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

    extractor = (
        CaseSegmentExtractor(
            metadata
        )
    )

    segment = (
        extractor.get_segment(
            subject_id=1,
            video_id=3,
        )
    )

    print(
        "\nVideo:",
        segment.video_id
    )

    print(
        "Name:",
        segment.video_name
    )

    print(
        "Start:",
        int(segment.start_ms)
    )

    print(
        "End:",
        int(segment.end_ms)
    )

    print(
        "Duration:",
        int(segment.duration_ms)
    )

    assert (
        segment.start_ms
        >= 0
    )

    assert (
        segment.end_ms
        >
        segment.start_ms
    )

    print(
        "\n✅ Segment resolved"
    )

    print(
        "✅ CASE SEGMENT RESOLVER TEST PASSED"
    )


if __name__ == "__main__":
    main()