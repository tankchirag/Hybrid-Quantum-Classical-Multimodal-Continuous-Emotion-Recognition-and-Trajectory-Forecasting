from pathlib import Path

from datasets.preprocessing.case_metadata import (
    CaseMetadataManager,
)

from datasets.preprocessing.case_segment_extractor import (
    CaseSegmentExtractor,
)

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
        "CASE VIDEO SLICING TEST"
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
            1
        )
    )

    print(
        "\nVideo:",
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
        "\nWhole Subject Shape:"
    )

    print(
        subject.X.shape
    )

    print(
        "\nSegment Length:"
    )

    print(
        int(segment.duration_ms)
    )

    assert (
        segment.duration_ms
        > 0
    )

    print(
        "\n✅ Segment boundaries resolved"
    )

    print(
        "✅ Ready for true slicing implementation"
    )

    print(
        "\n✅ CASE VIDEO SLICING TEST PASSED"
    )


if __name__ == "__main__":
    main()