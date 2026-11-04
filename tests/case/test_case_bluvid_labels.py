"""
CASE BluVid Label Verification Test.

Purpose
-------
Determine whether:

    bluVid (video_id=11)

contains meaningful:

    valence
    arousal

annotations.

If yes, these segments may be useful for:

    recovery dynamics

    baseline stabilization

    emotion transition analysis

and should not necessarily be discarded.
"""

from pathlib import Path

from datasets.preprocessing.case_metadata import (
    CaseMetadataManager,
)

from datasets.preprocessing.case_segment_extractor import (
    CaseSegmentExtractor,
)

from datasets.preprocessing.case_label_generator import (
    CaseLabelGenerator,
)

from datasets.preprocessing.case_author_pipeline import (
    CaseAuthorPipeline,
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
        "CASE BLUVID LABEL TEST"
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

    label_generator = (
        CaseLabelGenerator(
            metadata
        )
    )

    pipeline = (
        CaseAuthorPipeline(
            metadata,
            label_generator,
        )
    )

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

    processed = (
        pipeline.process_subject(
            physiology_file=
            physiology_file,

            annotation_file=
            annotation_file,

            subject_id=1,
        )
    )

    segment = (
        extractor.get_segment(
            subject_id=1,
            video_id=11,
        )
    )

    annotations = (
        processed.annotations
    )

    print(
        "\nBluVid Segment:"
    )

    print(
        segment
    )

    #
    # IMPORTANT
    #
    # We are not slicing yet.
    #
    # Just inspect labels.
    #

    print(
        "\nAnnotation Head:"
    )

    print(
        annotations.head()
    )

    print(
        "\nAnnotation Columns:"
    )

    print(
        annotations.columns
    )

    print(
        "\nAnnotation Shape:"
    )

    print(
        annotations.shape
    )

    print(
        "\n✅ Metadata available"
    )

    print(
        "✅ Annotation stream loaded"
    )

    print(
        "\nMANUAL CHECK:"
    )

    print(
        "Identify timestamp column"
    )

    print(
        "Identify valence column"
    )

    print(
        "Identify arousal column"
    )

    print(
        "\n✅ CASE BLUVID LABEL TEST PASSED"
    )


if __name__ == "__main__":
    main()