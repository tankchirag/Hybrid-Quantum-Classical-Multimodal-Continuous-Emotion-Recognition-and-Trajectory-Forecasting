"""
CASE Video Distribution Test.

Purpose
-------
Inspect annotation distribution
across all video IDs.
"""

from pathlib import Path

from datasets.preprocessing.case_metadata import (
    CaseMetadataManager,
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
        "CASE VIDEO DISTRIBUTION TEST"
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

    processed = (
        pipeline.process_subject(
            physiology_file=
            DATA_ROOT
            / "physiological"
            / "sub1_DAQ.txt",

            annotation_file=
            DATA_ROOT
            / "annotations"
            / "sub1_joystick.txt",

            subject_id=1,
        )
    )

    annotations = (
        processed.annotations
    )

    print(
        "\nVideo counts:"
    )

    counts = (
        annotations["video"]
        .value_counts()
        .sort_index()
    )

    print(counts)

    print(
        "\nUnique videos:"
    )

    print(
        sorted(
            annotations[
                "video"
            ].unique()
        )
    )

    print(
        "\nRows:"
    )

    print(
        len(
            annotations
        )
    )

    print(
        "\n✅ Distribution extracted"
    )

    print(
        "\n✅ CASE VIDEO DISTRIBUTION TEST PASSED"
    )


if __name__ == "__main__":
    main()