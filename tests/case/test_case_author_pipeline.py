# tests/case/test_case_author_pipeline.py

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


def main() -> None:

    metadata = CaseMetadataManager(
        METADATA_ROOT,
    )

    metadata.load()

    label_generator = (
        CaseLabelGenerator(
            metadata,
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

    result = pipeline.process_subject(
        physiology_file=physiology_file,
        annotation_file=annotation_file,
        subject_id=1,
    )

    print("\n=== PHYSIOLOGY ===")

    print(
        "Shape:",
        result.physiology.shape,
    )

    print(
        result.physiology.head(),
    )

    print("\n=== ANNOTATIONS ===")

    print(
        "Shape:",
        result.annotations.shape,
    )

    print(
        result.annotations.head(),
    )

    print("\n=== LABEL VALIDATION ===")

    physiology_labels = set(
        result.physiology["video"].unique()
    )

    annotation_labels = set(
        result.annotations["video"].unique()
    )

    print(
        "Physio labels:",
        sorted(physiology_labels),
    )

    print(
        "Annotation labels:",
        sorted(annotation_labels),
    )

    assert (
        0 not in physiology_labels
    ), "Zero labels remain in physiology"

    assert (
        0 not in annotation_labels
    ), "Zero labels remain in annotations"

    expected_labels = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        10, 11, 12,
    }

    assert (
        physiology_labels
        == expected_labels
    )

    assert (
        annotation_labels
        == expected_labels
    )

    print(
        "\n✅ Physiology conversion successful"
    )

    print(
        "✅ Annotation conversion successful"
    )

    print(
        "✅ Video label injection successful"
    )

    print(
        "✅ Zero-label trimming successful"
    )


if __name__ == "__main__":
    main()