"""
Tests for CASE Harmonization Pipeline.

Purpose
-------
Validate that interpolated CASE data is transformed
into the framework representation.

Validation
----------
1. Metadata loading
2. Label generation
3. Author preprocessing
4. Interpolation
5. Harmonization
6. Schema verification
7. Row-count verification
8. Video-label preservation
9. Subject-id preservation
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

from datasets.preprocessing.case_interpolator import (
    CaseInterpolator,
)

from datasets.preprocessing.case_harmonizer import (
    CaseHarmonizer,
)


# ==========================================================
# Configuration
# ==========================================================

DATA_ROOT = Path(
    r"D:\Datasets\CASE\data\raw"
)

METADATA_ROOT = Path(
    r"D:\Datasets\CASE\metadata"
)

SUBJECT_ID = 1


# ==========================================================
# Main Test
# ==========================================================

def main() -> None:
    """
    End-to-end validation of CASE harmonization.
    """

    print(
        "\n========================================"
    )

    print(
        "CASE HARMONIZATION TEST"
    )

    print(
        "========================================"
    )

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    metadata = (
        CaseMetadataManager(
            METADATA_ROOT
        )
    )

    metadata.load()

    print(
        "\n✅ Metadata loaded"
    )

    # ------------------------------------------------------
    # Labels
    # ------------------------------------------------------

    label_generator = (
        CaseLabelGenerator(
            metadata
        )
    )

    # ------------------------------------------------------
    # Author pipeline
    # ------------------------------------------------------

    author_pipeline = (
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
        author_pipeline.process_subject(
            physiology_file=physiology_file,
            annotation_file=annotation_file,
            subject_id=SUBJECT_ID,
        )
    )

    print(
        "✅ Author preprocessing done"
    )

    # ------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------

    interpolator = (
        CaseInterpolator()
    )

    interpolated = (
        interpolator.process(
            processed.physiology,
            processed.annotations,
        )
    )

    print(
        "✅ Interpolation done"
    )

    # ------------------------------------------------------
    # Harmonization
    # ------------------------------------------------------

    harmonizer = (
        CaseHarmonizer()
    )

    harmonized = (
        harmonizer.process_subject(
            physiology_df=
            interpolated.physiology,

            annotation_df=
            interpolated.annotations,

            subject_id=
            SUBJECT_ID,
        )
    )

    df = harmonized.dataframe

    print(
        "✅ Harmonization done"
    )

    # ------------------------------------------------------
    # Shape
    # ------------------------------------------------------

    print(
        "\nRows:",
        len(df)
    )

    print(
        "Columns:",
        len(df.columns)
    )

    assert len(df) > 4000
    assert len(df) < 5000

    # ------------------------------------------------------
    # Schema
    # ------------------------------------------------------

    expected_columns = [
        "time_ms",

        "ecg",
        "bvp",
        "gsr",
        "rsp",
        "skt",

        "emg_zygo",
        "emg_coru",
        "emg_trap",

        "video",

        "valence",
        "arousal",

        "subject_id",
    ]

    assert (
        list(df.columns)
        == expected_columns
    )

    print(
        "✅ Schema validated"
    )

    # ------------------------------------------------------
    # Subject validation
    # ------------------------------------------------------

    unique_subjects = (
        df["subject_id"]
        .unique()
        .tolist()
    )

    assert (
        unique_subjects
        == [SUBJECT_ID]
    )

    print(
        "✅ Subject IDs validated"
    )

    # ------------------------------------------------------
    # Video validation
    # ------------------------------------------------------

    expected_videos = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        10, 11, 12,
    }

    found_videos = set(
        df["video"]
        .unique()
        .tolist()
    )

    assert (
        found_videos
        == expected_videos
    )

    print(
        "✅ Video labels validated"
    )

    # ------------------------------------------------------
    # Targets
    # ------------------------------------------------------

    assert (
        df["valence"]
        .between(
            0.5,
            9.5
        )
        .all()
    )

    assert (
        df["arousal"]
        .between(
            0.5,
            9.5
        )
        .all()
    )

    print(
        "✅ Emotion ranges validated"
    )

    # ------------------------------------------------------
    # Preview
    # ------------------------------------------------------

    print(
        "\nPreview:"
    )

    print(
        df.head()
    )

    print(
        "\n✅ CASE harmonization test PASSED"
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()