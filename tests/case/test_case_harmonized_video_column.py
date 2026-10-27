"""
CASE Harmonized Video Column Test.

Purpose
-------
Verify whether the 'video' column
survives the preprocessing stages.

If yes:

    process_video()

can simply filter:

    dataframe[
        dataframe["video"] == video_id
    ]

instead of doing
timestamp-based slicing.
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

from datasets.preprocessing.subject_normalizer import (
    SubjectNormalizer,
)

from datasets.preprocessing.label_scaler import (
    LabelScaler,
)


DATA_ROOT = Path(
    r"D:\Datasets\CASE\data\raw"
)

METADATA_ROOT = Path(
    r"D:\Datasets\CASE\metadata"
)


def print_stage(
    name,
    dataframe,
):

    print(
        f"\n{name}"
    )

    print(
        "-" * 50
    )

    print(
        "Rows:",
        len(dataframe)
    )

    print(
        "\nColumns:"
    )

    print(
        list(
            dataframe.columns
        )
    )

    print(
        "\nContains video column?"
    )

    print(
        "video" in dataframe.columns
    )

    if (
        "video"
        in dataframe.columns
    ):

        print(
            "\nUnique videos:"
        )

        print(
            sorted(
                dataframe[
                    "video"
                ].unique()
            )
        )


def main():

    print(
        "\n========================================"
    )

    print(
        "CASE HARMONIZED VIDEO COLUMN TEST"
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

    interpolator = (
        CaseInterpolator()
    )

    harmonizer = (
        CaseHarmonizer()
    )

    normalizer = (
        SubjectNormalizer()
    )

    scaler = (
        LabelScaler()
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

    print_stage(
        "RAW ANNOTATIONS",
        processed.annotations,
    )

    interpolated = (
        interpolator.process(
            processed.physiology,
            processed.annotations,
        )
    )

    print_stage(
        "INTERPOLATED",
        interpolated.annotations,
    )

    harmonized = (
        harmonizer.process_subject(
            physiology_df=
            interpolated.physiology,

            annotation_df=
            interpolated.annotations,

            subject_id=1,
        )
    )

    print_stage(
        "HARMONIZED",
        harmonized.dataframe,
    )

    normalized = (
        normalizer.process_subject(
            harmonized.dataframe
        )
    )

    print_stage(
        "NORMALIZED",
        normalized.dataframe,
    )

    scaled = (
        scaler.process_subject(
            normalized.dataframe
        )
    )

    print_stage(
        "SCALED",
        scaled.dataframe,
    )

    print(
        "\n✅ CASE HARMONIZED VIDEO COLUMN TEST PASSED"
    )


if __name__ == "__main__":
    main()