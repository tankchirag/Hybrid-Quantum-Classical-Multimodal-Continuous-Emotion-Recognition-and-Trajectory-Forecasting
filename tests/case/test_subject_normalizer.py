"""
Subject Normalization Test.
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


DATA_ROOT = Path(
    r"D:\Datasets\CASE\data\raw"
)

METADATA_ROOT = Path(
    r"D:\Datasets\CASE\metadata"
)

SUBJECT_ID = 1


def main():

    print(
        "\n========================================"
    )

    print(
        "SUBJECT NORMALIZATION TEST"
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

    generator = (
        CaseLabelGenerator(
            metadata
        )
    )

    author = (
        CaseAuthorPipeline(
            metadata,
            generator,
        )
    )

    processed = (
        author.process_subject(
            physiology_file=
            DATA_ROOT
            / "physiological"
            / "sub1_DAQ.txt",

            annotation_file=
            DATA_ROOT
            / "annotations"
            / "sub1_joystick.txt",

            subject_id=
            SUBJECT_ID,
        )
    )

    interpolator = (
        CaseInterpolator()
    )

    interpolated = (
        interpolator.process(
            processed.physiology,
            processed.annotations,
        )
    )

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

    normalizer = (
        SubjectNormalizer()
    )

    result = (
        normalizer.process_subject(
            harmonized.dataframe
        )
    )

    df = result.dataframe

    columns = [

        "ecg",
        "bvp",
        "gsr",
        "rsp",
        "skt",

        "emg_zygo",
        "emg_coru",
        "emg_trap",
    ]

    for col in columns:

        mean = abs(
            df[col].mean()
        )

        std = abs(
            df[col].std() - 1.0
        )

        print(
            f"{col}: "
            f"mean={mean:.6f}, "
            f"std={df[col].std():.6f}"
        )

        assert mean < 0.01

        assert std < 0.01

    print(
        "\n✅ Means validated"
    )

    print(
        "✅ Standard deviations validated"
    )

    assert (
        df["subject_id"]
        .iloc[0]
        == SUBJECT_ID
    )

    print(
        "✅ Subject ID preserved"
    )

    print(
        "\n✅ SUBJECT NORMALIZATION TEST PASSED"
    )


if __name__ == "__main__":
    main()