"""
Label Scaling Test.
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

SUBJECT_ID = 1


def main():

    print(
        "\n========================================"
    )

    print(
        "LABEL SCALING TEST"
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

    interpolated = (
        CaseInterpolator()
        .process(
            processed.physiology,
            processed.annotations,
        )
    )

    harmonized = (
        CaseHarmonizer()
        .process_subject(
            physiology_df=
            interpolated.physiology,

            annotation_df=
            interpolated.annotations,

            subject_id=
            SUBJECT_ID,
        )
    )

    normalized = (
        SubjectNormalizer()
        .process_subject(
            harmonized.dataframe
        )
    )

    scaled = (
        LabelScaler()
        .process_subject(
            normalized.dataframe
        )
    )

    df = scaled.dataframe

    print(
        "Valence Range:",
        df["valence"].min(),
        df["valence"].max(),
    )

    print(
        "Arousal Range:",
        df["arousal"].min(),
        df["arousal"].max(),
    )

    assert (
        df["valence"]
        .between(
            -1.0,
            1.0
        )
        .all()
    )

    assert (
        df["arousal"]
        .between(
            -1.0,
            1.0
        )
        .all()
    )

    print(
        "\n✅ Valence scaled"
    )

    print(
        "✅ Arousal scaled"
    )

    print(
        "\n✅ LABEL SCALING TEST PASSED"
    )


if __name__ == "__main__":
    main()