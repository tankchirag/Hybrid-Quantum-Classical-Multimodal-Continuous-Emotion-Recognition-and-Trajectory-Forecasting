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


def main():

    print(
        "\n========================================"
    )

    print(
        "CASE VIDEO FILTER TEST"
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

    pipeline = (
        CaseAuthorPipeline(
            metadata,
            CaseLabelGenerator(
                metadata
            ),
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

    interpolated = (
        interpolator.process(
            processed.physiology,
            processed.annotations,
        )
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

    normalized = (
        normalizer.process_subject(
            harmonized.dataframe
        )
    )

    scaled = (
        scaler.process_subject(
            normalized.dataframe
        )
    )

    df = (
        scaled.dataframe
    )

    video_id = 3

    filtered = (
        df[
            df["video"]
            ==
            video_id
        ]
    )

    print(
        "\nFull Shape:"
    )

    print(
        df.shape
    )

    print(
        "\nFiltered Shape:"
    )

    print(
        filtered.shape
    )

    print(
        "\nUnique Videos In Filter:"
    )

    print(
        filtered[
            "video"
        ].unique()
    )

    print(
        "\nRows:"
    )

    print(
        len(filtered)
    )

    assert (
        len(filtered)
        > 0
    )

    assert (
        set(
            filtered[
                "video"
            ].unique()
        )
        ==
        {3}
    )

    print(
        "\n✅ Video filtering works"
    )

    print(
        "✅ CASE VIDEO FILTER TEST PASSED"
    )


if __name__ == "__main__":
    main()