"""
CASE Label Variance Analysis.

Purpose
-------
Determine whether:

    startVid (10)
    bluVid  (11)
    endVid  (12)

contain meaningful valence/arousal dynamics.

Compare against:

    amusing-1 (1)
    boring-1 (3)
    scary-1 (7)
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


VIDEOS_TO_ANALYZE = [
    10,   # startVid
    11,   # bluVid
    12,   # endVid

    1,    # amusing-1
    3,    # boring-1
    7,    # scary-1
]


def main():

    print(
        "\n========================================"
    )

    print(
        "CASE LABEL VARIANCE TEST"
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

    #
    # Based on previous discovery:
    #
    # column 0 -> timestamp
    # column 1 -> valence
    # column 2 -> arousal
    #

    TIME_COL = 0
    VALENCE_COL = 1
    AROUSAL_COL = 2

    print(
        "\nVideo Statistics"
    )

    print(
        "\nVideo | Rows | "
        "Val Mean | Val Std | "
        "Aro Mean | Aro Std"
    )

    print(
        "-" * 75
    )

    for video_id in VIDEOS_TO_ANALYZE:

        subset = (
            annotations[
                annotations[
                    "video"
                ]
                ==
                video_id
            ]
        )

        val_mean = (
            subset[
                VALENCE_COL
            ].mean()
        )

        val_std = (
            subset[
                VALENCE_COL
            ].std()
        )

        aro_mean = (
            subset[
                AROUSAL_COL
            ].mean()
        )

        aro_std = (
            subset[
                AROUSAL_COL
            ].std()
        )

        print(
            f"{video_id:>5} | "
            f"{len(subset):>5} | "
            f"{val_mean:>8.3f} | "
            f"{val_std:>7.3f} | "
            f"{aro_mean:>8.3f} | "
            f"{aro_std:>7.3f}"
        )

    print(
        "\nInterpretation:"
    )

    print(
        "Very low std => labels mostly constant"
    )

    print(
        "Higher std => active annotation"
    )

    print(
        "\n✅ CASE LABEL VARIANCE TEST PASSED"
    )


if __name__ == "__main__":
    main()