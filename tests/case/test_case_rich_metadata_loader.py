from pathlib import Path

from datasets.preprocessing.case_metadata import (
    CaseMetadataManager,
)


METADATA_ROOT = Path(
    r"D:\Datasets\CASE\metadata"
)


def main():

    print(
        "\n========================================"
    )

    print(
        "CASE RICH METADATA LOADER TEST"
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

    print(
        "\nParticipant 1"
    )

    print(
        metadata.get_participant_info(
            1
        )
    )

    print(
        "\nParticipant 30"
    )

    print(
        metadata.get_participant_info(
            30
        )
    )

    print(
        "\nVideo 1"
    )

    print(
        metadata.get_video_info(
            1
        )
    )

    print(
        "\nVideo 7"
    )

    print(
        metadata.get_video_info(
            7
        )
    )

    print(
        "\nVideo 11"
    )

    print(
        metadata.get_video_info(
            11
        )
    )

    print(
        "\nSubject 1 Sequence"
    )

    print(
        metadata.get_subject_sequence(
            1
        )
    )

    print(
        "\nSubject 30 Sequence"
    )

    print(
        metadata.get_subject_sequence(
            30
        )
    )

    print(
        "\nSummary"
    )

    print(
        metadata.summary()
    )

    print(
        "\n✅ CASE RICH METADATA LOADER TEST PASSED"
    )


if __name__ == "__main__":
    main()