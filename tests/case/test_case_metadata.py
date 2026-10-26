# tests/case/test_case_metadata.py

from pathlib import Path

from datasets.preprocessing.case_metadata import (
    CaseMetadataManager,
)


def main() -> None:

    manager = CaseMetadataManager(
        Path(
            r"D:\Datasets\CASE\metadata"
        )
    )

    manager.load()

    print(
        manager.summary()
    )

    print(
        manager.get_subject_duration_ms(
            1
        )
    )


if __name__ == "__main__":
    main()