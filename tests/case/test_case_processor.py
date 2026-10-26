"""
CASE Processor Contract Test.

Purpose
-------
Validate the dataset processor interface.

This test defines the expected
processor output contract.

The implementation will be added
in the next commit.
"""

from pathlib import Path

from datasets.processors.case_processor import (
    CaseProcessor,
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
        "CASE PROCESSOR TEST"
    )

    print(
        "========================================"
    )

    processor = (
        CaseProcessor(
            data_root=
            DATA_ROOT,

            metadata_root=
            METADATA_ROOT,
        )
    )

    result = (
        processor.process_subject(
            subject_id=1
        )
    )

    print(
        "\nSubject:",
        result.subject_id
    )

    print(
        "X Shape:",
        result.X.shape
    )

    print(
        "y Shape:",
        result.y.shape
    )

    print(
        "Feature Count:",
        len(
            result.feature_names
        )
    )

    print(
        "Target Count:",
        len(
            result.target_names
        )
    )

    assert (
        result.subject_id
        == 1
    )

    assert (
        result.X is not None
    )

    assert (
        result.y is not None
    )

    assert (
        len(
            result.feature_names
        )
        > 0
    )

    assert (
        len(
            result.target_names
        )
        > 0
    )

    print(
        "\n✅ Subject id validated"
    )

    print(
        "✅ Features validated"
    )

    print(
        "✅ Targets validated"
    )

    print(
        "✅ Tensor generation validated"
    )

    print(
        "\n✅ CASE PROCESSOR TEST PASSED"
    )


if __name__ == "__main__":
    main()