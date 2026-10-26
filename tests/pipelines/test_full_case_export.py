"""
Full CASE Export Integration Test.

Purpose
-------
Validate multi-subject export using:

    CaseProcessor
        ↓
    DatasetBuilder
        ↓
    NPZ Export

This is the final integration step
before full CASE dataset generation.
"""

from pathlib import Path

from datasets.processors.case_processor import (
    CaseProcessor,
)

from datasets.pipelines.dataset_builder import (
    DatasetBuilder,
)


DATA_ROOT = Path(
    r"D:\Datasets\CASE\data\raw"
)

METADATA_ROOT = Path(
    r"D:\Datasets\CASE\metadata"
)

OUTPUT_ROOT = Path(
    "artifacts/full_case_export_test"
)


def main():

    print(
        "\n========================================"
    )

    print(
        "FULL CASE EXPORT TEST"
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

    builder = (
        DatasetBuilder(
            processor=
            processor,

            output_root=
            OUTPUT_ROOT,
        )
    )

    test_subjects = [
        1,
        2,
        3,
    ]

    exported_files = []

    for subject_id in test_subjects:

        output_file = (
            builder.build_subject(
                subject_id
            )
        )

        exported_files.append(
            output_file
        )

        print(
            f"✅ Subject "
            f"{subject_id} exported"
        )

    print(
        "\nExported Files:"
    )

    for file_path in exported_files:

        print(
            file_path
        )

        assert (
            file_path.exists()
        )

    assert (
        len(exported_files)
        == 3
    )

    print(
        "\n✅ Multi-subject export validated"
    )

    print(
        "✅ Output artifacts validated"
    )

    print(
        "\n✅ FULL CASE EXPORT TEST PASSED"
    )


if __name__ == "__main__":
    main()