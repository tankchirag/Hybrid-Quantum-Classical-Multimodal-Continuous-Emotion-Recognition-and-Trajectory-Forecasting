"""
Dataset Builder Integration Test.
"""

from pathlib import Path

from datasets.processors.case_processor import (
    CaseProcessor,
)

from datasets.pipelines.dataset_builder import (
    DatasetBuilder,
)

from datasets.export.npz_exporter import (
    NpzExporter,
)


DATA_ROOT = Path(
    r"D:\Datasets\CASE\data\raw"
)

METADATA_ROOT = Path(
    r"D:\Datasets\CASE\metadata"
)

OUTPUT_ROOT = Path(
    "artifacts/dataset_builder_test"
)


def main():

    print(
        "\n========================================"
    )

    print(
        "DATASET BUILDER TEST"
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

    output_file = (
        builder.build_subject(
            1
        )
    )

    print(
        "\nOutput:"
    )

    print(
        output_file
    )

    assert (
        output_file.exists()
    )

    exporter = (
        NpzExporter()
    )

    assert (
        exporter.validate(
            output_file
        )
    )

    data = (
        exporter.load(
            output_file
        )
    )

    print(
        "\nX Shape:",
        data["X"].shape
    )

    print(
        "y Shape:",
        data["y"].shape
    )

    print(
        "\n✅ NPZ exists"
    )

    print(
        "✅ NPZ schema valid"
    )

    print(
        "✅ DatasetBuilder integration valid"
    )

    print(
        "\n✅ DATASET BUILDER TEST PASSED"
    )


if __name__ == "__main__":
    main()