from pathlib import Path
import csv

from datasets.manifests.export_case_manifests import (
    CaseManifestExporter,
)

DATASET_ROOT = Path(
    r"D:\Datasets\CASE\processed"
)


def count_rows(
    csv_path: Path,
) -> int:

    with open(
        csv_path,
        "r",
        encoding="utf-8",
    ) as fp:

        reader = csv.DictReader(fp)

        return sum(
            1 for _ in reader
        )


def main():

    print()
    print("=" * 60)
    print(
        "CASE MANIFEST EXPORT TEST"
    )
    print("=" * 60)

    exporter = (
        CaseManifestExporter(
            DATASET_ROOT
        )
    )

    exporter.export()

    manifest_dir = (
        DATASET_ROOT
        / "manifests"
    )

    train_manifest = (
        manifest_dir
        / "train_manifest.csv"
    )

    validation_manifest = (
        manifest_dir
        / "validation_manifest.csv"
    )

    test_manifest = (
        manifest_dir
        / "test_manifest.csv"
    )

    assert train_manifest.exists()
    assert validation_manifest.exists()
    assert test_manifest.exists()

    train_count = count_rows(
        train_manifest
    )

    validation_count = count_rows(
        validation_manifest
    )

    test_count = count_rows(
        test_manifest
    )

    print()
    print(
        f"Train Rows      : {train_count}"
    )

    print(
        f"Validation Rows : "
        f"{validation_count}"
    )

    print(
        f"Test Rows       : "
        f"{test_count}"
    )

    total = (
        train_count
        + validation_count
        + test_count
    )

    print(
        f"Total Rows      : {total}"
    )

    #
    # CASE split policy
    #
    # 20 train subjects
    # 5 validation subjects
    # 5 test subjects
    #
    # 8 included videos
    #

    assert train_count == 160, (
        f"Expected 160, "
        f"got {train_count}"
    )

    assert validation_count == 40, (
        f"Expected 40, "
        f"got {validation_count}"
    )

    assert test_count == 40, (
        f"Expected 40, "
        f"got {test_count}"
    )

    assert total == 240

    print()
    print(
        "✅ train manifest valid"
    )

    print(
        "✅ validation manifest valid"
    )

    print(
        "✅ test manifest valid"
    )

    print(
        "✅ CASE MANIFEST EXPORT TEST PASSED"
    )


if __name__ == "__main__":
    main()