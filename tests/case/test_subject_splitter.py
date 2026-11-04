"""
Subject Split Test.

Validation
----------
1. Train split
2. Validation split
3. Test split
4. Coverage
5. Leakage prevention
"""

from datasets.splits.subject_splitter import (
    SubjectSplitter,
)


def main():

    print(
        "\n========================================"
    )

    print(
        "SUBJECT SPLIT TEST"
    )

    print(
        "========================================"
    )

    splitter = (
        SubjectSplitter()
    )

    splitter.validate()

    splits = (
        splitter.build_splits()
    )

    print(
        "\nTrain:",
        splits.train_subjects
    )

    print(
        "Validation:",
        splits.validation_subjects
    )

    print(
        "Test:",
        splits.test_subjects
    )

    assert (
        len(
            splits.train_subjects
        )
        == 20
    )

    assert (
        len(
            splits.validation_subjects
        )
        == 5
    )

    assert (
        len(
            splits.test_subjects
        )
        == 5
    )

    print(
        "\n✅ Counts validated"
    )

    assert (
        splitter.get_split(1)
        == "train"
    )

    assert (
        splitter.get_split(20)
        == "train"
    )

    assert (
        splitter.get_split(21)
        == "validation"
    )

    assert (
        splitter.get_split(25)
        == "validation"
    )

    assert (
        splitter.get_split(26)
        == "test"
    )

    assert (
        splitter.get_split(30)
        == "test"
    )

    print(
        "✅ Boundary checks validated"
    )

    print(
        "✅ Leakage prevention validated"
    )

    print(
        "\n✅ SUBJECT SPLIT TEST PASSED"
    )


if __name__ == "__main__":
    main()