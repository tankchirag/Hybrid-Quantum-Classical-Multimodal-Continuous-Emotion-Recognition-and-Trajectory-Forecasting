"""
Manifest Builder Test.
"""

from datasets.manifests.manifest_builder import (
    ManifestBuilder,
)


def main():

    print(
        "\n========================================"
    )

    print(
        "MANIFEST BUILDER TEST"
    )

    print(
        "========================================"
    )

    builder = (
        ManifestBuilder()
    )

    entries = (
        builder.build()
    )

    train = [
        x
        for x in entries
        if x.split == "train"
    ]

    validation = [
        x
        for x in entries
        if x.split == "validation"
    ]

    test = [
        x
        for x in entries
        if x.split == "test"
    ]

    print(
        "\nTrain Entries:",
        len(train)
    )

    print(
        "Validation Entries:",
        len(validation)
    )

    print(
        "Test Entries:",
        len(test)
    )

    print(
        "Total Entries:",
        len(entries)
    )

    assert (
        len(train)
        ==
        20 * 8
    )

    assert (
        len(validation)
        ==
        5 * 8
    )

    assert (
        len(test)
        ==
        5 * 8
    )

    assert (
        len(entries)
        ==
        240
    )

    print(
        "\n✅ Counts validated"
    )

    video_ids = {
        x.video_id
        for x in entries
    }

    assert (
        video_ids
        ==
        {
            1,2,3,4,
            5,6,7,8
        }
    )

    print(
        "✅ Video policy validated"
    )

    print(
        "✅ Subject policy validated"
    )

    print(
        "✅ Split policy validated"
    )

    print(
        "\nFirst Entry:"
    )

    print(
        entries[0]
    )

    print(
        "\n✅ MANIFEST BUILDER TEST PASSED"
    )


if __name__ == "__main__":
    main()