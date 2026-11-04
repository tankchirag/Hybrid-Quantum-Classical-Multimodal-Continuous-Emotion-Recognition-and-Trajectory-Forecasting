from pathlib import Path

import pandas as pd


METADATA_ROOT = Path(
    r"D:\Datasets\CASE\metadata"
)


FILES = [

    "participants.xlsx",

    "videos.xlsx",

    "seqs_order_num.xlsx",
]


def main():

    print(
        "\n========================================"
    )

    print(
        "CASE METADATA SOURCE TEST"
    )

    print(
        "========================================"
    )

    for file_name in FILES:

        path = (
            METADATA_ROOT
            / file_name
        )

        print(
            "\n" + "=" * 80
        )

        print(
            file_name
        )

        print(
            "=" * 80
        )

        df = pd.read_excel(
            path,
            engine="openpyxl",
        )

        print(
            "\nShape:"
        )

        print(
            df.shape
        )

        print(
            "\nColumns:"
        )

        print(
            list(
                df.columns
            )
        )

        print(
            "\nHead:"
        )

        print(
            df.head()
        )

    print(
        "\n✅ CASE METADATA SOURCE TEST PASSED"
    )


if __name__ == "__main__":
    main()