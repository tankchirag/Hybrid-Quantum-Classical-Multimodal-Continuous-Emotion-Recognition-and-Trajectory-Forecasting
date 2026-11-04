from pathlib import Path

import numpy as np

from datasets.preprocessing.case_metadata import (
    CaseMetadataManager,
)
from datasets.preprocessing.case_label_generator import (
    CaseLabelGenerator,
)


def main() -> None:

    metadata = (
        CaseMetadataManager(
            Path(
                r"D:\Datasets\CASE\metadata"
            )
        )
    )

    metadata.load()

    generator = (
        CaseLabelGenerator(
            metadata
        )
    )

    time_vector = np.arange(
        0,
        2451583.333 + 1,
        1,
    )

    result = (
        generator.generate(
            time_vector=time_vector,
            subject_id=1,
        )
    )

    print(
        "Rows:",
        len(result.labels)
    )

    print(
        "Trimmed:",
        result.trimmed_rows
    )

    print(
        "Unique labels:",
        np.unique(
            result.labels
        )
    )


    zero_positions = np.where(
        result.labels == 0
    )[0]

    print(
        "Zero positions:",
        zero_positions
    )

    print(
        "Last index:",
        len(result.labels) - 1
    )

    assert result.trimmed_rows <= 1

if __name__ == "__main__":
    main()