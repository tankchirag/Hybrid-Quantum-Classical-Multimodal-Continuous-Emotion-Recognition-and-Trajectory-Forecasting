from pathlib import Path

from datasets.preprocessing.case_metadata import (
    CaseMetadataManager,
)

from datasets.preprocessing.case_label_generator import (
    CaseLabelGenerator,
)

from datasets.preprocessing.case_author_pipeline import (
    CaseAuthorPipeline,
)

from datasets.preprocessing.case_interpolator import (
    CaseInterpolator,
)


DATA_ROOT = Path(
    r"D:\Datasets\CASE\data\raw"
)

METADATA_ROOT = Path(
    r"D:\Datasets\CASE\metadata"
)


def main():

    metadata = (
        CaseMetadataManager(
            METADATA_ROOT
        )
    )

    metadata.load()

    generator = (
        CaseLabelGenerator(
            metadata
        )
    )

    pipeline = (
        CaseAuthorPipeline(
            metadata,
            generator,
        )
    )

    result = (
        pipeline.process_subject(
            physiology_file=
            DATA_ROOT
            / "physiological"
            / "sub1_DAQ.txt",

            annotation_file=
            DATA_ROOT
            / "annotations"
            / "sub1_joystick.txt",

            subject_id=1,
        )
    )

    interpolator = (
        CaseInterpolator()
    )

    interp = (
        interpolator.process(
            result.physiology,
            result.annotations,
        )
    )

    print(
        "Physiology:",
        interp.physiology.shape,
    )

    print(
        "Annotations:",
        interp.annotations.shape,
    )

    print(
        interp.physiology.head()
    )

    print(
        interp.annotations.head()
    )

    print(
        interp.physiology.columns
    )

    print(
        interp.annotations.columns
    )


if __name__ == "__main__":
    main()