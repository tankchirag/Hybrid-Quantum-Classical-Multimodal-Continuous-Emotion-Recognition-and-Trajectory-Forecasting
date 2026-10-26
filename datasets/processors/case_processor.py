from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datasets.processors.base_processor import (
    BaseDatasetProcessor,
)

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

from datasets.preprocessing.case_harmonizer import (
    CaseHarmonizer,
)

from datasets.preprocessing.subject_normalizer import (
    SubjectNormalizer,
)

from datasets.preprocessing.label_scaler import (
    LabelScaler,
)

from datasets.preprocessing.temporal_contract import (
    TemporalContract,
)


@dataclass
class ProcessedSubject:

    X: np.ndarray

    y: np.ndarray

    feature_names: list[str]

    target_names: list[str]

    subject_id: int


class CaseProcessor(
    BaseDatasetProcessor
):

    def __init__(
        self,
        data_root: Path,
        metadata_root: Path,
    ):

        self.data_root = Path(
            data_root
        )

        self.metadata_root = Path(
            metadata_root
        )

    def process_subject(
        self,
        subject_id: int,
    ) -> ProcessedSubject:

        metadata = (
            CaseMetadataManager(
                self.metadata_root
            )
        )

        metadata.load()

        label_generator = (
            CaseLabelGenerator(
                metadata
            )
        )

        author_pipeline = (
            CaseAuthorPipeline(
                metadata,
                label_generator,
            )
        )

        interpolator = (
            CaseInterpolator()
        )

        harmonizer = (
            CaseHarmonizer()
        )

        normalizer = (
            SubjectNormalizer()
        )

        scaler = (
            LabelScaler()
        )

        temporal = (
            TemporalContract()
        )

        physiology_file = (
            self.data_root
            / "physiological"
            / f"sub{subject_id}_DAQ.txt"
        )

        annotation_file = (
            self.data_root
            / "annotations"
            / f"sub{subject_id}_joystick.txt"
        )

        processed = (
            author_pipeline.process_subject(
                physiology_file=
                physiology_file,

                annotation_file=
                annotation_file,

                subject_id=
                subject_id,
            )
        )

        interpolated = (
            interpolator.process(
                processed.physiology,
                processed.annotations,
            )
        )

        harmonized = (
            harmonizer.process_subject(
                physiology_df=
                interpolated.physiology,

                annotation_df=
                interpolated.annotations,

                subject_id=
                subject_id,
            )
        )

        normalized = (
            normalizer.process_subject(
                harmonized.dataframe
            )
        )

        scaled = (
            scaler.process_subject(
                normalized.dataframe
            )
        )

        contract = (
            temporal.process_subject(
                scaled.dataframe
            )
        )

        return ProcessedSubject(

            X=contract.X,

            y=contract.y,

            feature_names=
            contract.feature_names,

            target_names=
            contract.target_names,

            subject_id=
            subject_id,
        )