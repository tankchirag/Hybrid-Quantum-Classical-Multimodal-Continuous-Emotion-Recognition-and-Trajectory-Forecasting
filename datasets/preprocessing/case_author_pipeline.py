from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from datasets.preprocessing.case_metadata import (
    CaseMetadataManager,
)

from datasets.preprocessing.case_label_generator import (
    CaseLabelGenerator,
)


@dataclass
class SubjectData:
    physiology: pd.DataFrame
    annotations: pd.DataFrame


@dataclass
class AuthorPipelineResult:
    physiology: pd.DataFrame
    annotations: pd.DataFrame


class CaseAuthorPipeline:
    """
    Commit 3

    - load raw files
    - convert time to ms
    - convert physiology units
    - convert joystick values
    - inject video labels
    - remove label=0 rows

    NO interpolation yet
    """

    def __init__(
        self,
        metadata: CaseMetadataManager,
        label_generator: CaseLabelGenerator,
    ) -> None:

        self.metadata = metadata
        self.label_generator = label_generator

    # --------------------------------------------------
    # Loading
    # --------------------------------------------------

    def load_raw_subject(
        self,
        physiology_path: Path,
        annotation_path: Path,
    ) -> SubjectData:

        physiology = pd.read_csv(
            physiology_path,
            sep="\t",
            header=None,
        )

        annotations = pd.read_csv(
            annotation_path,
            sep="\t",
            header=None,
        )

        return SubjectData(
            physiology=physiology,
            annotations=annotations,
        )

    # --------------------------------------------------
    # Physiology conversion
    # --------------------------------------------------

    @staticmethod
    def convert_physiology_units(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        # time seconds -> ms
        df[0] = df[0] * 1000.0

        # ECG
        df[1] = (
            (df[1] - 2.8)
            / 50.0
        ) * 1000.0

        # BVP
        df[2] = (
            58.962 * df[2]
        ) - 115.09

        # GSR
        df[3] = (
            24.0 * df[3]
        ) - 49.2

        # RESP
        df[4] = (
            58.923 * df[4]
        ) - 115.01

        # SKT
        df[5] = (
            21.341 * df[5]
        ) - 32.085

        # EMG-ZYGO
        df[6] = (
            (df[6] - 2.0)
            / 4000.0
        ) * 1_000_000

        # EMG-CORU
        df[7] = (
            (df[7] - 2.0)
            / 4000.0
        ) * 1_000_000

        # EMG-TRAP
        df[8] = (
            (df[8] - 2.0)
            / 4000.0
        ) * 1_000_000

        return df

    # --------------------------------------------------
    # Annotation conversion
    # --------------------------------------------------

    @staticmethod
    def convert_annotations(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        # time seconds -> ms
        df[0] = df[0] * 1000.0

        # valence
        df[1] = (
            0.5
            + 9.0
            * (df[1] + 26225)
            / 52450.0
        )

        # arousal
        df[2] = (
            0.5
            + 9.0
            * (df[2] + 26225)
            / 52450.0
        )

        return df

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------

    def inject_labels(
        self,
        df: pd.DataFrame,
        subject_id: int,
        time_column: int = 0,
    ) -> pd.DataFrame:

        result = self.label_generator.generate(
            time_vector=df[time_column].values,
            subject_id=subject_id,
        )

        df = df.copy()

        df["video"] = result.labels

        return df

    # --------------------------------------------------
    # Remove MATLAB-style zeros
    # --------------------------------------------------

    @staticmethod
    def remove_unassigned_rows(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if "video" not in df.columns:
            return df

        return (
            df[df["video"] != 0]
            .reset_index(drop=True)
        )

    # --------------------------------------------------
    # Full processing
    # --------------------------------------------------

    def process_subject(
        self,
        physiology_file: Path,
        annotation_file: Path,
        subject_id: int,
    ) -> AuthorPipelineResult:

        subject_data = self.load_raw_subject(
            physiology_file,
            annotation_file,
        )

        physiology = (
            subject_data.physiology
        )

        annotations = (
            subject_data.annotations
        )

        # convert first
        physiology = (
            self.convert_physiology_units(
                physiology
            )
        )

        annotations = (
            self.convert_annotations(
                annotations
            )
        )

        # then label
        physiology = self.inject_labels(
            physiology,
            subject_id,
        )

        annotations = self.inject_labels(
            annotations,
            subject_id,
        )

        # remove label==0
        physiology = (
            self.remove_unassigned_rows(
                physiology
            )
        )

        annotations = (
            self.remove_unassigned_rows(
                annotations
            )
        )

        return AuthorPipelineResult(
            physiology=physiology,
            annotations=annotations,
        )

    # --------------------------------------------------
    # Next commit
    # --------------------------------------------------

    def interpolate_subject(
        self,
    ):
        raise NotImplementedError(
            "Implemented in Commit 4."
        )