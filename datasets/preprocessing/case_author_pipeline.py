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


class CaseAuthorPipeline:
    """
    Partial replication of CASE s03_v2_interTransformData.m.

    Commit 3 scope:
      - load raw data
      - unit conversion
      - annotation conversion
      - label injection

    Interpolation is intentionally implemented later.
    """

    def __init__(
        self,
        metadata: CaseMetadataManager,
        label_generator: CaseLabelGenerator,
    ) -> None:
        self.metadata = metadata
        self.label_generator = label_generator

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

    @staticmethod
    def convert_time_to_ms(
        values,
    ):
        return values * 1000.0

    @staticmethod
    def convert_annotations(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        df[1] = (
            0.5
            + 9.0 * (df[1] + 26225)
            / 52450.0
        )

        df[2] = (
            0.5
            + 9.0 * (df[2] + 26225)
            / 52450.0
        )

        return df

    @staticmethod
    def convert_physiology(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        df[1] = ((df[1] - 2.8) / 50.0) * 1000.0
        df[2] = (58.962 * df[2]) - 115.09
        df[3] = (24.0 * df[3]) - 49.2
        df[4] = (58.923 * df[4]) - 115.01
        df[5] = (21.341 * df[5]) - 32.085

        df[6] = (
            ((df[6] - 2.0) / 4000.0)
            * 1_000_000
        )

        df[7] = (
            ((df[7] - 2.0) / 4000.0)
            * 1_000_000
        )

        df[8] = (
            ((df[8] - 2.0) / 4000.0)
            * 1_000_000
        )

        return df

    def interpolate_subject(self):
        raise NotImplementedError(
            "Implemented in next commit."
        )