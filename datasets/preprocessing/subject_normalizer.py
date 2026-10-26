"""
Subject Normalization.

Purpose
-------
Apply subject-level normalization to physiological
channels after harmonization and before temporal
window generation.

Normalization is intentionally restricted to input
modalities.

Targets are never normalized here.

Controlled entirely through dataset YAML.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from configs.dataset_config import (
    DatasetConfig,
)


CASE_CONFIG = DatasetConfig.load(
    "configs/datasets/case.yaml"
)


@dataclass
class NormalizedSubject:
    """
    Normalized subject dataframe.
    """

    dataframe: pd.DataFrame


class SubjectNormalizer:

    def __init__(
        self,
        config: DatasetConfig = CASE_CONFIG,
    ) -> None:

        self.config = config

        self.enabled = (
            config.raw["normalization"]["enabled"]
        )

        self.strategy = (
            config.raw["normalization"]["strategy"]
        )

        self.epsilon = (
            config.raw["normalization"]["epsilon"]
        )

        self.columns = (
            config.raw["normalization"]["apply_to"]
        )

    def normalize(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        if not self.enabled:
            return dataframe.copy()

        if self.strategy != "zscore":
            raise ValueError(
                f"Unsupported strategy: "
                f"{self.strategy}"
            )

        df = dataframe.copy()

        for column in self.columns:

            mean = df[column].mean()

            std = df[column].std()

            df[column] = (
                df[column] - mean
            ) / (
                std + self.epsilon
            )

        return df

    def process_subject(
        self,
        dataframe: pd.DataFrame,
    ) -> NormalizedSubject:

        return NormalizedSubject(
            dataframe=self.normalize(
                dataframe
            )
        )