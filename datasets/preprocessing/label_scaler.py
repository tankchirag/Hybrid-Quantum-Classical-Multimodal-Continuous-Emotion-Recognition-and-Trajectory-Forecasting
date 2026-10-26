"""
Continuous Emotion Label Scaling.

Purpose
-------
Scale continuous affect labels from the
original CASE range to the training range
used by the framework.

Default:

    [0.5, 9.5]
        ↓
    [-1.0, 1.0]

Configured entirely through YAML.
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
class ScaledLabels:
    dataframe: pd.DataFrame


class LabelScaler:

    def __init__(
        self,
        config: DatasetConfig = CASE_CONFIG,
    ) -> None:

        self.config = config

        settings = (
            config.raw[
                "label_processing"
            ]
        )

        self.enabled = (
            settings["scale_labels"]
        )

        self.source_min = (
            settings["source_range"]["min"]
        )

        self.source_max = (
            settings["source_range"]["max"]
        )

        self.target_min = (
            settings["target_range"]["min"]
        )

        self.target_max = (
            settings["target_range"]["max"]
        )

        self.columns = (
            config.target_columns
        )

    def scale_value(
        self,
        values,
    ):

        normalized = (
            values -
            self.source_min
        ) / (
            self.source_max -
            self.source_min
        )

        return (
            normalized
            *
            (
                self.target_max
                -
                self.target_min
            )
        ) + self.target_min

    def scale_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        if not self.enabled:
            return dataframe.copy()

        df = dataframe.copy()

        for column in self.columns:

            df[column] = (
                self.scale_value(
                    df[column]
                )
            )

        return df

    def process_subject(
        self,
        dataframe: pd.DataFrame,
    ) -> ScaledLabels:

        return ScaledLabels(
            dataframe=
            self.scale_dataframe(
                dataframe
            )
        )