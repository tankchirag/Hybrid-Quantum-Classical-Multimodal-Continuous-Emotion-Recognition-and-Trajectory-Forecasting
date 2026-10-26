"""
Temporal Contract Generator.

Purpose
-------
Transform a harmonized continuous emotion sequence
into supervised learning samples.

A temporal contract defines:

History Window
    ↓
Forecast Horizon

Example
-------
History:
    60 sec

Future:
    10 sec

2 Hz sampling

History length:
    120 samples

Forecast length:
    20 samples

Input
-----
Harmonized dataframe.

Output
------
X:
    [N, history_steps, feature_dim]

y:
    [N, forecast_steps, target_dim]

Ready for training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from configs.dataset_config import (
    DatasetConfig,
)


CASE_CONFIG = DatasetConfig.load(
    "configs/datasets/case.yaml"
)

# ==========================================================
# Result Containers
# ==========================================================

@dataclass
class TemporalContractResult:
    """
    Generated forecasting dataset.
    """

    X: np.ndarray
    y: np.ndarray

    feature_names: list[str]
    target_names: list[str]

    history_steps: int
    forecast_steps: int

    num_samples: int


# ==========================================================
# Generator
# ==========================================================

class TemporalContract:
    """
    Generate forecasting samples from
    harmonized CASE data.
    """

    def __init__(
        self,
        config: DatasetConfig = CASE_CONFIG,
    ) -> None:

        self.config = config

        self.sampling_rate_hz = (
            config.sampling_rate_hz
        )

        self.history_steps = (
            config.history_steps
        )

        self.forecast_steps = (
            config.forecast_steps
        )


    def get_feature_columns(
        self,
    ) -> list[str]:

        return (
            self.config
            .feature_columns
        )


    def get_target_columns(
        self,
    ) -> list[str]:

        return (
            self.config
            .target_columns
        )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    def validate_dataframe(
        self,
        df: pd.DataFrame,
    ) -> None:

        required = (
            self.get_feature_columns()
            +
            self.get_target_columns()
        )

        missing = [
            col
            for col in required
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing columns: {missing}"
            )

    # ------------------------------------------------------
    # Core Generation
    # ------------------------------------------------------

    def create_samples(
        self,
        dataframe: pd.DataFrame,
    ) -> TemporalContractResult:

        self.validate_dataframe(
            dataframe
        )

        feature_cols = (
            self.get_feature_columns()
        )

        target_cols = (
            self.get_target_columns()
        )

        features = (
            dataframe[
                feature_cols
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        targets = (
            dataframe[
                target_cols
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        X_list = []
        y_list = []

        max_start = (
            len(dataframe)
            - self.history_steps
            - self.forecast_steps
            + 1
        )

        for start_idx in range(
            max_start
        ):

            history_end = (
                start_idx
                +
                self.history_steps
            )

            forecast_end = (
                history_end
                +
                self.forecast_steps
            )

            X_window = (
                features[
                    start_idx:
                    history_end
                ]
            )

            y_window = (
                targets[
                    history_end:
                    forecast_end
                ]
            )

            X_list.append(
                X_window
            )

            y_list.append(
                y_window
            )

        X = np.asarray(
            X_list,
            dtype=np.float32,
        )

        y = np.asarray(
            y_list,
            dtype=np.float32,
        )

        return TemporalContractResult(
            X=X,
            y=y,

            feature_names=
            feature_cols,

            target_names=
            target_cols,

            history_steps=
            self.history_steps,

            forecast_steps=
            self.forecast_steps,

            num_samples=
            len(X),
        )

    # ------------------------------------------------------
    # Public API
    # ------------------------------------------------------

    def process_subject(
        self,
        dataframe: pd.DataFrame,
    ) -> TemporalContractResult:

        return self.create_samples(
            dataframe
        )
