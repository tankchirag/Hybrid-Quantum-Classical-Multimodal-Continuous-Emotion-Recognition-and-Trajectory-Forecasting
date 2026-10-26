from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


AUTHOR_SEQUENCE_DURATION_MS = 2451583.333


@dataclass
class InterpolationResult:
    physiology: pd.DataFrame
    annotations: pd.DataFrame


class CaseInterpolator:
    """
    Replicates the interpolation stage used
    in the CASE preprocessing scripts.

    Physiology:
        1 ms grid

    Annotations:
        50 ms grid

    Linear interpolation.
    """

    def __init__(
        self,
        sequence_duration_ms: float = (
            AUTHOR_SEQUENCE_DURATION_MS
        ),
    ) -> None:

        self.sequence_duration_ms = (
            sequence_duration_ms
        )

    # --------------------------------------------------
    # Target grids
    # --------------------------------------------------

    def create_daq_grid(
        self,
    ) -> np.ndarray:

        return np.arange(
            0,
            int(
                round(
                    self.sequence_duration_ms
                )
            )
            + 1,
            1,
            dtype=float,
        )

    def create_annotation_grid(
        self,
    ) -> np.ndarray:

        return np.arange(
            0,
            int(
                round(
                    self.sequence_duration_ms
                )
            )
            + 1,
            50,
            dtype=float,
        )

    # --------------------------------------------------
    # Core interpolation
    # --------------------------------------------------

    @staticmethod
    def interpolate_dataframe(
        df: pd.DataFrame,
        target_time: np.ndarray,
    ) -> pd.DataFrame:

        source_time = (
            df.iloc[:, 0]
            .to_numpy()
            .astype(float)
        )

        result = pd.DataFrame()

        result[0] = target_time

        for column_idx in range(
            1,
            len(df.columns),
        ):

            column = (
                df.iloc[:, column_idx]
                .to_numpy()
                .astype(float)
            )

            result[column_idx] = (
                np.interp(
                    target_time,
                    source_time,
                    column,
                )
            )

        return result

    # --------------------------------------------------
    # Physiology
    # --------------------------------------------------

    def interpolate_physiology(
        self,
        physiology: pd.DataFrame,
    ) -> pd.DataFrame:

        grid = (
            self.create_daq_grid()
        )

        return self.interpolate_dataframe(
            physiology,
            grid,
        )

    # --------------------------------------------------
    # Annotations
    # --------------------------------------------------

    def interpolate_annotations(
        self,
        annotations: pd.DataFrame,
    ) -> pd.DataFrame:

        grid = (
            self.create_annotation_grid()
        )

        return self.interpolate_dataframe(
            annotations,
            grid,
        )

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def process(
        self,
        physiology: pd.DataFrame,
        annotations: pd.DataFrame,
    ) -> InterpolationResult:

        physiology_interp = (
            self.interpolate_physiology(
                physiology
            )
        )

        annotations_interp = (
            self.interpolate_annotations(
                annotations
            )
        )

        return InterpolationResult(
            physiology=physiology_interp,
            annotations=annotations_interp,
        )