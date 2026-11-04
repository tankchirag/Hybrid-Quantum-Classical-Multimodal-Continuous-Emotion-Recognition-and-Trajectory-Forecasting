"""
CASE Harmonization Pipeline.

Purpose
-------
Transform CASE interpolated data into the unified framework format
used by the thesis.

Input
-----
Physiology:
    1 ms interpolated signals

Annotations:
    50 ms interpolated emotion annotations

Output
------
Single synchronized dataframe at 2 Hz.

Target Schema
-------------
time_ms

ecg
bvp
gsr
rsp
skt
emg_zygo
emg_coru
emg_trap

valence
arousal

video

subject_id
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from configs.dataset_config import DatasetConfig

CASE_CONFIG = DatasetConfig.load(
    "configs/datasets/case.yaml"
)

# ==========================================================
# Result
# ==========================================================

@dataclass
class HarmonizedSubject:
    """
    Final harmonized subject representation.
    """

    dataframe: pd.DataFrame

    @property
    def num_samples(self) -> int:
        return len(self.dataframe)


# ==========================================================
# Harmonizer
# ==========================================================

class CaseHarmonizer:
    """
    Harmonize CASE data to framework sampling rate.

    Workflow
    --------
    1. Resample physiology
    2. Resample annotations
    3. Merge modalities
    4. Preserve video labels
    """

    def __init__(
        self,
        config: DatasetConfig = CASE_CONFIG,
    ) -> None:

        self.config = config

        self.target_period_ms = (
            config.harmonization.target_period_ms
        )

        self.target_sampling_rate_hz = (
            config.harmonization.target_sampling_rate_hz
        )


    # ------------------------------------------------------
    # Grid
    # ------------------------------------------------------

    def build_target_grid(
        self,
        end_time_ms: float,
    ) -> np.ndarray:

        return np.arange(
            0,
            int(round(end_time_ms)) + 1,
            self.target_period_ms,
            dtype=float,
        )

    # ------------------------------------------------------
    # Physiology
    # ------------------------------------------------------

    def resample_physiology(
        self,
        physiology_df: pd.DataFrame,
    ) -> pd.DataFrame:



        source_time = (
            physiology_df.iloc[:, 0]
            .to_numpy()
            .astype(float)
        )

        target_time = self.build_target_grid(
            source_time[-1]
        )

        result = pd.DataFrame()

        result["time_ms"] = target_time

        names = [
            "ecg",
            "bvp",
            "gsr",
            "rsp",
            "skt",
            "emg_zygo",
            "emg_coru",
            "emg_trap",
        ]

        for src_col, dst_name in zip(
            range(1, 9),
            names,
        ):

            result[dst_name] = np.interp(
                target_time,
                source_time,
                physiology_df[src_col]
                .astype(float)
                .to_numpy(),
            )

        #
        # video labels
        #




        result["video"] = np.round(
            np.interp(
                target_time,
                source_time,
                physiology_df[9]
                .astype(float)
                .to_numpy(),
            )
        ).astype(int)

        return result

    # ------------------------------------------------------
    # Emotion
    # ------------------------------------------------------

    def resample_annotations(
        self,
        annotation_df: pd.DataFrame,
    ) -> pd.DataFrame:

        source_time = (
            annotation_df.iloc[:, 0]
            .to_numpy()
            .astype(float)
        )

        target_time = self.build_target_grid(
            source_time[-1]
        )

        result = pd.DataFrame()

        result["time_ms"] = target_time

        result["valence"] = np.interp(
            target_time,
            source_time,
            annotation_df[1]
            .astype(float)
            .to_numpy(),
        )

        result["arousal"] = np.interp(
            target_time,
            source_time,
            annotation_df[2]
            .astype(float)
            .to_numpy(),
        )

        return result

    # ------------------------------------------------------
    # Merge
    # ------------------------------------------------------

    def merge_modalities(
        self,
        physiology_df: pd.DataFrame,
        annotation_df: pd.DataFrame,
        subject_id: int,
    ) -> pd.DataFrame:

        merged = pd.merge(
            physiology_df,
            annotation_df,
            on="time_ms",
            how="inner",
        )

        merged["subject_id"] = (
            subject_id
        )

        return merged

    # ------------------------------------------------------
    # Public API
    # ------------------------------------------------------

    def process_subject(
        self,
        physiology_df: pd.DataFrame,
        annotation_df: pd.DataFrame,
        subject_id: int,
    ) -> HarmonizedSubject:

        physiology_resampled = (
            self.resample_physiology(
                physiology_df
            )
        )

        annotation_resampled = (
            self.resample_annotations(
                annotation_df
            )
        )

        dataframe = (
            self.merge_modalities(
                physiology_resampled,
                annotation_resampled,
                subject_id,
            )
        )

        return HarmonizedSubject(
            dataframe=dataframe
        )