# datasets/preprocessing/case_label_generator.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from datasets.preprocessing.case_metadata import (
    CaseMetadataManager,
)


@dataclass
class LabelGenerationResult:
    """
    Generated video labels.
    """

    labels: np.ndarray
    trimmed_rows: int


class CaseLabelGenerator:
    """
    Python port of CASE f_labelData.m

    Generates a video-id vector for a
    physiological or annotation time vector.
    """

    def __init__(
        self,
        metadata_manager: CaseMetadataManager,
    ) -> None:

        self.metadata = metadata_manager

    def generate(
        self,
        time_vector: np.ndarray,
        subject_id: int,
    ) -> LabelGenerationResult:
        """
        Parameters
        ----------
        time_vector:
            Time values in milliseconds.

        subject_id:
            CASE participant id [1..30]

        Returns
        -------
        LabelGenerationResult
        """

        time_vector = np.asarray(
            time_vector,
            dtype=float,
        )

        labels = np.zeros(
            len(time_vector),
            dtype=int,
        )

        subject_sequence = (
            self.metadata
            .get_subject_sequence(
                subject_id
            )
        )

        last_time_found = 0.0
        video_pos_start = 0

        for video_id in subject_sequence:

            duration_ms = (
                self.metadata
                .video_infos[
                    video_id
                ]
                .duration_ms
            )

            time_to_match = (
                last_time_found
                + duration_ms
            )

            closest_index = int(
                np.argmin(
                    np.abs(
                        time_vector
                        - time_to_match
                    )
                )
            )

            labels[
                video_pos_start:
                closest_index + 1
            ] = video_id

            last_time_found = (
                time_vector[
                    closest_index
                ]
            )

            video_pos_start = (
                closest_index + 1
            )

        trimmed_rows = int(
            np.sum(labels == 0)
        )

        return LabelGenerationResult(
            labels=labels,
            trimmed_rows=trimmed_rows,
        )