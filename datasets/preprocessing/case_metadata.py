# datasets/preprocessing/case_metadata.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


VIDEO_NAME_TO_ID = {
    "amusing-1": 1,
    "amusing-2": 2,
    "boring-1": 3,
    "boring-2": 4,
    "relaxed-1": 5,
    "relaxed-2": 6,
    "scary-1": 7,
    "scary-2": 8,
    "startVid": 10,
    "bluVid": 11,
    "endVid": 12,
}


VIDEO_ID_TO_NAME = {
    value: key
    for key, value in VIDEO_NAME_TO_ID.items()
}


AUTHOR_SEQUENCE_DURATION_MS = 2451583.333


@dataclass(frozen=True)
class VideoInfo:
    """
    Metadata for one video.
    """

    video_id: int
    video_name: str
    duration_ms: float


@dataclass
class SubjectSequence:
    """
    Video sequence used for one participant.
    """

    subject_id: int
    video_ids: List[int]


class CaseMetadataManager:
    """
    Loads and validates CASE metadata.

    Responsibilities
    ----------------
    - video durations
    - participant specific video ordering
    - metadata validation
    - sequence duration reconstruction

    Does NOT:
    - generate labels
    - preprocess signals
    - perform interpolation
    """

    def __init__(
        self,
        metadata_root: Path,
    ) -> None:

        self.metadata_root = Path(
            metadata_root
        )

        self.video_infos: Dict[
            int,
            VideoInfo,
        ] = {}

        self.subject_sequences: Dict[
            int,
            SubjectSequence,
        ] = {}

    # --------------------------------------------------
    # Video durations
    # --------------------------------------------------

    def load_video_durations(
        self,
    ) -> Dict[int, VideoInfo]:

        file_path = (
            self.metadata_root
            / "videos_duration.txt"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing file: {file_path}"
            )

        df = pd.read_csv(
            file_path,
            sep="\t",
            engine="python",
            quotechar='"',
            usecols=[0, 1],
        )

        video_infos: Dict[
            int,
            VideoInfo,
        ] = {}

        for _, row in df.iterrows():

            video_name = str(
                row["video_name"]
            ).strip()

            duration_ms = float(
                row["video_duration"]
            )

            if video_name not in VIDEO_NAME_TO_ID:

                raise ValueError(
                    f"Unknown video "
                    f"name: {video_name}"
                )

            video_id = (
                VIDEO_NAME_TO_ID[
                    video_name
                ]
            )

            video_infos[
                video_id
            ] = VideoInfo(
                video_id=video_id,
                video_name=video_name,
                duration_ms=duration_ms,
            )

        self.video_infos = video_infos

        return video_infos

    # --------------------------------------------------
    # Subject sequences
    # --------------------------------------------------

    def load_subject_sequences(
        self,
    ) -> Dict[int, SubjectSequence]:

        file_path = (
            self.metadata_root
            / "seqs_order.txt"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing file: {file_path}"
            )

        df = pd.read_csv(
            file_path,
            sep="\t",
            engine="python",
            quotechar='"',
        )

        subject_sequences = {}

        for subject_idx in range(
            len(df.columns)
        ):

            column_name = (
                df.columns[
                    subject_idx
                ]
            )

            subject_id = (
                subject_idx + 1
            )

            video_names = (
                df[column_name]
                .astype(str)
                .tolist()
            )

            video_ids = []

            for name in video_names:

                name = name.strip()

                if (
                    name
                    not in VIDEO_NAME_TO_ID
                ):
                    raise ValueError(
                        f"Unknown video "
                        f"name: {name}"
                    )

                video_ids.append(
                    VIDEO_NAME_TO_ID[
                        name
                    ]
                )

            subject_sequences[
                subject_id
            ] = SubjectSequence(
                subject_id=subject_id,
                video_ids=video_ids,
            )

        self.subject_sequences = (
            subject_sequences
        )

        return subject_sequences

    # --------------------------------------------------
    # Sequence duration
    # --------------------------------------------------

    def get_subject_duration_ms(
        self,
        subject_id: int,
    ) -> float:

        if (
            subject_id
            not in self.subject_sequences
        ):
            raise KeyError(
                f"Unknown subject "
                f"{subject_id}"
            )

        total = 0.0

        sequence = (
            self.subject_sequences[
                subject_id
            ]
        )

        for video_id in (
            sequence.video_ids
        ):

            total += (
                self.video_infos[
                    video_id
                ].duration_ms
            )

        return total

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def validate_subject_count(
        self,
    ) -> None:

        if (
            len(
                self.subject_sequences
            )
            != 30
        ):
            raise ValueError(
                "Expected "
                "30 subject sequences."
            )

    def validate_video_ids(
        self,
    ) -> None:

        expected = {
            1, 2, 3, 4,
            5, 6, 7, 8,
            10, 11, 12,
        }

        found = set(
            self.video_infos.keys()
        )

        if expected != found:

            raise ValueError(
                "Video IDs mismatch.\n"
                f"Expected: {expected}\n"
                f"Found: {found}"
            )

    def validate_sequence_duration(
        self,
    ) -> None:

        for subject_id in range(
            1,
            31,
        ):

            duration = (
                self.get_subject_duration_ms(
                    subject_id
                )
            )

            delta = abs(
                duration
                - AUTHOR_SEQUENCE_DURATION_MS
            )

            if delta > 1.0:

                raise ValueError(
                    f"Duration mismatch "
                    f"for subject "
                    f"{subject_id}: "
                    f"{duration}"
                )

    def validate(
        self,
    ) -> None:

        self.validate_video_ids()

        self.validate_subject_count()

        self.validate_sequence_duration()

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def load(
        self,
    ) -> None:

        self.load_video_durations()

        self.load_subject_sequences()

        self.validate()

    # --------------------------------------------------
    # Convenience helpers
    # --------------------------------------------------

    def get_subject_sequence(
        self,
        subject_id: int,
    ) -> List[int]:
        """
        Return the video sequence for a subject.
        """

        return (
            self.subject_sequences[
                subject_id
            ].video_ids
        )

    def summary(
        self,
    ) -> dict:

        return {
            "videos": len(
                self.video_infos
            ),
            "subjects": len(
                self.subject_sequences
            ),
            "duration_ms":
                AUTHOR_SEQUENCE_DURATION_MS,
        }