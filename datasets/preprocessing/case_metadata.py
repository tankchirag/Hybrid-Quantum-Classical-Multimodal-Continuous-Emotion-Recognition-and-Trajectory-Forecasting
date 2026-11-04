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


def infer_category(
    video_name: str,
) -> str:

    if video_name.startswith(
        "amusing"
    ):
        return "amusing"

    if video_name.startswith(
        "boring"
    ):
        return "boring"

    if video_name.startswith(
        "relaxed"
    ):
        return "relaxed"

    if video_name.startswith(
        "scary"
    ):
        return "scary"

    if video_name == "startVid":
        return "baseline_start"

    if video_name == "bluVid":
        return "baseline_blue"

    if video_name == "endVid":
        return "baseline_end"

    return "unknown"


@dataclass(frozen=True)
class VideoInfo:

    video_id: int

    video_name: str

    duration_ms: float

    category: str = "unknown"

    source_movie: str | None = None

    source_url: str | None = None

    notes: str | None = None


@dataclass
class SubjectSequence:
    """
    Video sequence used for one participant.
    """

    subject_id: int
    video_ids: List[int]


@dataclass(frozen=True)
class ParticipantInfo:

    subject_id: int

    sex: str

    age_group: str

    sequence_id: int


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

        self.participants: Dict[
            int,
            ParticipantInfo,
        ] = {}

    # --------------------------------------------------
    # Video durations
    # --------------------------------------------------

    def load_rich_video_metadata(
        self,
    ) -> Dict[int, VideoInfo]:

        file_path = (
            self.metadata_root
            / "videos.xlsx"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing file: {file_path}"
            )

        df = pd.read_excel(
            file_path,
            engine="openpyxl",
        )

        video_infos: Dict[
            int,
            VideoInfo,
        ] = {}

        def optional_text(
            value: object,
        ) -> str | None:

            if pd.isna(value):
                return None

            return str(value).strip()

        for _, row in df.iterrows():

            #
            # Skip header/helper rows
            #

            if pd.isna(
                row["Video-ID"]
            ):
                continue

            if pd.isna(
                row["Video-label"]
            ):
                continue

            video_name = str(
                row["Video-label"]
            ).strip()

            video_id = int(
                row["Video-ID"]
            )

            if video_name not in VIDEO_NAME_TO_ID:
                raise ValueError(
                    f"Unknown video "
                    f"name: {video_name}"
                )

            if VIDEO_NAME_TO_ID[video_name] != video_id:
                raise ValueError(
                    f"Video ID mismatch "
                    f"for {video_name}: "
                    f"{video_id}"
                )

            video_infos[
                video_id
            ] = VideoInfo(
                video_id=video_id,
                video_name=video_name,
                duration_ms=float(
                    row["Duration (in ms)"]
                ),
                category=infer_category(
                    video_name
                ),
                source_movie=optional_text(
                    row["Source (Year)"]
                ),
                source_url=optional_text(
                    row["Source--URL"]
                ),
                notes=optional_text(
                    row["Notes"]
                ),
            )

        self.video_infos = video_infos

        return video_infos

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

    def load_participants(
        self,
    ) -> Dict[int, ParticipantInfo]:

        file_path = (
            self.metadata_root
            / "participants.xlsx"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing file: {file_path}"
            )

        df = pd.read_excel(
            file_path,
            engine="openpyxl",
        )

        df.columns = [
            str(c).strip()
            for c in df.columns
        ]

        participants: Dict[
            int,
            ParticipantInfo,
        ] = {}

        for _, row in df.iterrows():

            subject_id = int(
                row["Participant-ID"]
            )

            participants[
                subject_id
            ] = ParticipantInfo(
                subject_id=subject_id,
                sex=str(
                    row["Sex"]
                ).strip(),
                age_group=str(
                    row["Age-Group"]
                ).strip(),
                sequence_id=int(
                    row["Video Sequence Used"]
                ),
            )

        self.participants = participants

        return participants

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

        self.load_rich_video_metadata()

        self.load_participants()

        self.load_subject_sequences()

        self.validate()

    def get_video_info(
        self,
        video_id: int,
    ) -> VideoInfo:

        return self.video_infos[
            video_id
        ]


    def get_video_name(
        self,
        video_id: int,
    ) -> str:

        return self.video_infos[
            video_id
        ].video_name

    def get_video_category(
        self,
        video_id: int,
    ) -> str:

        return self.video_infos[
            video_id
        ].category

    def get_participant_info(
        self,
        subject_id: int,
    ) -> ParticipantInfo:

        return self.participants[
            subject_id
        ]

    def get_subject_sequence_id(
        self,
        subject_id: int,
    ) -> int:

        return self.participants[
            subject_id
        ].sequence_id

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