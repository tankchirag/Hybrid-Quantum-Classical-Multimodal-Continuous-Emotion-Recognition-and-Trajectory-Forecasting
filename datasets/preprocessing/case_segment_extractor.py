"""
CASE Segment Extractor.

Purpose
-------
Convert:

    subject sequence
    +
    video durations

into temporal video segments.

Example
-------

Video 3

start_ms :
221500

end_ms :
340166

duration_ms :
118666
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoSegment:

    video_id: int

    video_name: str

    start_ms: float

    end_ms: float

    duration_ms: float


class CaseSegmentExtractor:

    def __init__(
        self,
        metadata_manager,
    ):

        self.metadata = (
            metadata_manager
        )

    def extract_subject_segments(
        self,
        subject_id: int,
    ) -> list[VideoSegment]:

        sequence = (
            self.metadata
            .get_subject_sequence(
                subject_id
            )
        )

        timeline_ms = 0.0

        segments = []

        for video_id in sequence:

            video_info = (
                self.metadata
                .video_infos[
                    video_id
                ]
            )

            start_ms = (
                timeline_ms
            )

            end_ms = (
                start_ms
                +
                video_info.duration_ms
            )

            segments.append(
                VideoSegment(
                    video_id=
                    video_id,

                    video_name=
                    video_info.video_name,

                    start_ms=
                    start_ms,

                    end_ms=
                    end_ms,

                    duration_ms=
                    video_info.duration_ms,
                )
            )

            timeline_ms = (
                end_ms
            )

        return segments

    def get_segment(
        self,
        subject_id: int,
        video_id: int,
    ) -> VideoSegment:

        segments = (
            self.extract_subject_segments(
                subject_id
            )
        )

        for segment in segments:

            if (
                segment.video_id
                == video_id
            ):

                return segment

        raise ValueError(
            f"Video {video_id} "
            f"not found for "
            f"subject {subject_id}"
        )