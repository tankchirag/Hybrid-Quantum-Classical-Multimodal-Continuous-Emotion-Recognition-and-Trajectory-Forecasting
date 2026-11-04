"""
Manifest Builder.

Creates subject/video inventory using
dataset configuration.

Used later to generate:

    train_manifest.csv
    val_manifest.csv
    test_manifest.csv
"""

from __future__ import annotations

from dataclasses import dataclass

from configs.dataset_config import (
    DatasetConfig,
)

from datasets.splits.subject_splitter import (
    SubjectSplitter,
)


CASE_CONFIG = DatasetConfig.load(
    "configs/datasets/case.yaml"
)


@dataclass
class ManifestEntry:

    subject_id: int

    video_id: int

    split: str

    npz_path: str


class ManifestBuilder:

    def __init__(
        self,
        config: DatasetConfig = CASE_CONFIG,
    ):

        self.config = config

        self.splitter = (
            SubjectSplitter(config)
        )

        self.video_ids = (
            config.raw["videos"]
            ["include"]
        )

        self.total_subjects = (
            config.raw["subjects"]
            ["total_subjects"]
        )

    def build(self):

        entries = []

        for subject_id in range(
            1,
            self.total_subjects + 1,
        ):

            split = (
                self.splitter.get_split(
                    subject_id
                )
            )

            for video_id in self.video_ids:

                npz_path = (
                    f"processed/"
                    f"sub_{subject_id:02d}_"
                    f"vid_{video_id:02d}.npz"
                )

                entries.append(
                    ManifestEntry(
                        subject_id=
                        subject_id,

                        video_id=
                        video_id,

                        split=
                        split,

                        npz_path=
                        npz_path,
                    )
                )

        return entries