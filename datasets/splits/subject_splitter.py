"""
Subject Split Policy.

Purpose
-------
Define train / validation / test partitions
using the YAML configuration.

This implements subject-independent
evaluation and prevents leakage across splits.
"""

from __future__ import annotations

from dataclasses import dataclass

from configs.dataset_config import (
    DatasetConfig,
)


CASE_CONFIG = DatasetConfig.load(
    "configs/datasets/case.yaml"
)


@dataclass(frozen=True)
class SubjectSplitResult:

    train_subjects: list[int]

    validation_subjects: list[int]

    test_subjects: list[int]


class SubjectSplitter:

    def __init__(
        self,
        config: DatasetConfig = CASE_CONFIG,
    ) -> None:

        self.config = config

        splits = (
            config.raw["splits"]
        )

        self.train_range = (
            splits["train_subjects"]
        )

        self.validation_range = (
            splits["validation_subjects"]
        )

        self.test_range = (
            splits["test_subjects"]
        )

        self.total_subjects = (
            config.raw["subjects"]
            ["total_subjects"]
        )

    @staticmethod
    def build_range(
        start: int,
        end: int,
    ) -> list[int]:

        return list(
            range(
                start,
                end + 1,
            )
        )

    def build_splits(
        self,
    ) -> SubjectSplitResult:

        train = self.build_range(
            self.train_range["start"],
            self.train_range["end"],
        )

        validation = self.build_range(
            self.validation_range["start"],
            self.validation_range["end"],
        )

        test = self.build_range(
            self.test_range["start"],
            self.test_range["end"],
        )

        return SubjectSplitResult(
            train_subjects=train,
            validation_subjects=validation,
            test_subjects=test,
        )

    def get_split(
        self,
        subject_id: int,
    ) -> str:

        splits = self.build_splits()

        if subject_id in splits.train_subjects:
            return "train"

        if subject_id in splits.validation_subjects:
            return "validation"

        if subject_id in splits.test_subjects:
            return "test"

        raise ValueError(
            f"Subject {subject_id} "
            f"not assigned to a split."
        )

    def validate(
        self,
    ) -> None:

        splits = self.build_splits()

        train = set(
            splits.train_subjects
        )

        validation = set(
            splits.validation_subjects
        )

        test = set(
            splits.test_subjects
        )

        assert train.isdisjoint(
            validation
        )

        assert train.isdisjoint(
            test
        )

        assert validation.isdisjoint(
            test
        )

        all_subjects = (
            train
            |
            validation
            |
            test
        )

        expected = set(
            range(
                1,
                self.total_subjects + 1,
            )
        )

        assert (
            all_subjects
            ==
            expected
        )