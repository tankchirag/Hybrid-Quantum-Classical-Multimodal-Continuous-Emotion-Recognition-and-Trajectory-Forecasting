# datasets/wrappers/case_training_dataset.py

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from datasets.loaders.case_loader import (
    CaseDataset,
)

from datasets.adapters.case_training_adapter import (
    CaseTrainingAdapter,
)


class CaseTrainingDataset(
    Dataset,
):
    """
    Training-ready CASE dataset.

    Output contract matches the
    universal trainer expectations.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
    ):

        self.base_dataset = (
            CaseDataset(
                dataset_root=
                dataset_root,

                split=
                split,
            )
        )

        self.adapter = (
            CaseTrainingAdapter()
        )

    def __len__(
        self,
    ) -> int:

        return len(
            self.base_dataset
        )

    def __getitem__(
        self,
        index: int,
    ):

        sample = (
            self.base_dataset[
                index
            ]
        )

        return (
            self.adapter(
                sample
            )
        )