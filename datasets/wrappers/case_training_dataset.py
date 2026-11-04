"""
datasets.wrappers.case_training_dataset
=======================================

CASE Training Dataset Wrapper

Purpose
-------
Convert CASE NPZ files into a true
training dataset where:

    one dataset item
    =
    one temporal window

This matches the assumptions used by:

    DataLoader
        ↓
    collate()
        ↓
    Trainer
        ↓
    BaselineB1

Without this wrapper, each NPZ file is
treated as a single sample:

    X.shape
        (N_windows, history, features)

which causes batching failures because
different videos contain different
numbers of windows.

Example
-------

File A

    X.shape = (232,120,8)

File B

    X.shape = (182,120,8)

These cannot be stacked.

The wrapper creates an internal
window index:

    dataset[0]
        -> window 0 of file A

    dataset[1]
        -> window 1 of file A

    ...

    dataset[232]
        -> window 0 of file B

Result

    X.shape

        (120,8)

for every returned training sample.
"""

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
    Window-level CASE dataset.

    Dataset item

        one temporal window

    instead of

        one full video.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
    ):
        super().__init__()

        self.base_dataset = (
            CaseDataset(
                dataset_root=dataset_root,
                split=split,
            )
        )

        self.adapter = (
            CaseTrainingAdapter()
        )

        # ----------------------------------
        # Global window index
        # ----------------------------------

        self.window_index = []

        print(
            "[INFO] Building CASE window index..."
        )

        for file_idx in range(
            len(self.base_dataset)
        ):

            sample = (
                self.base_dataset[
                    file_idx
                ]
            )

            num_windows = (
                sample["X"].shape[0]
            )

            for window_idx in range(
                num_windows
            ):

                self.window_index.append(
                    (
                        file_idx,
                        window_idx,
                    )
                )

        print(
            f"[INFO] CASE Windows="
            f"{len(self.window_index)}"
        )

    # ==================================================
    # Dataset API
    # ==================================================

    def __len__(
        self,
    ) -> int:

        return len(
            self.window_index
        )

    def __getitem__(
        self,
        index: int,
    ):

        file_idx, window_idx = (
            self.window_index[
                index
            ]
        )

        sample = (
            self.base_dataset[
                file_idx
            ]
        )

        X = sample["X"][window_idx]

        y = sample["y"][window_idx]

        label_history = (
            sample["label_history"][
                window_idx
            ]
        )

        metadata = sample.get(
            "metadata",
            {},
        )

        target_names = sample.get(
            "target_names",
            None,
        )

        return self.adapter(
            {
                "X": X,
                "y": y,
                "label_history":
                    label_history,
                "target_names":
                    target_names,
                "metadata":
                    metadata,
            }
        )