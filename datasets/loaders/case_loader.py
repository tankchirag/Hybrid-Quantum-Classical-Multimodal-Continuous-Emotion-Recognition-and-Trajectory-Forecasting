"""
datasets.loaders.case_loader
============================

CASE NPZ Dataset Loader

Reads:

    sub_XX_vid_YY.npz

Returns:

{
    "X": ...,
    "y": ...,
    "metadata": ...
}
"""

from __future__ import annotations

from pathlib import Path

import json
import numpy as np

from torch.utils.data import Dataset


class CaseDataset(
    Dataset,
):

    def __init__(
        self,
        dataset_root: str | Path,
    ):

        self.dataset_root = Path(
            dataset_root
        )

        self.samples_dir = (
            self.dataset_root
            / "samples"
        )

        if (
            not self.samples_dir.exists()
        ):

            raise FileNotFoundError(
                f"Missing directory:\n"
                f"{self.samples_dir}"
            )

        self.files = sorted(

            self.samples_dir.glob(
                "*.npz"
            )
        )

        if len(self.files) == 0:

            raise RuntimeError(
                f"No NPZ files found:\n"
                f"{self.samples_dir}"
            )

        print(
            f"[CaseDataset] "
            f"Files={len(self.files)}"
        )

    def __len__(
        self,
    ) -> int:

        return len(
            self.files
        )

    def __getitem__(
        self,
        index: int,
    ):

        file_path = (
            self.files[index]
        )

        data = np.load(
            file_path,
            allow_pickle=True,
        )

        metadata = json.loads(
            str(
                data["metadata"]
            )
        )

        return {

            "X":
            data["X"],

            "y":
            data["y"],

            "metadata":
            metadata,
        }

    def get_metadata(
        self,
        index: int,
    ):

        return self[
            index
        ][
            "metadata"
        ]