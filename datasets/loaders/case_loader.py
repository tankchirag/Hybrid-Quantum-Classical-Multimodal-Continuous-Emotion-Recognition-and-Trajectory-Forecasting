"""
datasets.loaders.case_loader
============================

CASE NPZ Dataset Loader

Supports:

    split="all"
    split="train"
    split="validation"
    split="test"

Returns:

{
    "X": ...,
    "y": ...,
    "metadata": ...
}
"""

from __future__ import annotations

from pathlib import Path

import csv
import json
import numpy as np

from torch.utils.data import Dataset


class CaseDataset(
    Dataset,
):

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "all",
    ):

        self.dataset_root = Path(
            dataset_root
        )

        self.split = (
            split.lower()
        )

        self.samples_dir = (
            self.dataset_root
            / "samples"
        )

        self.manifest_dir = (
            self.dataset_root
            / "manifests"
        )

        if (
            not self.samples_dir.exists()
        ):

            raise FileNotFoundError(
                f"Missing directory:\n"
                f"{self.samples_dir}"
            )

        valid_splits = {
            "all",
            "train",
            "validation",
            "test",
        }

        if (
            self.split
            not in valid_splits
        ):

            raise ValueError(
                f"Unsupported split: "
                f"{self.split}"
            )

        # ----------------------------------
        # Discover files
        # ----------------------------------

        if self.split == "all":

            self.files = sorted(
                self.samples_dir.glob(
                    "*.npz"
                )
            )

        else:

            self.files = (
                self._load_manifest_files()
            )

        if len(self.files) == 0:

            raise RuntimeError(
                f"No NPZ files found "
                f"for split={self.split}"
            )

        print(
            f"[CaseDataset] "
            f"Split={self.split} "
            f"Files={len(self.files)}"
        )

    # ==================================================
    # Manifest
    # ==================================================

    def _load_manifest_files(
        self,
    ):

        manifest_path = (
            self.manifest_dir
            /
            f"{self.split}_manifest.csv"
        )

        if (
            not manifest_path.exists()
        ):

            raise FileNotFoundError(
                f"Manifest not found:\n"
                f"{manifest_path}"
            )

        files = []

        with open(
            manifest_path,
            "r",
            encoding="utf-8",
        ) as fp:

            reader = csv.DictReader(
                fp
            )

            for row in reader:

                file_path = (
                    self.samples_dir
                    / row["file_name"]
                )

                if (
                    not file_path.exists()
                ):

                    raise FileNotFoundError(
                        f"Referenced file "
                        f"missing:\n"
                        f"{file_path}"
                    )

                files.append(
                    file_path
                )

        return sorted(files)

    # ==================================================
    # Dataset API
    # ==================================================

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

    # ==================================================
    # Helpers
    # ==================================================

    def get_metadata(
        self,
        index: int,
    ):

        return self[
            index
        ][
            "metadata"
        ]

    @property
    def num_files(
        self,
    ) -> int:

        return len(
            self.files
        )