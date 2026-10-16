"""
datasets.loaders.ulm_tsst_disk_dataset
======================================

ULM-TSST Disk Dataset

Purpose
-------
Load previously generated train/val/test splits
from disk.

This dataset becomes the official interface
between:

    ULM-TSST Dataset Release

and

    Training Pipeline

Supported Splits
----------------

train

val

test

Returned Structure
------------------

sample = {

    "modalities": {...},

    "continuous_labels": {...},

    "discrete_labels": {...},

    "metadata": {...}
}

Design Goals
------------

✓ Reproducible

✓ Training Ready

✓ Forecasting Ready

✓ Future WindowSampler Compatible

✓ Future Synchronizer Compatible

✓ PyTorch Native
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset


# ==========================================================
# Dataset
# ==========================================================

class ULMTSSTDiskDataset(
    Dataset,
):
    """
    Disk-based ULM-TSST dataset.

    Parameters
    ----------
    dataset_root : str | Path

        Dataset root directory.

        Example:

        data/generated/ulm_tsst_v1

    split : str

        train | val | test

    device : str

        cpu | cuda
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        device: str = "cpu",
    ):

        self.dataset_root = Path(
            dataset_root
        )

        self.split = split

        self.device = torch.device(
            device
        )

        if split == "raw":

            self.split_dir = (
                self.dataset_root
                / "raw"
            )

        else:

            self.split_dir = (
                self.dataset_root
                / split
            )

        if not self.split_dir.exists():

            raise FileNotFoundError(

                f"Split directory not found:\n"
                f"{self.split_dir}"
            )

        self.files = sorted(

            self.split_dir.glob(
                "*.npz"
            )
        )

        if len(self.files) == 0:

            raise RuntimeError(

                f"No NPZ files found in:\n"
                f"{self.split_dir}"
            )

        print(

            f"[ULMTSSTDiskDataset] "

            f"Split={split} "

            f"Files={len(self.files)}"
        )

    # ------------------------------------------------------
    # Length
    # ------------------------------------------------------

    def __len__(
        self,
    ) -> int:

        return len(
            self.files
        )

    # ------------------------------------------------------
    # Internal
    # ------------------------------------------------------

    def _load_sample(
        self,
        file_path: Path,
    ):

        return np.load(
            file_path,
            allow_pickle=True,
        )

    # ------------------------------------------------------
    # Get Item
    # ------------------------------------------------------

    def __getitem__(
        self,
        index: int,
    ):

        file_path = (
            self.files[index]
        )

        sample = self._load_sample(
            file_path
        )

        modalities = {}

        continuous_labels = {}

        discrete_labels = {}

        # --------------------------------------------------
        # Parse Keys
        # --------------------------------------------------

        for key in sample.files:

            data = sample[key]

            if key.startswith(
                "mod_"
            ):

                modality_name = (
                    key.replace(
                        "mod_",
                        "",
                    )
                )

                modalities[
                    modality_name
                ] = torch.tensor(
                    data,
                    dtype=torch.float32,
                )

            elif key.startswith(
                "cont_"
            ):

                label_name = (
                    key.replace(
                        "cont_",
                        "",
                    )
                )

                continuous_labels[
                    label_name
                ] = torch.tensor(
                    data,
                    dtype=torch.float32,
                )

            elif key.startswith(
                "disc_"
            ):

                label_name = (
                    key.replace(
                        "disc_",
                        "",
                    )
                )

                discrete_labels[
                    label_name
                ] = torch.tensor(
                    data,
                    dtype=torch.long,
                )

        metadata = {

            "sample_index":
                index,

            "file_name":
                file_path.name,

            "split":
                self.split,
        }

        if (
            "meta_clip_id"
            in sample.files
        ):

            metadata[
                "clip_id"
            ] = int(
                sample[
                    "meta_clip_id"
                ]
            )

        if (
            "meta_subject_id"
            in sample.files
        ):

            metadata[
                "subject_id"
            ] = int(
                sample[
                    "meta_subject_id"
                ]
            )

        if (
            "meta_session_id"
            in sample.files
        ):

            metadata[
                "session_id"
            ] = int(
                sample[
                    "meta_session_id"
                ]
            )

        if (
            "meta_duration_seconds"
            in sample.files
        ):

            metadata[
                "duration_seconds"
            ] = int(
                sample[
                    "meta_duration_seconds"
                ]
            )

        return {

            "modalities":
                modalities,

            "continuous_labels":
                continuous_labels,

            "discrete_labels":
                discrete_labels,

            "metadata":
                metadata,
        }

    # ------------------------------------------------------
    # Convenience
    # ------------------------------------------------------

    @property
    def num_files(
        self,
    ) -> int:

        return len(
            self.files
        )

    @property
    def split_name(
        self,
    ) -> str:

        return self.split