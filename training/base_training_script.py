"""
training.base_training_script
=============================

Shared Training Infrastructure

Supports

    • BaselineB1
    • BaselineB2
    • BaselineB3
    • QSAL

Uses

    ULMTSSTDiskDataset
        ↓
    TimelineSynchronizer
        ↓
    WindowSampler
        ↓
    Collate
        ↓
    Trainer

Design Goals
------------

✓ Dataset Agnostic

✓ Modality Agnostic

✓ Label Agnostic

✓ Recognition

✓ Forecasting

✓ Continuous Labels

✓ Discrete Labels

✓ History Compatible

✓ Future Proof
"""

from __future__ import annotations

import os

os.environ[
    "CUBLAS_WORKSPACE_CONFIG"
] = ":4096:8"

import json
import os
import time
from pathlib import Path

import torch

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from datasets.loaders.ulm_tsst_disk_dataset import (
    ULMTSSTDiskDataset,
)

from datasets.preprocessing.timeline_synchronizer import (
    TimelineSynchronizer,
)

from datasets.sampling.window_sampler import (
    WindowSampler,
)

from datasets.loaders.collate import (
    collate,
)

from training.optimizer import (
    build_optimizer,
)

from training.scheduler import (
    build_scheduler,
)

from training.trainer import (
    Trainer,
)

from utils.params import (
    print_parameter_report,
    save_parameter_report,
)

from utils.seed import (
    set_seed,
    print_seed_report,
)

from datasets.wrappers.case_training_dataset import (
    CaseTrainingDataset,
)


# ==========================================================
# Config Object
# ==========================================================

class ConfigNamespace:
    """
    Recursive attribute access.

    cfg["dataset"]["root_dir"]

    becomes

    cfg.dataset.root_dir
    """

    def __init__(self, data):

        for k, v in data.items():

            if isinstance(v, dict):
                v = ConfigNamespace(v)

            self.__dict__[k] = v


def to_namespace(config):

    if isinstance(config, dict):
        return ConfigNamespace(config)

    return config


# ==========================================================
# Run Directory
# ==========================================================

def create_run_dir(
    base_dir: str,
) -> str:

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = os.path.join(
        base_dir,
        timestamp,
    )

    os.makedirs(
        run_dir,
        exist_ok=True,
    )

    return run_dir


# ==========================================================
# Device
# ==========================================================

def get_device(config):

    runtime = getattr(
        config,
        "runtime",
        None,
    )

    requested = getattr(
        runtime,
        "device",
        "auto",
    )

    if requested == "auto":

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        requested == "cuda"
        and
        torch.cuda.is_available()
    ):
        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


# ==========================================================
# Window Dataset
# ==========================================================

class WindowedDataset(
    Dataset,
):

    def __init__(
        self,
        base_dataset,
        synchronizer,
        sampler,
    ):

        self.base_dataset = (
            base_dataset
        )

        self.synchronizer = (
            synchronizer
        )

        self.sampler = sampler

        self.indices = []

        self._build_index()

    def _build_index(self):

        print(
            "[INFO] Building window index..."
        )

        for clip_idx in range(
            len(self.base_dataset)
        ):

            clip = self.base_dataset[
                clip_idx
            ]

            clip = (
                self.synchronizer(
                    clip
                )
            )

            first_label = next(
                iter(
                    clip[
                        "continuous_labels"
                    ].values()
                )
            )

            clip_length = len(
                first_label
            )

            for center_idx in (
                self.sampler.valid_range(
                    clip_length
                )
            ):
                self.indices.append(
                    (
                        clip_idx,
                        center_idx,
                    )
                )

        print(
            f"[INFO] Windows={len(self.indices)}"
        )

    def __len__(self):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        index,
    ):

        clip_idx, center_idx = (
            self.indices[index]
        )

        clip = self.base_dataset[
            clip_idx
        ]

        clip = (
            self.synchronizer(
                clip
            )
        )

        return self.sampler.sample(

            synchronized_clip=
            clip,

            center_idx=
            center_idx,
        )


# ==========================================================
# Dataset Builder
# ==========================================================

def build_dataset(
    config,
    split,
):

    dataset_name = str(
        getattr(
            config.dataset,
            "name",
            "",
        )
    ).lower()

    # ----------------------------------
    # CASE
    # ----------------------------------

    if dataset_name == "case":

        return CaseTrainingDataset(

            dataset_root=
            config.dataset.root_dir,

            split=split,
        )

    # ----------------------------------
    # Default
    # ----------------------------------

    dataset = (
        ULMTSSTDiskDataset(

            dataset_root=
            config.dataset.root_dir,

            split=split,
        )
    )

    synchronizer = (
        TimelineSynchronizer()
    )

    sampler = WindowSampler(

        history_length=
        config.history.history_length,

        forecast_horizon=
        config.forecasting.horizon,
    )

    return WindowedDataset(

        base_dataset=
        dataset,

        synchronizer=
        synchronizer,

        sampler=
        sampler,
    )


# ==========================================================
# DataLoaders
# ==========================================================

def build_dataloaders(
    config,
    device,
):

    train_dataset = (
        build_dataset(
            config,
            "train",
        )
    )

    val_dataset = (
        build_dataset(
            config,
            "val",
        )
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=
        config.training.batch_size,

        shuffle=True,

        collate_fn=
        collate,

        pin_memory=
        (
            device.type
            == "cuda"
        ),
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=
        config.training.batch_size,

        shuffle=False,

        collate_fn=
        collate,

        pin_memory=
        (
            device.type
            == "cuda"
        ),
    )

    return (
        train_loader,
        val_loader,
    )


# ==========================================================
# Save Config
# ==========================================================

def save_config(
    config_dict,
    output_path,
):

    with open(
        output_path,
        "w",
    ) as fp:

        json.dump(
            config_dict,
            fp,
            indent=4,
        )


# ==========================================================
# Main Runner
# ==========================================================

def run_training(
    *,
    config,
    model_class,
    run_dir_name,
):

    raw_config = config

    config = to_namespace(
        config
    )

    # ==================================================
    # Reproducibility
    # ==================================================

    set_seed(

        seed=
        config.runtime.seed,

        deterministic=
        config.runtime.deterministic,
    )

    print_seed_report(

        seed=
        config.runtime.seed,

        deterministic=
        config.runtime.deterministic,
    )


    device = get_device(
        config
    )

    print(
        f"[INFO] Device={device}"
    )

    print()
    print("DATASET NAME")
    print(
        config.dataset.name
    )

    run_dir = create_run_dir(
        run_dir_name
    )

    save_config(

        raw_config,

        os.path.join(
            run_dir,
            "config_used.json",
        ),
    )

    train_loader, val_loader = (
        build_dataloaders(
            config,
            device,
        )
    )

    model = model_class(
        config
    )

    # ==================================================
    # Parameter Report
    # ==================================================

    print(
        "\n"
        + "=" * 70
    )
    print(
        "MODEL INITIALIZATION"
    )
    print(
        "=" * 70
    )
    
    print_parameter_report(
        model,
    )

    save_parameter_report(

        model,

        os.path.join(
            run_dir,
            "params.json",
        ),
    )

    optimizer = (
        build_optimizer(
            model,
            config,
        )
    )

    scheduler = (
        build_scheduler(
            optimizer,
            config,
        )
    )

    trainer = Trainer(

        model=model,

        optimizer=optimizer,

        scheduler=scheduler,

        device=device,

        patience=
        config.training.patience,

        save_path=
        os.path.join(
            run_dir,
            "best.pt",
        ),

        use_tqdm=True,

        log_batches=True,

        max_train_batches=
        getattr(
            config.training,
            "max_train_batches",
            None,
        ),

        max_val_batches=
        getattr(
            config.training,
            "max_val_batches",
            None,
        ),
    )

    last_checkpoint = os.path.join(
        run_dir,
        "last.pt",
    )

    if os.path.exists(
        last_checkpoint
    ):

        start_epoch = (
            trainer.load_checkpoint(
                last_checkpoint
            )
        )

        print(
            f"Loaded checkpoint "
            f"from epoch "
            f"{start_epoch}"
        )

    history = trainer.fit(

        train_loader=
        train_loader,

        val_loader=
        val_loader,

        epochs=
        config.training.epochs,
    )

    with open(

        os.path.join(
            run_dir,
            "history.json",
        ),

        "w",

    ) as fp:

        json.dump(
            history,
            fp,
            indent=4,
            default=str,
        )

    return history