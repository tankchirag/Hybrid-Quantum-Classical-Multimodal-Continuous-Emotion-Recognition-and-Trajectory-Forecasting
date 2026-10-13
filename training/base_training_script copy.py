"""
training.base_training_script
=============================

Shared Training Infrastructure

Used By

    • BaselineB1

    • BaselineB2

    • BaselineB3

    • QSAL

Design Goals
------------

✓ Dataset Agnostic

✓ Modality Agnostic

✓ Label Agnostic

✓ Continuous Labels

✓ Discrete Labels

✓ Recognition

✓ Forecasting

✓ Continuous Forecasting

✓ Discrete Forecasting

✓ History Labels

✓ Reusable

✓ Future Proof
"""

from __future__ import annotations

import json
import os
import time

import torch

from torch.utils.data import DataLoader

from datasets.ulm_tsst_synthetic import (
    SyntheticULMTSSTDataset,
)

from datasets.window_sampler import (
    WindowSampler,
)

from datasets.collate_fn import (
    multimodal_collate_fn,
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

def get_device(
    config,
) -> torch.device:

    requested = getattr(
        config,
        "device",
        "cpu",
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
# Window Dataset Wrapper
# ==========================================================

class WindowedDataset:
    """
    Converts clip-level dataset
    to sample-level window dataset.
    """

    def __init__(
        self,
        base_dataset,
        sampler,
    ):

        self.base_dataset = (
            base_dataset
        )

        self.sampler = sampler

        self.indices = []

        self._build_index()

    def _build_index(
        self,
    ):

        for clip_idx in range(
            len(self.base_dataset)
        ):

            clip = self.base_dataset[
                clip_idx
            ]

            if len(
                clip[
                    "continuous_labels"
                ]
            ) == 0:

                continue

            label_name = next(
                iter(
                    clip[
                        "continuous_labels"
                    ]
                )
            )

            clip_length = (
                clip[
                    "continuous_labels"
                ][
                    label_name
                ].shape[0]
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

    def __len__(
        self,
    ):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        index: int,
    ):

        clip_idx, center_idx = (
            self.indices[index]
        )

        clip = self.base_dataset[
            clip_idx
        ]

        return self.sampler.sample(
            clip=clip,
            center_idx=center_idx,
        )


# ==========================================================
# Dataset Factory
# ==========================================================

def build_clip_dataset(
    config,
):

    dataset_name = getattr(
        config,
        "dataset_name",
        "ulm_tsst_synthetic",
    )

    if dataset_name == (
        "ulm_tsst_synthetic"
    ):

        return SyntheticULMTSSTDataset(

            num_samples=
            config.num_samples,

            duration_seconds=
            config.duration_seconds,

            modalities=
            config.modalities,

            continuous_labels=
            config.continuous_labels,

            discrete_labels=
            config.discrete_labels,

            sampling_rates=
            config.sampling_rates,

            label_sampling_rate=
            getattr(
                config,
                "label_sampling_rate",
                25,
            ),
        )

    raise ValueError(
        f"Unsupported dataset: "
        f"{dataset_name}"
    )


# ==========================================================
# Dataset Builder
# ==========================================================

def build_dataset(
    config,
):

    clip_dataset = (
        build_clip_dataset(
            config
        )
    )

    sampler = WindowSampler(

        history_length=
        config.history_length,

        forecast_horizon=
        config.forecast_horizon,
    )

    return WindowedDataset(

        base_dataset=
        clip_dataset,

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
            config
        )
    )

    val_dataset = (
        build_dataset(
            config
        )
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=
        config.batch_size,

        shuffle=True,

        collate_fn=
        multimodal_collate_fn,

        pin_memory=
        (
            device.type
            ==
            "cuda"
        ),
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=
        config.batch_size,

        shuffle=False,

        collate_fn=
        multimodal_collate_fn,

        pin_memory=
        (
            device.type
            ==
            "cuda"
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
    config,
    config_path: str,
):

    with open(
        config_path,
        "w",
    ) as fp:

        json.dump(

            vars(config),

            fp,

            indent=4,

            default=str,
        )


# ==========================================================
# Shared Training Runner
# ==========================================================

def run_training(
    *,
    config,
    model_class,
    run_dir_name: str,
):

    device = get_device(
        config
    )

    print(
        f"[INFO] Device: {device}"
    )

    run_dir = create_run_dir(
        run_dir_name
    )

    checkpoint_path = os.path.join(
        run_dir,
        "best_model.pt",
    )

    history_path = os.path.join(
        run_dir,
        "history.json",
    )

    config_path = os.path.join(
        run_dir,
        "config_used.json",
    )

    save_config(
        config,
        config_path,
    )

    train_loader, val_loader = (
        build_dataloaders(
            config,
            device,
        )
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = model_class(
        config=config
    )

    # ------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------

    optimizer = build_optimizer(

        model=model,

        config=config,
    )

    # ------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------

    scheduler = build_scheduler(

        optimizer=optimizer,

        config=config,
    )

    # ------------------------------------------------------
    # Trainer
    # ------------------------------------------------------

    trainer = Trainer(

        model=model,

        optimizer=
        optimizer,

        scheduler=
        scheduler,

        device=device,

        patience=
        config.patience,

        save_path=
        checkpoint_path,

        use_tqdm=True,

        log_batches=True,
    )

    # ------------------------------------------------------
    # Fit
    # ------------------------------------------------------

    start_time = time.time()

    history = trainer.fit(

        train_loader=
        train_loader,

        val_loader=
        val_loader,

        epochs=
        config.epochs,
    )

    elapsed_time = (
        time.time()
        - start_time
    )

    with open(
        history_path,
        "w",
    ) as fp:

        json.dump(

            history,

            fp,

            indent=4,

            default=str,
        )

    print(
        "\n[INFO] Training Completed"
    )

    print(
        f"[INFO] Runtime: "
        f"{elapsed_time:.2f}s"
    )

    print(
        f"[INFO] Best Model: "
        f"{checkpoint_path}"
    )

    print(
        f"[INFO] History: "
        f"{history_path}"
    )

    return history