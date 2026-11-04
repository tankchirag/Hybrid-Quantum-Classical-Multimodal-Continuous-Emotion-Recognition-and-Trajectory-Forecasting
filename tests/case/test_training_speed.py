from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

import torch

from torch.utils.data import DataLoader

from utils.config import load_config

from datasets.wrappers.case_training_dataset import (
    CaseTrainingDataset,
)

from datasets.loaders.collate import (
    collate,
)

from training.losses import (
    UnifiedMultiTaskLoss,
)

from models.wrappers.baseline_b1 import (
    BaselineB1,
)

from models.wrappers.baseline_b2 import (
    BaselineB2,
)

from models.wrappers.baseline_b3 import (
    BaselineB3,
)

from models.wrappers.qsal import (
    QSAL,
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CASE_WINDOWS = 28560
CASE_BATCH_SIZE = 8

BATCHES_PER_EPOCH = (
    CASE_WINDOWS
    //
    CASE_BATCH_SIZE
)


def move_to_device(obj):

    if torch.is_tensor(obj):
        return obj.to(DEVICE)

    if isinstance(
        obj,
        dict,
    ):
        return {
            k: move_to_device(v)
            for k, v
            in obj.items()
        }

    return obj


def benchmark_model(
    model_name,
    model_class,
    config_path,
):

    print()
    print("=" * 80)
    print(model_name)
    print("=" * 80)

    config = load_config(
        config_path
    )

    dataset = (
        CaseTrainingDataset(
            dataset_root=
            config["dataset"][
                "root_dir"
            ],
            split="train",
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate,
    )

    batch = next(
        iter(loader)
    )

    batch = move_to_device(
        batch
    )

    model = model_class(
        config
    ).to(
        DEVICE
    )

    model.train()

    loss_fn = (
        UnifiedMultiTaskLoss()
    )

    # ----------------------------------
    # Warmup
    # ----------------------------------

    with torch.no_grad():

        _ = model(
            batch
        )

    if DEVICE == "cuda":

        torch.cuda.synchronize()

    # ----------------------------------
    # Forward
    # ----------------------------------

    start = time.perf_counter()

    outputs = model(
        batch
    )

    if DEVICE == "cuda":

        torch.cuda.synchronize()

    forward_time = (
        time.perf_counter()
        - start
    )

    # ----------------------------------
    # Loss
    # ----------------------------------

    start = time.perf_counter()

    losses = loss_fn(
        outputs=outputs,
        recognition_targets=
        batch[
            "recognition_targets"
        ],
        forecast_targets=
        batch[
            "forecast_targets"
        ],
    )

    loss = losses[
        "total_loss"
    ]

    if DEVICE == "cuda":

        torch.cuda.synchronize()

    loss_time = (
        time.perf_counter()
        - start
    )

    # ----------------------------------
    # Backward
    # ----------------------------------

    start = time.perf_counter()

    loss.backward()

    if DEVICE == "cuda":

        torch.cuda.synchronize()

    backward_time = (
        time.perf_counter()
        - start
    )

    total_time = (
        forward_time
        + loss_time
        + backward_time
    )

    print(
        f"Forward : "
        f"{forward_time:.3f} sec"
    )

    print(
        f"Loss    : "
        f"{loss_time:.3f} sec"
    )

    print(
        f"Backward: "
        f"{backward_time:.3f} sec"
    )

    print(
        f"Total   : "
        f"{total_time:.3f} sec"
    )

    print(
        f"Loss    : "
        f"{loss.item():.4f}"
    )

    return total_time


def main():

    print()
    print("=" * 80)
    print("CASE TRAINING SPEED BENCHMARK")
    print("=" * 80)

    print(
        "Device:",
        DEVICE
    )

    results = {}

    results[
        "BaselineB1"
    ] = benchmark_model(
        "BaselineB1",
        BaselineB1,
        "configs/case_b1.yaml",
    )

    results[
        "BaselineB2"
    ] = benchmark_model(
        "BaselineB2",
        BaselineB2,
        "configs/case_b2.yaml",
    )

    results[
        "BaselineB3"
    ] = benchmark_model(
        "BaselineB3",
        BaselineB3,
        "configs/case_b3.yaml",
    )

    results[
        "QSAL"
    ] = benchmark_model(
        "QSAL",
        QSAL,
        "configs/case_qsal.yaml",
    )

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print()

    print(
        f"{'Model':<15}"
        f"{'Sec/Batch':>15}"
        f"{'Min/Epoch':>15}"
        f"{'Hours/Epoch':>15}"
    )

    print("-" * 60)

    for name, batch_time in results.items():

        epoch_time_sec = (
            batch_time
            * BATCHES_PER_EPOCH
        )

        epoch_time_min = (
            epoch_time_sec
            / 60
        )

        epoch_time_hr = (
            epoch_time_sec
            / 3600
        )

        print(
            f"{name:<15}"
            f"{batch_time:>15.3f}"
            f"{epoch_time_min:>15.2f}"
            f"{epoch_time_hr:>15.2f}"
        )

    print()
    print(
        f"CASE Windows: "
        f"{CASE_WINDOWS}"
    )

    print(
        f"Batch Size: "
        f"{CASE_BATCH_SIZE}"
    )

    print(
        f"Batches/Epoch: "
        f"{BATCHES_PER_EPOCH}"
    )


if __name__ == "__main__":
    main()