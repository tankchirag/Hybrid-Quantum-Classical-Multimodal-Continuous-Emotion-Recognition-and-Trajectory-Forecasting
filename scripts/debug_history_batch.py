"""
scripts.debug_history_batch
===========================

Inspect one fully processed training batch.

Purpose
-------
Verify actual tensor shapes after:

ULMTSSTDiskDataset
    ↓
TimelineSynchronizer
    ↓
WindowSampler
    ↓
Collate

This script is used before implementing the
History Encoder architecture.
"""

from pathlib import Path
import sys

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from utils.config import load_config

from training.base_training_script import (
    to_namespace,
    build_dataloaders,
)

from training.base_training_script import (
    get_device,
)


def main():

    config = load_config(
        "configs/qsal.yaml"
    )

    config = to_namespace(
        config
    )

    device = get_device(
        config
    )

    train_loader, _ = (
        build_dataloaders(
            config,
            device,
        )
    )

    batch = next(
        iter(train_loader)
    )

    print("\n")
    print("=" * 80)
    print("BATCH KEYS")
    print("=" * 80)

    for key in batch.keys():
        print(key)

    print("\n")
    print("=" * 80)
    print("CONTINUOUS LABEL HISTORIES")
    print("=" * 80)

    for (
        label_name,
        tensor,
    ) in batch[
        "continuous_labels"
    ].items():

        print(
            f"{label_name:<25}"
            f"{tuple(tensor.shape)}"
        )

    print("\n")
    print("=" * 80)
    print("DISCRETE LABEL HISTORIES")
    print("=" * 80)

    for (
        label_name,
        tensor,
    ) in batch[
        "discrete_labels"
    ].items():

        print(
            f"{label_name:<25}"
            f"{tuple(tensor.shape)}"
        )

    print("\n")
    print("=" * 80)
    print("MODALITIES")
    print("=" * 80)

    for (
        modality_name,
        tensor,
    ) in batch[
        "modalities"
    ].items():

        print(
            f"{modality_name:<25}"
            f"{tuple(tensor.shape)}"
        )

    print("\n")
    print("=" * 80)
    print("RECOGNITION TARGETS")
    print("=" * 80)

    for (
        label_name,
        tensor,
    ) in batch[
        "recognition_targets"
    ]["continuous"].items():

        print(
            f"continuous/{label_name:<15}"
            f"{tuple(tensor.shape)}"
        )

    for (
        label_name,
        tensor,
    ) in batch[
        "recognition_targets"
    ]["discrete"].items():

        print(
            f"discrete/{label_name:<17}"
            f"{tuple(tensor.shape)}"
        )

    print("\n")
    print("=" * 80)
    print("FORECAST TARGETS")
    print("=" * 80)

    for (
        label_name,
        tensor,
    ) in batch[
        "forecast_targets"
    ]["continuous"].items():

        print(
            f"continuous/{label_name:<15}"
            f"{tuple(tensor.shape)}"
        )

    for (
        label_name,
        tensor,
    ) in batch[
        "forecast_targets"
    ]["discrete"].items():

        print(
            f"discrete/{label_name:<17}"
            f"{tuple(tensor.shape)}"
        )

    print("\n")
    print("=" * 80)
    print("ACTIVE LABELS")
    print("=" * 80)

    print(
        "Continuous:",
        batch[
            "active_continuous_labels"
        ]
    )

    print(
        "Discrete:",
        batch[
            "active_discrete_labels"
        ]
    )


if __name__ == "__main__":
    main()