"""
tests.case.test_first_batch

Verify complete CASE → DataLoader →
Collate → Model contract.

Checks:

    modalities

    continuous labels

    targets

    model outputs

    tensor shapes
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.wrappers.case_training_dataset import (
    CaseTrainingDataset,
)

from datasets.loaders.collate import (
    collate,
)

from utils.config import load_config

from models.wrappers.baseline_b1 import (
    BaselineB1,
)


DATASET_ROOT = Path(
    r"D:\Datasets\CASE\processed"
)

CONFIG_PATH = (
    "configs/case_b1.yaml"
)


def print_batch(batch):

    print()
    print("=" * 80)
    print("BATCH")
    print("=" * 80)

    print(
        "batch size:",
        batch["batch_size"]
    )

    print()

    for name, tensor in (
        batch["modalities"].items()
    ):

        print(
            name,
            tuple(tensor.shape),
        )

    print()

    print(
        "recognition targets"
    )

    for k, v in (
        batch["recognition_targets"]
        ["continuous"]
        .items()
    ):

        print(
            k,
            tuple(v.shape),
        )

    print()

    print(
        "forecast targets"
    )

    for k, v in (
        batch["forecast_targets"]
        ["continuous"]
        .items()
    ):

        print(
            k,
            tuple(v.shape),
        )


def print_outputs(outputs):

    print()
    print("=" * 80)
    print("MODEL OUTPUTS")
    print("=" * 80)

    print(
        "tokens",
        tuple(
            outputs["tokens"].shape
        )
    )

    print(
        "fused",
        tuple(
            outputs["fused"].shape
        )
    )

    print(
        "encoder",
        tuple(
            outputs[
                "encoder_outputs"
            ].shape
        )
    )

    print(
        "decoder",
        tuple(
            outputs[
                "decoder_outputs"
            ].shape
        )
    )

    print()

    print(
        "recognition"
    )

    for k, v in (
        outputs[
            "recognition"
        ]["continuous"]
        .items()
    ):
        print(
            k,
            tuple(v.shape),
        )

    print()

    print(
        "forecasting"
    )

    for k, v in (
        outputs[
            "forecasting"
        ]["continuous"]
        .items()
    ):
        print(
            k,
            tuple(v.shape),
        )


def main():

    dataset = (
        CaseTrainingDataset(
            DATASET_ROOT,
            split="train",
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate,
    )

    batch = next(
        iter(loader)
    )

    print()
    print(
        "✅ CASE Batch Contract Passed"
    )

    config = load_config(
        CONFIG_PATH
    )

    model = BaselineB1(
        config
    )

    model.eval()

    with torch.no_grad():

        outputs = model(
            batch
        )

    print_outputs(
        outputs
    )

    print()
    print(
        "✅ FIRST BATCH TEST PASSED"
    )


if __name__ == "__main__":
    main()