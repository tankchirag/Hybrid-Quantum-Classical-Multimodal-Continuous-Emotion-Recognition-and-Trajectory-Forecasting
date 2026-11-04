from torch.utils.data import DataLoader

from utils.config import load_config

from datasets.wrappers.case_training_dataset import (
    CaseTrainingDataset,
)

from datasets.loaders.collate import (
    collate,
)

from models.wrappers.baseline_b2 import (
    BaselineB2,
)


def main():

    config = load_config(
        "configs/case_b1.yaml"
    )

    dataset = CaseTrainingDataset(
        dataset_root=
        config["dataset"]["root_dir"],
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate,
    )

    batch = next(iter(loader))

    model = BaselineB2(
        config
    )

    outputs = model(
        batch
    )

    print()

    print(
        "history_enabled:",
        outputs[
            "history_enabled"
        ]
    )

    print(
        "tokens:",
        outputs[
            "tokens"
        ].shape
    )

    print(
        "fused:",
        outputs[
            "fused"
        ].shape
    )

    print()

    print(
        outputs[
            "recognition"
        ][
            "continuous"
        ].keys()
    )

    print(
        outputs[
            "forecasting"
        ][
            "continuous"
        ].keys()
    )


if __name__ == "__main__":
    main()