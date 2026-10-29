from torch.utils.data import DataLoader

from utils.config import load_config

from datasets.wrappers.case_training_dataset import (
    CaseTrainingDataset,
)

from datasets.loaders.collate import (
    collate,
)

from models.wrappers.baseline_b1 import (
    BaselineB1,
)

from training.losses import (
    UnifiedMultiTaskLoss,
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
        shuffle=False,
        collate_fn=collate,
    )

    batch = next(iter(loader))

    model = BaselineB1(
        config
    )

    model.train()

    outputs = model(
        batch
    )

    print()
    print("=" * 80)
    print("MODEL FORWARD")
    print("=" * 80)

    print(
        "history enabled:",
        outputs[
            "history_enabled"
        ]
    )

    print(
        "history state shape:",
        outputs[
            "history_state"
        ].shape
    )

    print()

    print(
        "recognition continuous:"
    )

    print(
        outputs[
            "recognition"
        ][
            "continuous"
        ].keys()
    )

    print()

    print(
        "forecast continuous:"
    )

    print(
        outputs[
            "forecasting"
        ][
            "continuous"
        ].keys()
    )

    loss_fn = (
        UnifiedMultiTaskLoss()
    )

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

    print()
    print("=" * 80)
    print("LOSS")
    print("=" * 80)

    print(
        "total loss:",
        float(loss.item())
    )

    print(
        "requires grad:",
        loss.requires_grad
    )

    loss.backward()

    print()

    print(
        "backward pass successful"
    )


if __name__ == "__main__":
    main()