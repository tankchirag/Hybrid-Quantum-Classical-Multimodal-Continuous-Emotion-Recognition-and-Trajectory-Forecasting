import sys
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



from torch.utils.data import DataLoader

from utils.config import load_config

from datasets.wrappers.case_training_dataset import (
    CaseTrainingDataset,
)

from datasets.loaders.collate import (
    collate,
)

from models.wrappers.baseline_b3 import (
    BaselineB3,
)

from training.losses import (
    UnifiedMultiTaskLoss,
)


def main():

    config = load_config(
        "configs/case_b3.yaml"
    )

    dataset = CaseTrainingDataset(
        dataset_root=
        config["dataset"][
            "root_dir"
        ],
        split="train",
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

    model = BaselineB3(
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

    history_state = outputs[
        "history_state"
    ]

    if history_state is None:

        print(
            "history state: disabled"
        )

    else:

        print(
            "history state shape:",
            history_state.shape
        )

    print()

    print(
        "fused:",
        outputs[
            "fused"
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

    print()
    print("=" * 80)
    print("SHAPE DEBUG")
    print("=" * 80)

    print(
        "pred valence recognition:",
        outputs[
            "recognition"
        ][
            "continuous"
        ][
            "valence"
        ].shape
    )

    print(
        "target valence recognition:",
        batch[
            "recognition_targets"
        ][
            "continuous"
        ][
            "valence"
        ].shape
    )

    print()

    print(
        "pred valence forecast:",
        outputs[
            "forecasting"
        ][
            "continuous"
        ][
            "valence"
        ].shape
    )

    print(
        "target valence forecast:",
        batch[
            "forecast_targets"
        ][
            "continuous"
        ][
            "valence"
        ].shape
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
        float(
            loss.item()
        )
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