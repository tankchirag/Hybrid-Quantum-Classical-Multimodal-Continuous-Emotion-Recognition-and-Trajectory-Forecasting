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

from models.wrappers.qsal import (
    QSAL,
)


def main():

    config = load_config(
        "configs/case_qsal.yaml"
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

    model = QSAL(
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

    if outputs[
        "history_state"
    ] is not None:

        print(
            "history_state:",
            outputs[
                "history_state"
            ].shape
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

    print()

    print(
        model.summary()
    )


if __name__ == "__main__":
    main()